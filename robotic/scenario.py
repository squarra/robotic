import numpy as np

from robotic._robotic import JT, ST, CameraView, Config, raiPath
from robotic.helpers import DEBUG, compute_look_at_matrix, generate_circular_camera_positions, rgb_to_gray


class Scenario(Config):
    def __init__(self, camera_positions):
        super().__init__()
        self.world = self.addFrame("world")

        self.camera = self.addFrame("camera", "world").setAttributes({"focalLength": 0.895, "width": 640.0, "height": 360.0, "zRange": [0.01, 5.0]})
        self.camera_view = CameraView(self)
        self.camera_view.setCamera(self.camera)

        self.camera_positions = camera_positions
        self.env_frames = set(self.getFrameNames())

    def set_camera(self, position: np.typing.ArrayLike):
        self.camera.setRelativePosition(position).setRotationMatrix(compute_look_at_matrix(self.camera.getPosition(), self.world.getPosition()))

    def compute_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        return self.camera_view.computeImageAndDepth(self)

    def compute_image(self, grayscale=False):
        rgb = self.compute_rgbd()[0]
        return rgb_to_gray(rgb) if grayscale else rgb

    def compute_seg_rgb(self) -> np.ndarray:
        """RGB-encoded frame IDs; IDs >= len(self.getFrames()) are background."""
        return self.camera_view.computeSegmentationImage(self)

    def compute_seg_ids(self) -> np.ndarray:
        """Frame IDs per pixel; IDs >= len(self.getFrames()) are background."""
        return self.camera_view.computeSegmentationID(self)

    def delete_man_frames(self):
        for man_frame in self.man_frames:
            self.delFrame(man_frame)

    def compute_images_and_seg_ids(self, grayscale=False) -> np.ndarray:
        images = []
        seg_ids = []
        for pos in self.camera_positions:
            self.set_camera(pos)
            images.append(self.compute_image(grayscale))
            seg_ids.append(self.compute_seg_ids())
        return np.stack(images), np.stack(seg_ids)

    def compute_collisions(self):
        return self.getCollisions(verbose=DEBUG.value)

    @property
    def man_frames(self):
        return set(self.getFrameNames()) - self.env_frames

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(env_frames={len(self.env_frames)}, man_frames={len(self.man_frames)}, positions={len(self.camera_positions)})"
        )


class PandaScenario(Scenario):
    def __init__(self):
        super().__init__(generate_circular_camera_positions(1.0, 3, [2.0]))
        self.table = (
            self.addFrame("table", "world")
            .setShape(ST.ssBox, [1.5, 1.5, 0.1, 0.02])
            .setColor([0.3, 0.3, 0.3])
            .setContact(1)
            .setAttributes({"friction": 0.1, "logical": 0})
        )
        self.addFile(raiPath("panda/panda.g")).setParent(self.table).setRelativePoseByText("t(0 -0.2 0.05) d(90 0 0 1)").setJoint(JT.rigid)

        self.env_frames = set(self.getFrameNames())

    def create_random_scene(self, n_objects_range=(2, 6), size_range=(0.02, 0.16), seed=None, max_tries=100):
        rng = np.random.default_rng(seed)

        n_objects = rng.integers(*n_objects_range)
        table_x, table_y, table_z = self.table.getSize()[:3]
        half_table_x = table_x / 2.0
        half_table_y = table_y / 2.0

        for i in range(n_objects):
            size = rng.uniform(*size_range, size=3)
            up_axis = rng.integers(0, 3)
            perm = [ax for ax in range(3) if ax != up_axis] + [up_axis]
            a, b, height = size[perm]
            box = self.addFrame(f"box{i}", "table").setJoint(JT.rigid).setShape(ST.ssBox, [a, b, height, 0.005]).setContact(1)
            yaw = rng.uniform(-np.pi, np.pi)
            placed = False
            for _ in range(max_tries):
                ca, sa = np.cos(yaw), np.sin(yaw)
                half_extent_x = 0.5 * (abs(a * ca) + abs(b * sa))
                half_extent_y = 0.5 * (abs(a * sa) + abs(b * ca))

                x = rng.uniform(-half_table_x + half_extent_x, half_table_x - half_extent_x)
                y = rng.uniform(-half_table_y + half_extent_y, half_table_y - half_extent_y)
                z = table_z / 2.0 + height / 2.0 + np.finfo(np.float32).eps
                box.setRelativePosition([x, y, z])

                half = 0.5 * yaw
                box.setRelativeQuaternion([np.cos(half), 0.0, 0.0, np.sin(half)])

                box.ensure_X()
                if not self.compute_collisions():
                    placed = True
                    break

            if not placed:
                self.delFrame(box.name)
