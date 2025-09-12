import numpy as np

from robotic._robotic import JT, ST, CameraView, Config, raiPath
from robotic.helpers import (
    DEBUG,
    compute_look_at_matrix,
    generate_circular_camera_positions,
    matrix_to_quat,
    random_z_rotation_matrix,
    rgb_to_gray,
    rotation_matrices_for_up,
)


class Scenario(Config):
    def __init__(self, camera_positions):
        super().__init__()
        self.world = self.addFrame("world")

        self.camera = self.addFrame("camera", "world").setAttributes({"focalLength": 0.895, "width": 640.0, "height": 360.0, "zRange": [0.01, 5.0]})
        self.camera_view = CameraView(self)
        self.camera_view.setCamera(self.camera)

        self.camera_positions = camera_positions
        self.env_frames = set(self.getFrameNames())

    def add_markers(self):
        for i, obj in enumerate(self.man_frames):
            self.addFrame(f"marker{i}", obj).setShape(ST.marker, [0.2])

    def remove_markers(self):
        for frame in self.getFrameNames():
            if "marker" in frame:
                self.delFrame(frame)

    def set_camera(self, position: np.typing.ArrayLike):
        self.camera.setRelativePosition(position).setRotationMatrix(compute_look_at_matrix(self.camera.getPosition(), self.world.getPosition()))

    def compute_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        return self.camera_view.computeImageAndDepth(self)

    def compute_image(self, grayscale=False):
        rgb = self.compute_rgbd()[0]
        return rgb_to_gray(rgb) if grayscale else rgb

    def compute_depth(self):
        return self.compute_rgbd()[1]

    def compute_seg_rgb(self) -> np.ndarray:
        """RGB-encoded frame IDs; IDs >= len(self.getFrames()) are background."""
        return self.camera_view.computeSegmentationImage(self)

    def compute_seg_ids(self) -> np.ndarray:
        """Frame IDs per pixel; IDs >= len(self.getFrames()) are background."""
        return self.camera_view.computeSegmentationID(self)

    def delete_man_frames(self):
        for man_frame in self.man_frames:
            self.delFrame(man_frame)

    def compute_images_and_seg_ids(self, grayscale=False):
        images, seg_ids = [], []
        for pos in self.camera_positions:
            self.set_camera(pos)
            images.append(self.compute_image(grayscale))
            seg_ids.append(self.compute_seg_ids())
        return np.stack(images), np.stack(seg_ids)

    def compute_depths_and_seg_ids(self):
        depths, seg_ids = [], []
        for pos in self.camera_positions:
            self.set_camera(pos)
            depths.append(self.compute_depth())
            seg_ids.append(self.compute_seg_ids())
        return np.stack(depths), np.stack(seg_ids)

    def compute_images_depths_and_seg_ids(self):
        images, depths, seg_ids = [], [], []
        for pos in self.camera_positions:
            self.set_camera(pos)
            image, depth = self.compute_rgbd()
            images.append(image)
            depths.append(depth)
            seg_ids.append(self.compute_seg_ids())
        return np.stack(images), np.stack(depths), np.stack(seg_ids)

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
            .setShape(ST.ssBox, [1.2, 1.2, 0.1, 0.02])
            .setColor([0.3, 0.3, 0.3])
            .setContact(1)
            .setAttributes({"friction": 0.1, "logical": 0})
        )
        self.addFile(raiPath("panda/panda.g")).setParent(self.table).setRelativePoseByText("t(0 -0.2 0.05) d(90 0 0 1)").setJoint(JT.rigid)

        self.env_frames = set(self.getFrameNames())

    def add_topdown_camera(self, height=1.5):
        table_pos = self.table.getPosition()
        cam_pos = np.array([table_pos[0], table_pos[1], table_pos[2] + height])
        self.camera_positions = np.vstack([self.camera_positions, cam_pos])

    def add_box(self, name: str, size: np.typing.ArrayLike, pos: np.typing.ArrayLike):
        return self.addFrame(name, "table").setJoint(JT.rigid).setShape(ST.ssBox, size).setRelativePosition([pos]).setContact(1)

    def add_boxes_to_scene(self, num_boxes_range=(2, 12), box_size_range=(0.02, 0.08), seed=None, max_tries=100):
        rng = np.random.default_rng(seed)
        n_objects = rng.integers(*num_boxes_range)

        # Keep objects within table bounds
        table_x, table_y, table_z = self.table.getSize()[:3]
        max_pos_x = table_x / 2 - box_size_range[1] / 2
        max_pos_y = table_y / 2 - box_size_range[1] / 2

        for i in range(n_objects):
            size = rng.uniform(*box_size_range, size=(3))
            rot = random_z_rotation_matrix(rng)

            # Randomize orientation (up-axis + z-rotation) and reorder size to match
            up_rot = rng.choice(rotation_matrices_for_up())
            quat = matrix_to_quat(rot @ up_rot)
            permuted_size = size[np.argmax(np.abs(up_rot), axis=0)]

            box = (
                self.addFrame(f"box{i}", "table")
                .setJoint(JT.rigid)
                .setShape(ST.ssBox, [*permuted_size, 0.005])
                .setRelativeQuaternion(quat)
                .setContact(1)
            )

            placed = False
            for _ in range(max_tries):
                x = rng.uniform(-max_pos_x, max_pos_x)
                y = rng.uniform(-max_pos_y, max_pos_y)
                z = table_z / 2 + size[2] / 2 + np.finfo(np.float32).eps
                box.setRelativePosition([x, y, z])
                box.ensure_X()
                if not self.compute_collisions():
                    placed = True
                    break
            if not placed:
                self.delFrame(box.name)
