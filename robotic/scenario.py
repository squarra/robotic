import numpy as np

from robotic._robotic import JT, ST, CameraView, Config, raiPath
from robotic.helpers import DEBUG, compute_look_at_quat, generate_circular_camera_positions, matrix_to_quat, random_z_rotation_matrix


class Scenario(Config):
    def __init__(self):
        super().__init__()
        self.world = self.addFrame("world")

        self.cam = self.addFrame("camera", "world").setAttributes({"focalLength": 0.895, "width": 640.0, "height": 360.0, "zRange": [0.01, 5.0]})
        self.cam_view = CameraView(self)
        self.cam_view.setCamera(self.cam)
        self._cam_poses = []

        self.env_frames = set(self.getFrameNames())

    @property
    def cam_poses(self):
        return np.asarray(self._cam_poses, dtype=np.float32)

    def add_cam_pose(self, pose: np.typing.ArrayLike):
        self._cam_poses.append(pose)

    def add_circular_cam_poses(self, radius=1.0, num_views=3, heights=[1.0]):
        for pos in generate_circular_camera_positions(radius, num_views, heights):
            quat = compute_look_at_quat(pos, [0, 0, 0])
            self.add_cam_pose([*pos, *quat])

    def add_topdown_cam(self, height=1.5):
        self.add_cam_pose([0, 0, height, 0, 0, 1, 0])

    def add_marker(self, pose: np.typing.ArrayLike):
        return self.addFrame("marker", "table").setShape(ST.marker, [0.1]).setRelativePose(pose)

    def add_markers(self):
        for i, obj in enumerate(self.man_frames):
            self.addFrame(f"marker{i}", obj).setShape(ST.marker, [0.2])

    def remove_markers(self):
        for frame in self.getFrameNames():
            if "marker" in frame:
                self.delFrame(frame)

    def set_cam_pose(self, pose: np.typing.ArrayLike):
        self.cam.setRelativePose(pose)

    def compute_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        image, depth = self.cam_view.computeImageAndDepth(self)
        return image.astype(np.float32), depth.astype(np.float32)

    def compute_image(self):
        return self.compute_rgbd()[0]

    def compute_depth(self):
        return self.compute_rgbd()[1]

    def compute_seg_rgb(self) -> np.ndarray:
        """RGB-encoded frame IDs; IDs >= len(self.getFrames()) are background."""
        return self.cam_view.computeSegmentationImage(self)

    def compute_seg_ids(self) -> np.ndarray:
        """Frame IDs per pixel; IDs >= len(self.getFrames()) are background."""
        return self.cam_view.computeSegmentationID(self)

    def delete_man_frames(self):
        for man_frame in self.man_frames:
            self.delFrame(man_frame)

    def compute_images_and_seg_ids(self):
        images, seg_ids = [], []
        for pose in self.cam_poses:
            self.set_cam_pose(pose)
            images.append(self.compute_image())
            seg_ids.append(self.compute_seg_ids())
        return np.stack(images), np.stack(seg_ids)

    def compute_depths_and_seg_ids(self):
        depths, seg_ids = [], []
        for pose in self.cam_poses:
            self.set_cam_pose(pose)
            depths.append(self.compute_depth())
            seg_ids.append(self.compute_seg_ids())
        return np.stack(depths), np.stack(seg_ids)

    def compute_images_depths_and_seg_ids(self):
        images, depths, seg_ids = [], [], []
        for pose in self.cam_poses:
            self.set_cam_pose(pose)
            image, depth = self.compute_rgbd()
            images.append(image)
            depths.append(depth)
            seg_ids.append(self.compute_seg_ids())
        return np.stack(images), np.stack(depths), np.stack(seg_ids)

    def compute_collisions(self):
        return self.getCollisions(verbose=DEBUG.value)

    @property
    def man_frames(self):
        return sorted(set(self.getFrameNames()) - self.env_frames)

    def __repr__(self):
        return f"{self.__class__.__name__}(env_frames={len(self.env_frames)}, man_frames={len(self.man_frames)}, poses={len(self.cam_poses)})"


class PandaScenario(Scenario):
    def __init__(self, add_circular_cam_poses=True):
        super().__init__()

        self.table = (
            self.addFrame("table", "world")
            .setShape(ST.ssBox, [2.0, 2.0, 0.1, 0.02])
            .setColor([0.3, 0.3, 0.3])
            .setContact(1)
            .setAttributes({"friction": 0.1, "logical": 0})
        )
        self.addFile(raiPath("panda/panda.g")).setParent(self.table).setRelativePoseByText("t(0 -0.2 0.05) d(90 0 0 1)").setJoint(JT.rigid)

        self.env_frames = set(self.getFrameNames())

        if add_circular_cam_poses:
            self.add_circular_cam_poses()

    def add_box(self, name: str, size: np.typing.ArrayLike, pos: np.typing.ArrayLike):
        return self.addFrame(name, "table").setJoint(JT.rigid).setShape(ST.ssBox, size).setRelativePosition([pos]).setContact(1)

    def add_boxes(
        self, num_boxes_range=(2, 12), box_size_range=(0.04, 0.12), xy_range=((-0.5, 0.5), (-0.5, 0.5)), density=500.0, seed=None, max_tries=100
    ):
        rng = np.random.default_rng(seed)
        n_objects = rng.integers(*num_boxes_range)

        table_z = self.table.getSize()[2]

        for i in range(n_objects):
            size = rng.uniform(*box_size_range, size=3)
            quat = matrix_to_quat(random_z_rotation_matrix(rng))
            mass = density * np.prod(size)

            box = (
                self.addFrame(f"box{i}", "table")
                .setJoint(JT.rigid)
                .setShape(ST.ssBox, [*size, 0.005])
                .setRelativeQuaternion(quat)
                .setContact(1)
                .setMass(mass)
            )

            placed = False
            for _ in range(max_tries):
                x = rng.uniform(*xy_range[0])
                y = rng.uniform(*xy_range[1])
                z = table_z / 2 + size[2] / 2 + np.finfo(np.float32).eps

                box.setRelativePosition([x, y, z])
                box.ensure_X()

                if not self.compute_collisions():
                    placed = True
                    break

            if not placed:
                self.delFrame(box.name)

    def sample_target_pos(self, obj: str, direction: np.typing.ArrayLike, offset_range=(0.15, 0.3), max_tries=20, seed=None):
        rng = np.random.default_rng(seed)

        frame = self.getFrame(obj)
        original_pos = frame.getRelativePosition()

        for _ in range(max_tries):
            offset = np.asarray(direction) * rng.uniform(*offset_range)
            offset_world = frame.getRotationMatrix() @ offset
            target_pos = frame.getRelativePosition() + offset_world

            frame.setRelativePosition(target_pos)
            frame.ensure_X()
            collisions = self.compute_collisions()

            frame.setRelativePosition(original_pos)  # always restore!

            if not collisions:
                return target_pos

        if DEBUG > 0:
            print("no collision free offset found")
        return original_pos
