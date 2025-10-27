import numpy as np

from robotic._robotic import JT, ST, CameraView, Config, raiPath
from robotic.helpers import DEBUG, compute_look_at_quat, generate_circular_camera_positions, matrix_to_quat, random_z_rotation_matrix


class Scenario(Config):
    def __init__(self):
        super().__init__()
        self.world = self.addFrame("world")

        self.cam = self.addFrame("camera", "world").setAttributes({"focalLength": 1.0, "width": 384.0, "height": 256.0, "zRange": [0.01, 5.0]})
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

    def add_topdown_cam(self):
        self.add_cam_pose([0, 0.7, 0.7, 0, 0, 0.954, -0.301])

    def add_marker(self, name: str, pose: np.typing.ArrayLike):
        return self.addFrame(f"{name}_marker", "table").setShape(ST.marker, [0.1]).setRelativePose(pose)

    def add_markers(self):
        for i, obj in enumerate(self.man_frames):
            self.addFrame(f"marker{i}", obj).setShape(ST.marker, [0.2])

    def remove_markers(self):
        for frame in self.getFrameNames():
            if "marker" in frame:
                self.delFrame(frame)

    def set_cam_pose(self, pose: np.typing.ArrayLike):
        self.cam.setRelativePose(pose)

    def delete_man_frames(self):
        for man_frame in self.man_frames:
            self.delFrame(man_frame)

    def compute_images_depths_and_seg_ids(self):
        images, depths, seg_ids = [], [], []
        for pose in self.cam_poses:
            self.set_cam_pose(pose)
            cam_view = CameraView(self)
            cam_view.setCamera(self.cam)
            image, depth = cam_view.computeImageAndDepth(self)
            images.append(image.astype(np.float32))
            depths.append(depth.astype(np.float32))
            seg_ids.append(cam_view.computeSegmentationID(self))
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
        panda_base_f = (
            self.addFile(raiPath("panda/panda.g")).setParent(self.table).setRelativePoseByText("t(0 -0.2 0.05) d(90 0 0 1)").setJoint(JT.rigid)
        )
        self.addFrame("panda_safety", panda_base_f.name).setShape(ST.ssCylinder, [0.1, 0.3, 0.05]).setColor([1.0, 1.0, 1.0, 0.1]).setContact(-1)

        self.env_frames = set(self.getFrameNames())

        if add_circular_cam_poses:
            self.add_circular_cam_poses(radius=0.75)

    def add_box(self, name: str, size: np.typing.ArrayLike, pos: np.typing.ArrayLike):
        return self.addFrame(name, "table").setJoint(JT.rigid).setShape(ST.ssBox, size).setRelativePosition([pos]).setContact(1)

    def add_boxes(self, num_boxes_range=(2, 8), box_size_range=(0.04, 0.10), radius=0.5, density=500.0, seed=None, max_tries=100):
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
                angle = rng.uniform(0, np.pi)
                r = np.sqrt(rng.uniform(0, 1)) * radius
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                z = (0.5 * table_z) + (0.5 * size[2]) + np.finfo(np.float32).eps

                box.setRelativePosition([x, y, z])
                box.ensure_X()

                collisions = self.compute_collisions()
                if collisions and DEBUG > 0:
                    print(f"Collisions while adding box to scene: {collisions}")
                if not collisions:
                    placed = True
                    break

            if not placed:
                self.delFrame(box.name)

    def sample_target_pose(
        self, obj: str, directions=((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)), offset_range=(0.15, 0.3), max_tries=20, seed=None
    ):
        rng = np.random.default_rng(seed)
        directions = rng.permutation(directions)

        frame = self.getFrame(obj)
        original_pos = frame.getRelativePosition()
        original_quat = frame.getRelativeQuaternion()

        for direction in directions:
            for _ in range(max_tries):
                offset = np.asarray(direction) * rng.uniform(*offset_range)
                offset_world = frame.getRotationMatrix() @ offset
                target_pos = frame.getRelativePosition() + offset_world

                frame.setRelativePosition(target_pos)
                frame.ensure_X()
                collisions = self.compute_collisions()

                # Always restore!
                frame.setRelativePosition(original_pos)

                if not collisions:
                    return np.concatenate([target_pos, original_quat])

        if DEBUG > 0:
            print("no collision free target pose found")
        return np.concatenate([original_pos, original_quat])
