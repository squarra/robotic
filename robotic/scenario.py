import numpy as np

from robotic._robotic import JT, ST, CameraView, Config, raiPath
from robotic.helpers import compute_look_at_matrix, mask_colors


class Scenario(Config):
    def __init__(self, radius=1, num_views=3, heights=[2.0]):
        super().__init__()
        self.world = self.addFrame("world")

        self.camera = self.addFrame("camera", "world").setAttributes({"focalLength": 0.895, "width": 640.0, "height": 360.0, "zRange": [0.01, 5.0]})
        self.camera_view = CameraView(self)
        self.camera_view.setCamera(self.camera)

        self.camera_positions = self.generate_camera_positions(radius, num_views, heights)
        self.env_colors = self.compute_seg_colors()

    def generate_camera_positions(self, radius: float, num_views: int, heights: list[float]):
        """Generate circular camera positions at given heights, with num_views evenly spaced angles per height."""
        positions = []
        for h in heights:
            for i in range(num_views):
                angle = 2 * np.pi * i / num_views
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                positions.append([x, y, h])
        return positions

    def set_camera(self, position: np.typing.ArrayLike):
        self.camera.setRelativePosition(position).setRotationMatrix(compute_look_at_matrix(self.camera.getPosition(), self.world.getPosition()))

    def compute_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        return self.camera_view.computeImageAndDepth(self)

    def compute_seg_rgb(self, step=64, update_config=False) -> np.ndarray:
        if update_config:
            self.compute_rgbd()  #  need to call this for ConfigurationViewer::updateConfiguration() to be called
        return (self.camera_view.computeSegmentationImage() // step) * step

    def compute_seg_rgb_for_positions(self):
        images = []
        for position in self.camera_positions:
            self.set_camera(position)
            images.append(self.compute_seg_rgb(update_config=True))
        return np.stack(images)

    def compute_seg_colors(self):
        return np.unique(self.compute_seg_rgb_for_positions().reshape(-1, 3), axis=0)

    def __repr__(self):
        return f"{self.__class__.__name__}(frames={len(self.getFrames())}, positions={len(self.camera_positions)})"


class PandaScenario(Scenario):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.table = (
            self.addFrame("table", "world")
            .setShape(ST.ssBox, [2.5, 2.5, 0.1, 0.02])
            .setPosition([0, 0, 0.6])
            .setColor([0.3, 0.3, 0.3])
            .setContact(1)
            .setAttributes({"friction": 0.1, "logical": 0})
        )
        self.addFile(raiPath("panda/panda.g")).setParent(self.table).setRelativePoseByText("t(0 -0.2 0.05) d(90 0 0 1)").setJoint(JT.rigid)

        self.robot_colors = self.compute_seg_colors()

    def compute_object_colors(self):
        all_colors = self.compute_seg_colors()  # (N, 3)
        robot_colors = self.robot_colors  # (M, 3)
        all_int = (all_colors[:, 0].astype(np.uint32) << 16) | (all_colors[:, 1].astype(np.uint32) << 8) | all_colors[:, 2].astype(np.uint32)
        robot_int = (robot_colors[:, 0].astype(np.uint32) << 16) | (robot_colors[:, 1].astype(np.uint32) << 8) | robot_colors[:, 2].astype(np.uint32)
        return all_colors[~np.isin(all_int, robot_int)]

    def capture_masked_object_images(self):
        images = []
        for pos in self.camera_positions:
            self.set_camera(pos)
            rgb, _ = self.compute_rgbd()
            seg_rgb = self.compute_seg_rgb()
            mask_env = mask_colors(seg_rgb, self.env_colors)
            mask_robot = mask_colors(seg_rgb, self.robot_colors)
            rgb[(mask_env | mask_robot)] = 0
            images.append(rgb)
        return images
