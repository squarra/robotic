import numpy as np
import open3d as o3d

from robotic._robotic import ST
from robotic.helpers import mask_colors
from robotic.scenario import PandaScenario


def reconstruct_tsdf(scenario: PandaScenario, positions: list):
    fx, fy, cx, cy = scenario.camera_view.getFxycxy()
    width = int(scenario.camera.getAttributes()["width"])
    height = int(scenario.camera.getAttributes()["height"])
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    env_colors = scenario.env_colors
    robot_colors = scenario.robot_colors
    object_colors = scenario.compute_object_colors()

    volumes = [
        o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.005, sdf_trunc=0.01, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        for _ in range(2 + len(object_colors))  # environment, robot_support, ...objects
    ]

    for pos in positions:
        scenario.set_camera(pos)
        rgb, depth = scenario.compute_rgbd()
        seg_rgb = scenario.compute_seg_rgb(update_config=True)

        extrinsic = np.linalg.inv(scenario.camera.getTransform())

        mask_env = mask_colors(seg_rgb, env_colors)
        rgb_env = rgb.copy()
        rgb_env[~mask_env] = 0
        depth_env = depth.copy()
        depth_env[~mask_env] = 0
        rgbd_env = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_env), o3d.geometry.Image(depth_env * 1000), depth_scale=1000.0, depth_trunc=2.0, convert_rgb_to_intensity=False
        )
        volumes[0].integrate(rgbd_env, intrinsic, extrinsic)

        mask_robot = mask_colors(seg_rgb, robot_colors)
        rgb_robot = rgb.copy()
        rgb_robot[~mask_robot] = 0
        depth_robot = depth.copy()
        depth_robot[~mask_robot] = 0
        rgbd_robot = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_robot), o3d.geometry.Image(depth_robot * 1000), depth_scale=1000.0, depth_trunc=2.0, convert_rgb_to_intensity=False
        )
        volumes[1].integrate(rgbd_robot, intrinsic, extrinsic)

        for i, color in enumerate(object_colors):
            mask_obj = np.all(seg_rgb == color, axis=-1)
            rgb_obj = rgb.copy()
            rgb_obj[~mask_obj] = 0
            depth_obj = depth.copy()
            depth_obj[~mask_obj] = 0
            rgbd_obj = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb_obj), o3d.geometry.Image(depth_obj * 1000), depth_scale=1000.0, depth_trunc=2.0, convert_rgb_to_intensity=False
            )
            volumes[2 + i].integrate(rgbd_obj, intrinsic, extrinsic)

    return [vol.extract_triangle_mesh().compute_vertex_normals() for vol in volumes]


config = PandaScenario(heights=[1.5, 2.0])
config.addFrame("box", "table").setShape(ST.ssBox, [0.2, 0.2, 0.1, 0.02]).setRelativePosition([0.4, 0.4, 0.08])
config.addFrame("box2", "table").setShape(ST.ssBox, [0.2, 0.2, 0.1, 0.02]).setRelativePosition([0.1, 0.1, 0.08])
config.addFrame("box3", "table").setShape(ST.ssBox, [0.2, 0.2, 0.1, 0.02]).setRelativePosition([0.4, 0.1, 0.08])
config.addFrame("box4", "table").setShape(ST.ssBox, [0.2, 0.2, 0.1, 0.02]).setRelativePosition([0.1, 0.4, 0.08])
meshes = reconstruct_tsdf(config, config.camera_positions)
for mesh in meshes:
    o3d.visualization.draw_geometries([mesh])
o3d.visualization.draw_geometries(meshes)
