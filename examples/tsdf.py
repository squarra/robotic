import numpy as np
import open3d as o3d

from robotic._robotic import ST
from robotic.scenario import PandaScenario


def reconstruct_tsdf(scenario: PandaScenario):
    fx, fy, cx, cy = scenario.camera_view.getFxycxy()
    width = int(scenario.camera.getAttributes()["width"])
    height = int(scenario.camera.getAttributes()["height"])
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.005, sdf_trunc=0.01, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for pos in scenario.camera_positions:
        scenario.set_camera(pos)
        rgb, depth = scenario.compute_rgbd()
        o3d_rgb = o3d.geometry.Image(rgb)
        o3d_depth = o3d.geometry.Image(depth * 1000)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_rgb, o3d_depth, depth_scale=1000.0, depth_trunc=2.0, convert_rgb_to_intensity=False
        )
        extrinsic = np.linalg.inv(scenario.camera.getTransform())
        volume.integrate(rgbd, intrinsic, extrinsic)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


config = PandaScenario(heights=[1.5, 2.0])
config.addFrame("box", "table").setShape(ST.ssBox, [0.2, 0.2, 0.1, 0.02]).setRelativePosition([0.4, 0.4, 0.08])
config.addFrame("box2", "table").setShape(ST.ssBox, [0.2, 0.2, 0.1, 0.02]).setRelativePosition([0.1, 0.1, 0.08])
config.addFrame("box3", "table").setShape(ST.ssBox, [0.2, 0.2, 0.1, 0.02]).setRelativePosition([0.4, 0.1, 0.08])
config.addFrame("box4", "table").setShape(ST.ssBox, [0.2, 0.2, 0.1, 0.02]).setRelativePosition([0.1, 0.4, 0.08])
mesh = reconstruct_tsdf(config)
o3d.visualization.draw_geometries([mesh])
