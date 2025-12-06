import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from models.easy import DATASET_PATH

SOURCE_H5_PATH = "dataset.h5"
INTRINSICS = {"focalLength": 1.0, "width": 420.0, "height": 360.0, "zRange": [0.01, 2.0]}


def quat_to_matrix(quaternion: np.typing.ArrayLike):
    return R.from_quat(np.roll(quaternion, -1)).as_matrix()


def project_point(world_point, cam_pose):
    t = cam_pose[:3]
    rot = quat_to_matrix(cam_pose[3:])
    p_cam = rot.T @ (world_point - t)
    f, w, h = INTRINSICS["focalLength"], INTRINSICS["width"], INTRINSICS["height"]
    u = (f * p_cam[0] / p_cam[2]) * (w / 2) + w / 2
    v = (f * p_cam[1] / p_cam[2]) * (h / 2) + h / 2
    return int(round(u)), int(round(v))


def make_gaussian_map(width, height, center, sigma=8):
    if center is None:
        return np.zeros((height, width), dtype=np.float32)
    u, v = center
    xs = np.arange(width)
    ys = np.arange(height)
    X, Y = np.meshgrid(xs, ys)
    g = np.exp(-((X - u) ** 2 + (Y - v) ** 2) / (2 * sigma**2))
    g /= g.max() + 1e-8
    return g.astype(np.float32)


def compute_pose_scalar_diffs(pose, final_pose):
    pos_i, rot_i = pose[..., :3], pose[..., 3:]
    pos_f, rot_f = final_pose[..., :3], final_pose[..., 3:]

    pos_diff = np.linalg.norm(pos_f - pos_i).astype(np.float32)

    rot_i = rot_i / np.linalg.norm(rot_i)
    rot_f = rot_f / np.linalg.norm(rot_f)

    dot = np.clip(np.abs(np.sum(rot_i * rot_f)), -1.0, 1.0)
    rot_diff = (2.0 * np.arccos(dot)).astype(np.float32)

    return np.array([pos_diff, rot_diff], dtype=np.float32)


with h5py.File(SOURCE_H5_PATH, "r") as f_in, h5py.File(DATASET_PATH, "w") as f_out:
    f_out.attrs.update(f_in.attrs)

    for dp_key in tqdm(f_in.keys(), desc="Processing scenes"):
        dp = f_in[dp_key]

        depths = dp["depths"][()]
        cam_poses = dp["cam_poses"][()]
        obj_ids = dp["obj_ids"][()]
        seg_ids = dp["seg_ids"][()]
        poses = dp["poses"][()]
        final_poses = dp["final_poses"][()]

        num_objects = poses.shape[0]
        for oi in range(num_objects):
            target_pose = dp["target_poses"][oi]
            pose_diffs = compute_pose_scalar_diffs(final_poses[oi], target_pose)

            target_pos = target_pose[:3]
            num_views = cam_poses.shape[0]
            goal_maps = np.zeros_like(depths, dtype=np.float32)
            for vi in range(num_views):
                uv = project_point(target_pos, cam_poses[vi])
                goal_maps[vi] = make_gaussian_map(int(INTRINSICS["width"]), int(INTRINSICS["height"]), uv, sigma=6)

            dp_group = f_out.create_group(f"{dp_key}_obj_{oi}")
            dp_group.create_dataset("quat", data=poses[oi][3:])
            dp_group.create_dataset("depths", data=depths)
            dp_group.create_dataset("masks", data=(seg_ids == obj_ids[oi]).astype(np.float32))
            dp_group.create_dataset("goal_maps", data=goal_maps)
            dp_group.create_dataset("feasibles", data=dp["feasibles"][oi])
            dp_group.create_dataset("pose_diffs", data=pose_diffs)
