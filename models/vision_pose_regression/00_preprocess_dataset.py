import h5py
import numpy as np
from tqdm import tqdm

from models.vision_pose_regression import DATASET_PATH

SOURCE_H5_PATH = "dataset.h5"


def quat_mul(q1, q2):
    w1, x1, y1, z1 = np.split(q1, 4, axis=-1)
    w2, x2, y2, z2 = np.split(q2, 4, axis=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.concatenate([w, x, y, z], axis=-1)


def quat_inv(q):
    w, x, y, z = np.split(q, 4, axis=-1)
    return np.concatenate([w, -x, -y, -z], axis=-1)


def compute_pose_diff(pose, final_pose):
    pos_i, rot_i = pose[..., :3], pose[..., 3:]
    pos_f, rot_f = final_pose[..., :3], final_pose[..., 3:]

    d_pos = pos_f - pos_i
    rot_delta = quat_mul(rot_f, quat_inv(rot_i))

    rot_delta /= np.linalg.norm(rot_delta, axis=-1, keepdims=True)
    return np.concatenate([d_pos, rot_delta], axis=-1)


with h5py.File(SOURCE_H5_PATH, "r") as f_in, h5py.File(DATASET_PATH, "w") as f_out:
    f_out.attrs.update(f_in.attrs)

    for dp_key in tqdm(f_in.keys(), desc="Processing scenes"):
        dp = f_in[dp_key]

        depths = dp["depths"][()]
        cam_poses = dp["cam_poses"][()]
        seg_ids = dp["seg_ids"][()]

        poses = dp["poses"][()]
        final_poses = dp["final_poses"][()]

        num_objects = poses.shape[0]
        for oi in range(num_objects):
            pose_diff = compute_pose_diff(np.expand_dims(poses[oi], axis=0), final_poses[oi])

            dp_group = f_out.create_group(f"{dp_key}_obj_{oi}")
            dp_group.create_dataset("pose", data=poses[oi])
            dp_group.create_dataset("cam_poses", data=cam_poses)
            dp_group.create_dataset("depths", data=depths)
            dp_group.create_dataset("masks", data=(seg_ids == oi).astype(np.float32))
            dp_group.create_dataset("target_pose", data=dp["target_poses"][oi])
            dp_group.create_dataset("feasibles", data=dp["feasibles"][oi])
            dp_group.create_dataset("pose_diffs", data=pose_diff)
