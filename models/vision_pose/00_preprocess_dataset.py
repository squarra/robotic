import h5py
import numpy as np
from tqdm import tqdm

SOURCE_H5_PATH = "dataset.h5"
DEST_H5_PATH = "vision_pose_dataset.h5"


with h5py.File(SOURCE_H5_PATH, "r") as f_in, h5py.File(DEST_H5_PATH, "w") as f_out:
    f_out.attrs.update(f_in.attrs)

    for dp_key in tqdm(f_in.keys(), desc="Processing scenes"):
        scene_group = f_in[dp_key]

        depths = scene_group["depths"][()]
        cam_pos = scene_group["camera_positions"][()]
        seg_ids = scene_group["seg_ids"][()]

        num_objects = scene_group["poses"].shape[0]

        for oi in range(num_objects):
            new_group = f_out.create_group(f"{dp_key}_obj_{oi}")

            new_group.create_dataset("depths", data=depths)
            new_group.create_dataset("camera_positions", data=cam_pos)

            new_group.create_dataset("poses", data=scene_group["poses"][oi])
            new_group.create_dataset("feasibles", data=scene_group["feasibles"][oi])

            new_group.create_dataset("masks", data=(seg_ids == oi).astype(np.float32))
