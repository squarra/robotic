import h5py
import numpy as np
from tqdm import tqdm

from models.vision_pose import DATASET_PATH

SOURCE_H5_PATH = "dataset.h5"


with h5py.File(SOURCE_H5_PATH, "r") as f_in, h5py.File(DATASET_PATH, "w") as f_out:
    f_out.attrs.update(f_in.attrs)

    for dp_key in tqdm(f_in.keys(), desc="Processing scenes"):
        dp = f_in[dp_key]

        depths = dp["depths"][()]
        cam_poses = dp["cam_poses"][()]
        seg_ids = dp["seg_ids"][()]

        num_objects = dp["poses"].shape[0]
        for oi in range(num_objects):
            dp_group = f_out.create_group(f"{dp_key}_obj_{oi}")
            dp_group.create_dataset("pose", data=dp["poses"][oi])
            dp_group.create_dataset("cam_poses", data=cam_poses)
            dp_group.create_dataset("depths", data=depths)
            dp_group.create_dataset("masks", data=(seg_ids == oi).astype(np.float32))
            dp_group.create_dataset("target_pose", data=dp["target_poses"][oi])
            dp_group.create_dataset("feasibles", data=dp["feasibles"][oi].astype(np.float32))
