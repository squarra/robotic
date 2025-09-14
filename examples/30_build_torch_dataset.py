import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

DATASET_PATH = "dataset.h5"


class PoseSizeDataset(Dataset):
    """pose and size concatenated into one tensor of shape (10)"""

    def __init__(self):
        self.index = []
        with h5py.File(DATASET_PATH, "r") as f:
            self.primitives = list(f.attrs["primitives"])

            for dp_key in f.keys():
                objects_group = f[dp_key]["objects"]
                for obj_key in objects_group.keys():
                    self.index.append((dp_key, obj_key))
        print(self.index[0], self.index[1])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        dp_key, obj_key = self.index[idx]

        with h5py.File(DATASET_PATH, "r") as f:
            obj_group = f[dp_key]["objects"][obj_key]

            pose = obj_group["pose"][()]  # shape (7,)
            size = obj_group["size"][()]  # shape (3,)
            x = np.concatenate([pose, size]).astype(np.float32)

            feasibles = [int(obj_group["primitives"][prim]["feasible"][()]) for prim in self.primitives]
            y = np.array(feasibles, dtype=np.float32)

        return torch.from_numpy(x), torch.from_numpy(y)


class DepthSegPoseDataset(Dataset):
    """one depth view, binary segmentation mask and obj pose as three different tensors"""

    def __init__(self):
        self.index = []
        with h5py.File(DATASET_PATH, "r") as f:
            self.primitives = list(f.attrs["primitives"])

            for dp_key in f.keys():
                dp_group = f[dp_key]

                num_cams = dp_group["depths"].shape[0]
                objects_group = dp_group["objects"]

                for obj_key in objects_group.keys():
                    for cam_idx in range(num_cams):
                        self.index.append((dp_key, obj_key, cam_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        dp_key, obj_key, cam_idx = self.index[idx]

        with h5py.File(DATASET_PATH, "r") as f:
            dp_group = f[dp_key]
            obj_group = dp_group["objects"][obj_key]

            depth = dp_group["depths"][cam_idx].astype(np.float32)
            mask = (dp_group["seg_ids"][cam_idx] == obj_group.attrs["id"]).astype(np.float32)
            pose = obj_group["pose"][()].astype(np.float32)

            feasibles = [int(obj_group["primitives"][prim]["feasible"][()]) for prim in self.primitives]
            y = np.array(feasibles, dtype=np.float32)

        return torch.from_numpy(depth), torch.from_numpy(mask), torch.from_numpy(pose), torch.from_numpy(y)


class DepthSegPoseAllViewsDataset(Dataset):
    """all depth views, binary segmentation masks and obj pose as three different tensors"""

    def __init__(self):
        self.index = []
        with h5py.File(DATASET_PATH, "r") as f:
            self.primitives = list(f.attrs["primitives"])

            for dp_key in f.keys():
                objects_group = f[dp_key]["objects"]
                for obj_key in objects_group.keys():
                    self.index.append((dp_key, obj_key))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        dp_key, obj_key = self.index[idx]

        with h5py.File(DATASET_PATH, "r") as f:
            dp_group = f[dp_key]
            obj_group = dp_group["objects"][obj_key]

            depths = dp_group["depths"][()].astype(np.float32)
            masks = (dp_group["seg_ids"][()] == obj_group.attrs["id"]).astype(np.float32)
            pose = obj_group["pose"][()].astype(np.float32)

            feasibles = [int(obj_group["primitives"][prim]["feasible"][()]) for prim in self.primitives]
            y = np.array(feasibles, dtype=np.float32)

        return torch.from_numpy(depths), torch.from_numpy(masks), torch.from_numpy(pose), torch.from_numpy(y)
