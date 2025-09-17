import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class PoseSizeDataset(Dataset):
    """
    - in memory dataset for the nested h5 structure
    - pose and size concatenated into one tensor of shape (10)
    """

    def __init__(self, h5_path: str):
        self.data = []
        with h5py.File(h5_path, "r") as f:
            self.primitives = list(f.attrs["primitives"])
            for dp_key in tqdm(f.keys(), desc="Caching data"):
                objects_group = f[dp_key]["objects"]
                for obj_key in objects_group.keys():
                    obj_group = objects_group[obj_key]
                    pose = obj_group["pose"][()].astype(np.float32)
                    size = obj_group["size"][()].astype(np.float32)
                    x = np.concatenate([pose, size])

                    feasibles = [int(obj_group["primitives"][prim]["feasible"][()]) for prim in self.primitives]
                    y = np.array(feasibles, dtype=np.float32)
                    self.data.append((torch.from_numpy(x), torch.from_numpy(y)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DepthSegPoseDataset(Dataset):
    def __init__(self, h5_path: str):
        self.data = []
        with h5py.File(h5_path, "r") as f:
            self.primitives = list(f.attrs["primitives"])
            self.image_shape = f[next(iter(f))]["depths"][0].shape

            index = []
            for dp_key in f.keys():
                num_cams = f[dp_key]["depths"].shape[0]
                for obj_key in f[dp_key]["objects"].keys():
                    for cam_idx in range(num_cams):
                        index.append((dp_key, obj_key, cam_idx))

            for dp_key, obj_key, cam_idx in tqdm(index, desc="Caching data"):
                dp_group = f[dp_key]
                obj_group = dp_group["objects"][obj_key]

                depth = dp_group["depths"][cam_idx].astype(np.float32)
                mask = (dp_group["seg_ids"][cam_idx] == obj_group.attrs["id"]).astype(np.float32)
                pose = obj_group["pose"][()].astype(np.float32)
                feasibles = [int(obj_group["primitives"][prim]["feasible"][()]) for prim in self.primitives]
                y = np.array(feasibles, dtype=np.float32)

                image_channels = np.stack([depth, mask], axis=0)

                self.data.append((torch.from_numpy(image_channels), torch.from_numpy(pose), torch.from_numpy(y)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
