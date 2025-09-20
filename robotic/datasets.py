import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class PoseSizeDataset(Dataset):
    """
    - x: concatenation of pose (7) and size (3) → tensor of shape (10,)
    - y: feasibility of each primitive → tensor of shape (num_primitives,)
    """

    def __init__(self, h5_path: str):
        self.data = []
        with h5py.File(h5_path, "r") as f:
            self.primitives = list(f.attrs["primitives"])
            for dp_key in tqdm(f.keys(), desc="Caching data"):
                dp_group = f[dp_key]

                poses = dp_group["poses"][()]  # (num_objects, 7)
                sizes = dp_group["sizes"][()]  # (num_objects, 3)
                feasibles = dp_group["feasibles"][()]  # (num_objects, num_primitives)

                num_objects = poses.shape[0]

                for oi in range(num_objects):
                    x = np.concatenate([poses[oi], sizes[oi]]).astype(np.float32)
                    y = feasibles[oi].astype(np.float32)
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
