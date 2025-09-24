import h5py
from torch.utils.data import Dataset

SCENE_FIELDS = ["camera_positions", "images", "depths", "seg_ids"]
OBJECT_FIELDS = ["poses", "sizes", "target_poses", "feasibles", "final_poses"]


class InMemoryDataset(Dataset):
    def __init__(self, h5_path: str, fields: list[str]):
        self.h5_path = h5_path
        self.data = []

        with h5py.File(h5_path, "r") as f:
            self.primitives = list(f.attrs["primitives"])
            for dp_key in f.keys():
                dp_group = f[dp_key]
                num_objects = dp_group["poses"].shape[0]

                for oi in range(num_objects):
                    entry = []
                    for field in fields:
                        if field in SCENE_FIELDS:
                            entry.append(dp_group[field][()])
                        elif field in OBJECT_FIELDS:
                            entry.append(dp_group[field][oi])
                        elif field == "masks":
                            entry.append(dp_group["seg_ids"][()] == oi)
                        else:
                            raise ValueError
                    self.data.append(entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class LazyDataset(Dataset):
    def __init__(self, h5_path: str, fields: list[str]):
        self.h5_path = h5_path
        self.fields = fields
        self.index = []
        self._h5_file = None

        with h5py.File(h5_path, "r") as f:
            self.primitives = list(f.attrs["primitives"])
            for dp_key in f.keys():
                poses = f[dp_key]["poses"]
                for oi in range(poses.shape[0]):
                    self.index.append((dp_key, oi))

    def _init_h5(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")

    def close(self):
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __del__(self):
        self.close()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        self._init_h5()
        dp_key, oi = self.index[idx]
        dp_group = self._h5_file[dp_key]
        data = []
        for field in self.fields:
            if field in SCENE_FIELDS:
                data.append(dp_group[field][()])
            elif field in OBJECT_FIELDS:
                data.append(dp_group[field][oi])
            elif field == "masks":
                data.append(dp_group["seg_ids"][()] == oi)
            else:
                raise ValueError
        return data
