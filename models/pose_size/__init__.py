import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

DATASET_PATH = "dataset.h5"
MODEL_PATH = "pose_size_net.pt"


class PoseSizeDataset(Dataset):
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self.data = []

        with h5py.File(h5_path, "r") as f:
            self.primitives = list(f.attrs["primitives"])
            for dp_key in f.keys():
                dp_group = f[dp_key]
                num_objects = dp_group["poses"].shape[0]

                for oi in range(num_objects):
                    x = np.concatenate([dp_group["poses"][oi], dp_group["sizes"][oi]])
                    y = dp_group["feasibles"][oi]
                    self.data.append((torch.from_numpy(x), torch.from_numpy(y)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PoseSizeMlp(nn.Module):
    def __init__(self, num_primitives, hidden_dim=128, num_layers=2, dropout=0.0):
        super().__init__()
        layers = []
        input_dim = 10

        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, num_primitives))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


generator = torch.Generator().manual_seed(42)
DATASET = PoseSizeDataset(DATASET_PATH)
TRAIN_DATASET, TEST_DATASET = torch.utils.data.random_split(DATASET, [0.8, 0.2], generator=generator)
