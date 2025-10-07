import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset

DATASET_PATH = "vision_pose_dataset.h5"
MODEL_PATH = "vision_pose_net.pt"


class VisionPoseDataset(Dataset):
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self._h5_file = None

        with h5py.File(h5_path, "r") as f:
            self.keys = list(f.keys())
            self.primitives = list(f.attrs["primitives"])

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
        return len(self.keys)

    def __getitem__(self, idx):
        self._init_h5()
        dp = self._h5_file[self.keys[idx]]
        return dp["cam_poses"][()], dp["depths"][()], dp["masks"][()], dp["pose"][()], dp["target_pose"][()], dp["feasibles"][()]


class VisionPoseNet(nn.Module):
    def __init__(self, out_features: int, cnn_channels: int = 16, hidden_dim: int = 64, head_layers: int = 2, dropout: float = 0.0):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(2, cnn_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_channels, cnn_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        cnn_out_dim = cnn_channels * 2

        self.cam_pose_fc = nn.Sequential(nn.Linear(7, hidden_dim), nn.ReLU())
        self.pose_fc = nn.Sequential(nn.Linear(7, hidden_dim), nn.ReLU())
        self.target_pose_fc = nn.Sequential(nn.Linear(7, hidden_dim), nn.ReLU())

        layers = []
        input_dim = cnn_out_dim + hidden_dim * 3
        for _ in range(head_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, out_features))
        self.head = nn.Sequential(*layers)

    def forward(self, cam_poses, depths, masks, pose, target_pose):
        B, V, H, W = depths.shape

        per_view_feats = []
        for v in range(V):
            view_input = torch.stack([depths[:, v], masks[:, v]], dim=1)
            view_feat = self.cnn(view_input).view(B, -1)
            per_view_feats.append(view_feat)
        x_img = torch.stack(per_view_feats, dim=1).mean(dim=1)

        cam_feats = []
        for v in range(V):
            cam_feat = self.cam_pose_fc(cam_poses[:, v])
            cam_feats.append(cam_feat)
        x_cam = torch.stack(cam_feats, dim=1).mean(dim=1)

        x_pose = self.pose_fc(pose)
        x_tpose = self.target_pose_fc(target_pose)

        x = torch.cat([x_img, x_cam, x_pose, x_tpose], dim=1)
        return self.head(x)


generator = torch.Generator().manual_seed(42)
DATASET = VisionPoseDataset(DATASET_PATH)
TRAIN_DATASET, TEST_DATASET = torch.utils.data.random_split(DATASET, [0.8, 0.2], generator=generator)
