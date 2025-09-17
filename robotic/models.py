import torch
import torch.nn as nn


class PoseSizeMlp(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=128, num_primitives=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_primitives),
        )

    def forward(self, x):
        return self.net(x)


class DepthPoseFeasibilityNet(nn.Module):
    def __init__(self, image_shape, pose_dim, num_primitives, hidden_dim=128):
        super().__init__()
        # --- 1. Vision Encoder (CNN) ---
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 2, *image_shape)
            cnn_output_dim = self.cnn(dummy_input).shape[1]

        # --- 2. Pose Encoder (MLP) ---
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        # --- 3. Fusion Head (MLP) ---
        self.head = nn.Sequential(
            nn.Linear(cnn_output_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_primitives),
        )

    def forward(self, image_input, pose_input):
        vision_features = self.cnn(image_input)
        pose_features = self.pose_encoder(pose_input)
        return self.head(torch.cat([vision_features, pose_features], dim=1))
