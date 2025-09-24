import torch
import torch.nn as nn


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


class DepthsMasksPoseNetSmall(nn.Module):
    def __init__(self, num_primitives: int):
        super().__init__()

        # Tiny CNN for depth+mask
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=2, padding=1),  # → (H/2,W/2)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # → (H/4,W/4)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # → (B,16,1,1)
        )

        # Camera position embedding
        self.cam_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
        )

        # Pose embedding
        self.pose_mlp = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
        )

        # Fusion layer
        fused_cam_dim = 16 + 16  # CNN + cam_pos
        fusion_dim = fused_cam_dim + 32  # add pose

        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_primitives),
        )

    def forward(self, depths, masks, pose, cam_positions):
        """
        depths: (B, C, H, W)
        masks: (B, C, H, W)
        pose: (B, 7)
        cam_positions: (B, C, 3)
        """
        B, C, H, W = depths.shape

        # stack depth+mask
        x = torch.stack([depths, masks], dim=2)  # (B,C,2,H,W)
        x = x.view(B * C, 2, H, W)
        cnn_feats = self.cnn(x).view(B, C, -1)  # (B,C,16)

        # camera positions
        cam_pos_feats = self.cam_mlp(cam_positions.view(B * C, 3)).view(B, C, -1)  # (B,C,16)

        # concat per-camera
        cam_feats = torch.cat([cnn_feats, cam_pos_feats], dim=-1)  # (B,C,32)
        cam_feats = cam_feats.mean(dim=1)  # avg across cams → (B,32)

        # pose embedding
        pose_feat = self.pose_mlp(pose)  # (B,32)

        # fuse all
        fused = torch.cat([cam_feats, pose_feat], dim=-1)  # (B,64)
        out = self.fc(fused)  # (B,num_primitives)

        return out
