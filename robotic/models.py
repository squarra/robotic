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


class VisionPoseNet(nn.Module):
    def __init__(
        self,
        num_primitives,
        vision_dim=64,
        state_dim=64,
        fusion_dim=128,
        num_mlp_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        # --- Vision Pathway (shallow CNN) ---
        # For 2-channel input (depth + mask)
        self.vision_cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),  # (H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(32, vision_dim, kernel_size=3, stride=2, padding=1),  # (H/8, W/8)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # → (B*N_cams, vision_dim, 1, 1)
        )

        # --- State Pathway (pose + camera_positions) ---
        state_input_dim = 7 + 9
        state_layers = []
        for i in range(num_mlp_layers):
            in_dim = state_input_dim if i == 0 else state_dim
            state_layers.extend([nn.Linear(in_dim, state_dim), nn.ReLU(), nn.Dropout(dropout)])
        self.state_mlp = nn.Sequential(*state_layers)

        # --- Fusion Head (vision + state features) ---
        fusion_input_dim = vision_dim + state_dim
        fusion_layers = []
        for i in range(num_mlp_layers):
            in_dim = fusion_input_dim if i == 0 else fusion_dim
            fusion_layers.extend([nn.Linear(in_dim, fusion_dim), nn.ReLU(), nn.Dropout(dropout)])
        fusion_layers.append(nn.Linear(fusion_dim, num_primitives))
        self.fusion_head = nn.Sequential(*fusion_layers)

    def forward(self, depths, masks, pose, camera_positions):
        """
        Args:
            depths: [B, 3, H, W]
            masks: [B, 3, H, W]
            pose: [B, 7]
            camera_positions: [B, 3, 3]
        """
        B, N_cams, H, W = depths.shape

        # ---- Vision ----
        # stack depth & mask → [B, N_cams, 2, H, W]
        vision_in = torch.stack([depths, masks], dim=2)
        vision_in = vision_in.view(B * N_cams, 2, H, W)

        vision_feats = self.vision_cnn(vision_in)  # [B*N_cams, vision_dim, 1, 1]
        vision_feats = vision_feats.view(B, N_cams, -1)  # [B, N_cams, vision_dim]

        # aggregate across cameras (mean or max)
        vision_feats = vision_feats.mean(dim=1)  # [B, vision_dim]

        # ---- State ----
        cam_pos_flat = camera_positions.view(B, -1)  # [B, 9]
        state_in = torch.cat([pose, cam_pos_flat], dim=1)  # [B, 16]
        state_feats = self.state_mlp(state_in)  # [B, state_dim]

        # ---- Fusion ----
        fused = torch.cat([vision_feats, state_feats], dim=1)  # [B, vision+state]
        logits = self.fusion_head(fused)  # [B, num_primitives]
        return logits
