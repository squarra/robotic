import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DATASET_PATH = "vision_pose_dataset.h5"
PROFILER_STEPS = 32
BATCH_SIZE = 32
LOG_DIR = "log"


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
        return dp["depths"][()], dp["masks"][()], dp["poses"][()], dp["camera_positions"][()], dp["feasibles"][()]


class VisionPoseNet(nn.Module):
    def __init__(self, num_primitives, vision_dim=64, state_dim=64, fusion_dim=128, num_mlp_layers=2, dropout=0.1):
        super().__init__()

        # --- Vision Pathway (shallow CNN) ---
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset
dataset = VisionPoseDataset(DATASET_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# Model
model = VisionPoseNet(len(dataset.primitives)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=PROFILER_STEPS, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(LOG_DIR),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for i, (depths, masks, pose, cam_positions, y) in enumerate(dataloader):
        if i >= (1 + 1 + PROFILER_STEPS):  # wait + warmup + active
            break
        depths, masks, pose, cam_positions, y = (depths.to(device), masks.to(device), pose.to(device), cam_positions.to(device), y.to(device))
        optimizer.zero_grad()
        logits = model(depths, masks, pose, cam_positions)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        prof.step()

print(prof.key_averages().table())
