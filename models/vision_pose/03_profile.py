import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.vision_pose import DATASET, VisionPoseNet

PROFILER_STEPS = 32
BATCH_SIZE = 32
LOG_DIR = "log"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataloader = DataLoader(DATASET, batch_size=BATCH_SIZE)
model = VisionPoseNet(len(DATASET.primitives)).to(device)
optimizer = torch.optim.Adam(model.parameters())
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
