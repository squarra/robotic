"""
This demonstrates what the current performance bottlenecks are. It's mostly the Dataloading Disk IO.
If you put some record_function calls inside the Dataset.__get_item__ function, you can see that even better.
For profiling num_workers is not set. In practice it should be set and will increase the Disk IO times.
"""

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader

from models.vision_pose import DATASET, VisionPoseNet

NUM_EPOCHS = 64
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataloader = DataLoader(DATASET, batch_size=BATCH_SIZE)
data_iter = iter(dataloader)
model = VisionPoseNet(len(DATASET.primitives)).to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

with profile(activities=[ProfilerActivity.CPU]) as prof:
    for i in range(NUM_EPOCHS):
        with record_function("Dataloading"):
            depths, masks, pose, cam_positions, y = next(data_iter)
        with record_function("Training"):
            depths, masks, pose, cam_positions, y = (depths.to(device), masks.to(device), pose.to(device), cam_positions.to(device), y.to(device))
            optimizer.zero_grad()
            logits = model(depths, masks, pose, cam_positions)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            prof.step()

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
