import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from robotic.datasets import DepthSegPoseDataset
from robotic.models import DepthPoseFeasibilityNet

DATASET_PATH = "dataset.h5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


dataset = DepthSegPoseDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

model = DepthPoseFeasibilityNet(image_shape=dataset.image_shape, pose_dim=7, num_primitives=len(dataset.primitives)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(100):
    total_loss = 0
    model.train()
    for image, pose, y in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
        image, pose, y = image.to(device), pose.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(image, pose)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * image.size(0)

    print(f"Epoch {epoch + 1} - Loss: {total_loss / len(dataset):.4f}")


model.eval()
with torch.no_grad():
    image, pose, y_true = dataset[0]
    image = image.unsqueeze(0).to(device)
    pose = pose.unsqueeze(0).to(device)

    logits = model(image, pose)
    probs = torch.sigmoid(logits)
    print("Predicted:", probs.round().squeeze().cpu().numpy())
    print("Actual:   ", y_true.numpy())
