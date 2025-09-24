import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from robotic.datasets import LazyDataset
from robotic.models import DepthsMasksPoseNetSmall

DATASET_PATH = "dataset.h5"
NUM_EPOCHS = 100
BATCH_SIZE = 16
EVAL_EVERY = 1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset
dataset = LazyDataset(DATASET_PATH, ["depths", "masks", "poses", "camera_positions", "feasibles"])
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Model
model = DepthsMasksPoseNetSmall(num_primitives=len(dataset.primitives)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()
history = {"train_loss": [], "val_loss": [], "val_acc": []}


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for depths, masks, pose, cam_positions, y in loader:
            depths, masks, pose, cam_positions, y = (depths.to(device), masks.to(device), pose.to(device), cam_positions.to(device), y.to(device))
            logits = model(depths, masks, pose, cam_positions)
            loss = criterion(logits, y)
            total_loss += loss.item() * depths.size(0)

            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == y.long()).all(dim=1).sum().item()
            total += depths.size(0)

    return total_loss / len(loader.dataset), correct / total


# Training Loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    for depths, masks, pose, cam_positions, y in tqdm(train_loader):
        depths, masks, pose, cam_positions, y = (depths.to(device), masks.to(device), pose.to(device), cam_positions.to(device), y.to(device))
        optimizer.zero_grad()
        logits = model(depths, masks, pose, cam_positions)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * depths.size(0)

    train_loss = total_loss / len(train_loader.dataset)
    history["train_loss"].append(train_loss)

    if (epoch + 1) % EVAL_EVERY == 0:
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"epoch {epoch + 1:03d}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")


# Save model
torch.save(model.state_dict(), "DepthsMasksPoseNet.pt")
