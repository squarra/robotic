import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from robotic.datasets import PoseSizeDataset
from robotic.models import PoseSizeMlp

DATASET_PATH = "dataset.h5"
NUM_EPOCHS = 100
BATCH_SIZE = 64
EVAL_EVERY = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset & Dataloader
dataset = PoseSizeDataset(DATASET_PATH)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Model setup
model = PoseSizeMlp(input_dim=10, num_primitives=len(dataset.primitives)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()
history = {"train_loss": [], "val_loss": [], "val_acc": []}


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == y.long()).all(dim=1).sum().item()
            total += x.size(0)

    return total_loss / len(loader.dataset), correct / total


# Training
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)

    train_loss = total_loss / len(train_loader.dataset)
    history["train_loss"].append(train_loss)

    if (epoch + 1) % EVAL_EVERY == 0:
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"epoch {epoch + 1:03d}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")


# Save model
torch.save(model.state_dict(), "PoseSizeMlp.pt")
