import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.pose_size import DATASET, TEST_DATASET, TRAIN_DATASET, PoseSizeMlp

NUM_EPOCHS = 1000
BATCH_SIZE = 64
EVAL_EVERY = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_loader = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

model = PoseSizeMlp(len(DATASET.primitives)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()


def evaluate():
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for pose, size, target_pose, y in test_loader:
            x = torch.cat((pose, size, target_pose), dim=1)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == y.long()).all(dim=1).sum().item()
            total += x.size(0)

    return total_loss / len(test_loader.dataset), correct / total


for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    for pose, size, target_pose, y in train_loader:
        x = torch.cat((pose, size, target_pose), dim=1)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)

    train_loss = total_loss / len(train_loader.dataset)

    if (epoch + 1) % EVAL_EVERY == 0:
        val_loss, val_acc = evaluate()
        print(f"epoch {epoch + 1:03d}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")


torch.save(model.state_dict(), "PoseSizeMlp.pt")
