import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from robotic.datasets import PoseSizeDataset
from robotic.models import PoseSizeMlp

DATASET_PATH = "dataset.h5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


dataset = PoseSizeDataset()
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

model = PoseSizeMlp(input_dim=10, num_primitives=len(dataset.primitives)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(100):
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    print(f"Epoch {epoch + 1} - Loss: {total_loss / len(dataset):.4f}")


model.eval()
with torch.no_grad():
    x, y = dataset[0]
    x = x.to(device)
    logits = model(x.unsqueeze(0))
    probs = torch.sigmoid(logits)
    print("Predicted:", probs.round().squeeze().cpu().numpy())
    print("Actual:   ", y.numpy())
