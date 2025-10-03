import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.vision_pose import DATASET, TEST_DATASET, TRAIN_DATASET, VisionPoseNet

NUM_EPOCHS = 10
BATCH_SIZE = 16
EVAL_EVERY = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_loader = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

model = VisionPoseNet(num_primitives=len(DATASET.primitives), num_views=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()


def evaluate():
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for cam_poses, depths, masks, pose, target_pose, y in test_loader:
            cam_poses = cam_poses.to(device, non_blocking=True)
            depths = depths.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            pose = pose.to(device, non_blocking=True)
            target_pose = target_pose.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(cam_poses, depths, masks, pose, target_pose)
            loss = criterion(logits, y)
            total_loss += loss.item() * depths.size(0)

            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == y.long()).all(dim=1).sum().item()
            total += depths.size(0)

    return total_loss / len(test_loader.dataset), correct / total


for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    for cam_poses, depths, masks, pose, target_pose, y in train_loader:
        cam_poses = cam_poses.to(device, non_blocking=True)
        depths = depths.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        pose = pose.to(device, non_blocking=True)
        target_pose = target_pose.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(cam_poses, depths, masks, pose, target_pose)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * depths.size(0)

    if (epoch + 1) % EVAL_EVERY == 0:
        train_loss = total_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate()
        print(f"epoch {epoch + 1:03d}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")


torch.save(model.state_dict(), "VisionPoseNet.pt")
