import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.vision_pose_regression import DATASET, TEST_DATASET, TRAIN_DATASET, VisionPoseRegressionNet

CHECKPOINT_PATH = "VisionPoseRegressionNet.pt"
NUM_EPOCHS = 100
BATCH_SIZE = 32
EVAL_EVERY = 1
POSE_LOSS_WEIGHT = 10.0
STARTING_EPOCH = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
num_primitives = len(DATASET.primitives)

train_loader = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
train_dataset_size = len(train_loader.dataset)
test_dataset_size = len(test_loader.dataset)

# Initialize model and load checkpoint
model = VisionPoseRegressionNet(out_features=num_primitives).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
bce_loss = nn.BCEWithLogitsLoss()
mse_loss = nn.MSELoss(reduction="none")


def move_to_device(batch: tuple[torch.Tensor, ...]):
    return tuple(item.to(device, non_blocking=True) for item in batch)


def compute_masked_pose_loss(pred_pose: torch.Tensor, target_pose: torch.Tensor, feas_mask: torch.Tensor):
    if feas_mask.sum() == 0:
        return pred_pose.new_tensor(0.0)
    diff = mse_loss(pred_pose, target_pose).mean(dim=-1)
    diff = diff * feas_mask
    return diff.sum() / (feas_mask.sum() + 1e-8)


def evaluate():
    model.eval()
    total_feas_loss, total_pose_loss, total_acc = 0.0, 0.0, 0.0

    with torch.no_grad():
        for batch in test_loader:
            cam_poses, depths, masks, pose, target_pose, feasibles, pose_diffs = move_to_device(batch)

            feas_logits, pose_pred = model(cam_poses, depths, masks, pose, target_pose)
            feas_loss = bce_loss(feas_logits, feasibles)
            pose_loss = compute_masked_pose_loss(pose_pred, pose_diffs, feasibles)

            batch_size = depths.size(0)
            total_feas_loss += feas_loss * batch_size
            total_pose_loss += pose_loss * batch_size

            preds = torch.sigmoid(feas_logits) > 0.5
            total_acc += (preds == feasibles).sum()

    feas_loss = total_feas_loss / test_dataset_size
    pose_loss = total_pose_loss / test_dataset_size
    acc = total_acc / (test_dataset_size * num_primitives)
    return feas_loss, pose_loss, acc


for epoch in range(STARTING_EPOCH, NUM_EPOCHS):
    model.train()
    total_feas_loss, total_pose_loss = 0.0, 0.0

    for batch in train_loader:
        cam_poses, depths, masks, pose, target_pose, feasibles, pose_diffs = move_to_device(batch)

        optimizer.zero_grad()
        feas_logits, pose_pred = model(cam_poses, depths, masks, pose, target_pose)
        feas_loss = bce_loss(feas_logits, feasibles)
        pose_loss = compute_masked_pose_loss(pose_pred, pose_diffs, feasibles)
        loss = feas_loss + POSE_LOSS_WEIGHT * pose_loss
        loss.backward()
        optimizer.step()

        batch_size = depths.size(0)
        total_feas_loss += feas_loss * batch_size
        total_pose_loss += pose_loss * batch_size

    if (epoch + 1) % EVAL_EVERY == 0:
        feas_loss = total_feas_loss / train_dataset_size
        pose_loss = total_pose_loss / train_dataset_size
        val_feas_loss, val_pose_loss, val_acc = evaluate()
        print(
            f"epoch {epoch + 1:03d}: "
            f"train=(feas_loss={feas_loss:.4f}, pose_loss={pose_loss:.4f}), "
            f"val=(feas_loss={val_feas_loss:.4f}, pose_loss={val_pose_loss:.4f}, acc={val_acc:.2%})"
        )

torch.save(model.state_dict(), "VisionPoseRegressionNet.pt")
