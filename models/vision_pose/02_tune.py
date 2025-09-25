import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from robotic.datasets import LazyDataset
from robotic.models import VisionPoseNet

DATASET_PATH = "dataset.h5"
EPOCHS = 30
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = LazyDataset(DATASET_PATH, ["depths", "masks", "poses", "camera_positions", "feasibles"])
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


def evaluate(model, loader, device):
    model.eval()
    total_correct_per_label, total_labels = 0, 0
    tp, fp, fn = 0, 0, 0

    with torch.no_grad():
        for depths, masks, poses, cam_pos, y in loader:
            depths, masks, poses, cam_pos, y = (depths.to(device), masks.to(device), poses.to(device), cam_pos.to(device), y.to(device))

            logits = model(depths, masks, poses, cam_pos)
            preds = (torch.sigmoid(logits) > 0.5).long()
            y_long = y.long()

            total_correct_per_label += (preds == y_long).sum().item()
            total_labels += y.numel()

            tp += ((preds == 1) & (y_long == 1)).sum().item()
            fp += ((preds == 1) & (y_long == 0)).sum().item()
            fn += ((preds == 0) & (y_long == 1)).sum().item()

    acc = total_correct_per_label / total_labels
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return acc, precision, recall, f1


best_acc = 0.0


def objective(trial):
    global best_acc

    vision_dim = trial.suggest_categorical("vision_dim", [32, 64, 128])
    state_dim = trial.suggest_categorical("state_dim", [32, 64, 128])
    fusion_dim = trial.suggest_categorical("fusion_dim", [64, 128, 256])
    num_mlp_layers = trial.suggest_int("num_mlp_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    model = VisionPoseNet(
        num_primitives=len(dataset.primitives),
        vision_dim=vision_dim,
        state_dim=state_dim,
        fusion_dim=fusion_dim,
        num_mlp_layers=num_mlp_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(EPOCHS):
        model.train()
        for depths, masks, poses, cam_pos, y in train_loader:
            depths, masks, poses, cam_pos, y = (
                depths.to(device),
                masks.to(device),
                poses.to(device),
                cam_pos.to(device),
                y.to(device),
            )

            optimizer.zero_grad()
            logits = model(depths, masks, poses, cam_pos)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    acc, precision, recall, f1 = evaluate(model, test_loader, device)

    if acc > best_acc:
        best_acc = acc
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "params": trial.params,
                "val_acc": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
            "vision_pose_best.pt",
        )

    return acc


study = optuna.create_study(study_name="vision_pose", direction="maximize")
study.optimize(objective, n_trials=20)

print("Best trial:")
trial = study.best_trial
checkpoint = torch.load("best_simple_model.pt", map_location=device)
print(f"  Val Per-label Acc: {checkpoint['val_acc']:.4f}")
print(f"  Precision: {checkpoint['precision']:.4f}")
print(f"  Recall: {checkpoint['recall']:.4f}")
print(f"  F1: {checkpoint['f1']:.4f}")
print("  Params:", checkpoint["params"])
