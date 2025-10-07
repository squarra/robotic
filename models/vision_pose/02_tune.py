import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.vision_pose import DATASET, MODEL_PATH, TEST_DATASET, TRAIN_DATASET, VisionPoseNet

NUM_TRIALS = 10
NUM_EPOCHS = 10
BATCH_SIZE = 16
EVAL_EVERY = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_loader = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


def move_to_device(batch: tuple[torch.Tensor, ...]):
    return tuple(item.to(device, non_blocking=True) for item in batch)


def collect_test_logits(model):
    logits, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            cam_poses, depths, masks, pose, target_pose, y = move_to_device(batch)
            logits.append(model(cam_poses, depths, masks, pose, target_pose))
            labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def find_best_threshold(probs, labels):
    best_thresh, best_metrics = 0.5, (-1, -1, -1, -1)
    for t in torch.linspace(0.1, 0.9, steps=9):
        preds = (probs > t).long()
        acc = (preds == labels).all(dim=1).float().mean().item()
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        if f1 > best_metrics[3]:
            best_thresh, best_metrics = float(t), (acc, precision, recall, f1)
    return best_thresh, best_metrics


best_f1 = -1.0


def objective(trial):
    global best_f1

    cnn_channels = trial.suggest_categorical("cnn_channels", [8, 16, 32])
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    head_layers = trial.suggest_int("head_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)

    model = VisionPoseNet(
        out_features=len(DATASET.primitives),
        cnn_channels=cnn_channels,
        hidden_dim=hidden_dim,
        head_layers=head_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            cam_poses, depths, masks, pose, target_pose, y = move_to_device(batch)

            optimizer.zero_grad()
            logits = model(cam_poses, depths, masks, pose, target_pose)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)

        if (epoch + 1) % EVAL_EVERY == 0:
            model.eval()
            logits, labels = collect_test_logits(model)
            val_loss = criterion(logits, labels).item()
            trial.report(-val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    logits, labels = collect_test_logits(model)
    probs = torch.sigmoid(logits)
    threshold, (val_acc, precision, recall, f1) = find_best_threshold(probs, labels)

    if f1 > best_f1:
        best_f1 = f1
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "params": trial.params,
                "threshold": threshold,
                "val_acc": val_acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
            MODEL_PATH,
        )

    return f1


study = optuna.create_study(study_name="vision_pose", direction="maximize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=NUM_TRIALS)

checkpoint = torch.load(MODEL_PATH, map_location=device)
print(f"Best params: {checkpoint['params']}")
print(f"  threshold: {checkpoint['threshold']:.4f}")
print(f"  val_acc:   {checkpoint['val_acc']:.4f}")
print(f"  precision: {checkpoint['precision']:.4f}")
print(f"  recall:    {checkpoint['recall']:.4f}")
print(f"  f1:        {checkpoint['f1']:.4f}")
