import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.pose_size import DATASET, MODEL_PATH, TEST_DATASET, TRAIN_DATASET, PoseSizeMlp

NUM_TRIALS = 20
NUM_EPOCHS = 100
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_loader = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


def evaluate(model, criterion):
    model.eval()
    total_loss, n_samples = 0.0, 0

    total_correct_per_label = 0
    total_labels = 0
    tp, fp, fn = 0, 0, 0

    with torch.no_grad():
        for pose, size, target_pose, y in test_loader:
            x = torch.cat((pose, size, target_pose), dim=1)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            n_samples += x.size(0)

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

    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 1024])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)

    model = PoseSizeMlp(len(DATASET.primitives), hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(NUM_EPOCHS):
        model.train()
        for pose, size, target_pose, y in train_loader:
            x = torch.cat((pose, size, target_pose), dim=1)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    acc, precision, recall, f1 = evaluate(model, criterion)

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
            MODEL_PATH,
        )

    return acc


study = optuna.create_study(study_name="pose_size", direction="maximize")
study.optimize(objective, n_trials=NUM_TRIALS)

checkpoint = torch.load(MODEL_PATH, map_location=device)
print(f"Best trial params: {checkpoint['params']}")
print(f"  val_acc:   {checkpoint['val_acc']:.4f}")
print(f"  precision: {checkpoint['precision']:.4f}")
print(f"  recall:    {checkpoint['recall']:.4f}")
print(f"  f1:        {checkpoint['f1']:.4f}")
