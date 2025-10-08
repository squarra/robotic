import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.vision_pose_regression import DATASET, MODEL_PATH, TEST_DATASET, TRAIN_DATASET, VisionPoseRegressionNet

NUM_TRIALS = 10
NUM_EPOCHS = 20
BATCH_SIZE = 16
EVAL_EVERY = 5
POSE_LOSS_WEIGHT = 10.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_loader = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

mse_loss = nn.MSELoss(reduction="none")
bce_loss = nn.BCEWithLogitsLoss()


def move_to_device(batch: tuple[torch.Tensor, ...]):
    return tuple(item.to(device, non_blocking=True) for item in batch)


def compute_masked_pose_loss(pred_pose: torch.Tensor, target_pose: torch.Tensor, feas_mask: torch.Tensor):
    if feas_mask.sum() == 0:
        return pred_pose.new_tensor(0.0)
    diff = mse_loss(pred_pose, target_pose).mean(dim=-1)
    diff = diff * feas_mask
    return diff.sum() / (feas_mask.sum() + 1e-8)


def collect_test_outputs(model):
    all_feas_logits, all_pose_preds, all_feats, all_pose_diffs = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            cam_poses, depths, masks, pose, target_pose, feasibles, pose_diffs = move_to_device(batch)
            feas_logits, pose_pred = model(cam_poses, depths, masks, pose, target_pose)
            all_feas_logits.append(feas_logits)
            all_pose_preds.append(pose_pred)
            all_feats.append(feasibles)
            all_pose_diffs.append(pose_diffs)
    return (torch.cat(all_feas_logits), torch.cat(all_pose_preds), torch.cat(all_feats), torch.cat(all_pose_diffs))


best_loss = float("inf")


def objective(trial):
    global best_loss

    cnn_channels = trial.suggest_categorical("cnn_channels", [8, 16, 32])
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    head_layers = trial.suggest_int("head_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)

    model = VisionPoseRegressionNet(
        out_features=len(DATASET.primitives),
        cnn_channels=cnn_channels,
        hidden_dim=hidden_dim,
        head_layers=head_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(NUM_EPOCHS):
        model.train()

        for batch in train_loader:
            cam_poses, depths, masks, pose, target_pose, feasibles, pose_diffs = move_to_device(batch)

            optimizer.zero_grad()
            feas_logits, pose_pred = model(cam_poses, depths, masks, pose, target_pose)
            feas_loss = bce_loss(feas_logits, feasibles)
            pose_loss = compute_masked_pose_loss(pose_pred, pose_diffs, feasibles)
            loss = feas_loss + POSE_LOSS_WEIGHT * pose_loss
            loss.backward()
            optimizer.step()

        if (epoch + 1) % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                feas_logits, pose_preds, feasibles, pose_diffs = collect_test_outputs(model)
                feas_loss = bce_loss(feas_logits, feasibles).item()
                pose_loss = compute_masked_pose_loss(pose_preds, pose_diffs, feasibles).item()
                loss = feas_loss + POSE_LOSS_WEIGHT * pose_loss

            trial.report(loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    model.eval()
    with torch.no_grad():
        feas_logits, pose_preds, feasibles, pose_diffs = collect_test_outputs(model)
        feas_loss = bce_loss(feas_logits, feasibles).item()
        pose_loss = compute_masked_pose_loss(pose_preds, pose_diffs, feasibles).item()
        loss = feas_loss + POSE_LOSS_WEIGHT * pose_loss

    if loss < best_loss:
        best_loss = loss
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "params": trial.params,
                "val_loss_feas": feas_loss,
                "val_loss_pose": pose_loss,
                "combined_val_loss": loss,
            },
            MODEL_PATH,
        )

    return loss


study = optuna.create_study(study_name="vision_pose_regression", direction="minimize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=NUM_TRIALS)

checkpoint = torch.load(MODEL_PATH, map_location=device)
print("\nBest parameters:")
for k, v in checkpoint["params"].items():
    print(f"  {k}: {v}")
print(f"  val_loss_feas:   {checkpoint['val_loss_feas']:.6f}")
print(f"  val_loss_pose:   {checkpoint['val_loss_pose']:.6f}")
print(f"  combined_val_loss: {checkpoint['combined_val_loss']:.6f}")
