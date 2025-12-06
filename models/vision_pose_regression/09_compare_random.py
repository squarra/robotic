import numpy as np
import torch
from torch.utils.data import DataLoader

from models.vision_pose_regression import DATASET, MODEL_PATH, VisionPoseRegressionNet

BATCH_SIZE = 256
SEED = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataloader = DataLoader(DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
num_primitives = len(DATASET.primitives)

checkpoint = torch.load(MODEL_PATH, map_location=device)
model = VisionPoseRegressionNet(
    out_features=num_primitives,
    cnn_channels=checkpoint["params"]["cnn_channels"],
    state_dim=checkpoint["params"]["state_dim"],
    fusion_dim=checkpoint["params"]["fusion_dim"],
    num_mlp_layers=checkpoint["params"]["num_mlp_layers"],
    dropout=checkpoint["params"]["dropout"],
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def move_to_device(batch: tuple[torch.Tensor, ...]):
    return tuple(item.to(device, non_blocking=True) for item in batch)


rng = np.random.default_rng(SEED)
random_calls = []
model_calls = []
with torch.no_grad():
    for batch in dataloader:
        cam_poses, depths, masks, pose, target_pose, feasibles, pose_diffs = move_to_device(batch)
        feas_logits, pose_pred = model(cam_poses, depths, masks, pose, target_pose)
        probs = torch.sigmoid(feas_logits)

        batch_size = depths.size(0)
        for i in range(batch_size):
            feasible_mask = feasibles[i] == 1
            if not feasible_mask.any():
                continue

            calls_rand = 0
            for idx in rng.permutation(num_primitives):
                calls_rand += 1
                if feasible_mask[idx]:
                    break
            random_calls.append(calls_rand)

            order = np.argsort(probs[i])[::-1]
            calls_model = 0
            for idx in order:
                calls_model += 1
                if feasible_mask[idx]:
                    break
            model_calls.append(calls_model)

avg_random = np.mean(random_calls)
avg_model = np.mean(model_calls)

print("==== Evaluation ====")
print(f"Average solver calls (random order baseline): {avg_random:.2f}")
print(f"Average solver calls (model-guided): {avg_model:.2f}")
print(f"Improvement: {(avg_random - avg_model):.2f} fewer calls on average")
