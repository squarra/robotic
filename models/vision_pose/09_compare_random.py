import numpy as np
import torch
from torch.utils.data import DataLoader

from models.vision_pose import DATASET_PATH, MODEL_PATH, VisionPoseDataset, VisionPoseNet

BATCH_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataset = VisionPoseDataset(DATASET_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

num_primitives = len(dataset.primitives)

checkpoint = torch.load(MODEL_PATH, map_location=device)
model = VisionPoseNet(
    num_primitives=num_primitives,
    vision_dim=checkpoint["params"]["vision_dim"],
    state_dim=checkpoint["params"]["state_dim"],
    fusion_dim=checkpoint["params"]["fusion_dim"],
    num_mlp_layers=checkpoint["params"]["num_mlp_layers"],
    dropout=checkpoint["params"]["dropout"],
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Metrics
random_calls = []
model_calls = []

rng = np.random.default_rng(42)

with torch.no_grad():
    for depths, masks, pose, cam_positions, feasibles in dataloader:
        depths, masks, pose, cam_positions = depths.to(device), masks.to(device), pose.to(device), cam_positions.to(device)
        feasibles = feasibles.numpy()
        logits = model(depths, masks, pose, cam_positions)
        probs = torch.sigmoid(logits).cpu().numpy()

        batch_size = feasibles.shape[0]

        for i in range(batch_size):
            feasible_mask = feasibles[i] == 1
            if not feasible_mask.any():
                continue  # skip objects with no feasible primitive

            # Baseline: random order
            perm = rng.permutation(num_primitives)
            calls_rand = 0
            for idx in perm:
                calls_rand += 1
                if feasible_mask[idx]:
                    break
            random_calls.append(calls_rand)

            # Model-based order
            order = np.argsort(probs[i])[::-1]
            calls_model = 0
            for idx in order:
                calls_model += 1
                if feasible_mask[idx]:
                    break
            model_calls.append(calls_model)

# Results
avg_random = np.mean(random_calls)
avg_model = np.mean(model_calls)

print("==== Evaluation ====")
print(f"Average solver calls (random order baseline): {avg_random:.2f}")
print(f"Average solver calls (model-guided): {avg_model:.2f}")
print(f"Improvement: {(avg_random - avg_model):.2f} fewer calls on average")
