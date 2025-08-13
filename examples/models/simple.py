import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

with h5py.File("dataset.h5", "r") as f:
    dp = f["datapoint_0001"]

    images = np.array(dp["images"], dtype=np.float32) / 255.0
    camera_positions = np.array(dp["camera_positions"], dtype=np.float32)
    target_pos = np.array(dp["target_pos"], dtype=np.float32)

    xs, ys = [], []
    manipulations = dp["manipulations"]
    for obj in manipulations:
        obj_group = manipulations[obj]
        masks = np.array(obj_group["masks"])
        xs.append(np.concatenate([images.flatten(), masks.flatten(), camera_positions.flatten(), target_pos.flatten()]))
        primitives = obj_group["primitives"]
        ys.append([np.array(primitives[prim]) for prim in primitives])

X = np.stack(xs)
Y = np.stack(ys, dtype=np.float32)
print(f"X shape: {X.shape}, Y shape: {Y.shape}")
print(f"X shape: {X.dtype}, Y shape: {Y.dtype}")

X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)


class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


model = Model(input_dim=X_tensor.shape[1], output_dim=Y_tensor.shape[1])
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    optimizer.zero_grad()
    logits = model(X_tensor)
    loss = criterion(logits, Y_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

with torch.no_grad():
    logits = model(X_tensor)
    pred = torch.sigmoid(logits)
    print("\nFinal predictions:")
    for pred_row, label_row in zip(pred.tolist(), Y_tensor.tolist()):
        for p, l in zip(pred_row, label_row):
            print(f"  pred={p:.3f}, label={l}")
