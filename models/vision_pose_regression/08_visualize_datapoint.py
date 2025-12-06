import h5py
import matplotlib.pyplot as plt
import numpy as np

from models.vision_pose_regression import DATASET_PATH


def visualize_datapoint(h5_path, datapoint_key=None):
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        if datapoint_key is None:
            datapoint_key = keys[3]
        print(f"Using: {datapoint_key}")

        dp = f[datapoint_key]
        depths = dp["depths"][()]
        masks = dp["masks"][()]
        goals = dp["goal_maps"][()]

        num_views = depths.shape[0]

        fig1, axes1 = plt.subplots(num_views, 3, figsize=(12, 4 * num_views), squeeze=False)
        fig1.suptitle(f"Datapoint: {datapoint_key} — Individual Maps", fontsize=14)

        for vi in range(num_views):
            depth = depths[vi]
            mask = masks[vi]
            goal = goals[vi]

            axes1[vi, 0].imshow(depth, cmap="viridis")
            axes1[vi, 0].set_title(f"Depth (View {vi})")

            axes1[vi, 1].imshow(mask, cmap="gray")
            axes1[vi, 1].set_title("Segmentation Mask")

            axes1[vi, 2].imshow(goal, cmap="hot")
            axes1[vi, 2].set_title("Goal (Gaussian) Map")

            for ax in axes1[vi]:
                ax.axis("off")

        plt.tight_layout()
        plt.show()

        fig2, axes2 = plt.subplots(1, num_views, figsize=(5 * num_views, 5), squeeze=False)
        fig2.suptitle(f"Datapoint: {datapoint_key} — Overlay View", fontsize=14)

        for vi in range(num_views):
            depth = depths[vi]
            mask = masks[vi]
            goal = goals[vi]

            depth[mask == 1.0] = 0.0
            target_v, target_u = np.unravel_index(np.argmax(goal), goal.shape)

            ax = axes2[0, vi]
            ax.imshow(depth, cmap="viridis", interpolation="none")
            ax.imshow(goal, cmap="hot", alpha=0.4, interpolation="none")
            ax.scatter([target_u], [target_v], color="red", s=40, edgecolors="white", linewidths=1.0)

            ax.set_title(f"View {vi}")
            ax.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    visualize_datapoint(DATASET_PATH)
