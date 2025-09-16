import os

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

DATASET_PATH = "dataset.h5"
FIGURE_OUTPUT_PATH = "figs"

os.makedirs(FIGURE_OUTPUT_PATH, exist_ok=True)


def compute_pose_errors(df_feasible: pd.DataFrame):
    poses_target = np.vstack(df_feasible["target_pose"].to_numpy())
    poses_final = np.vstack(df_feasible["final_pose"].to_numpy())
    # Position error (L2 norm)
    position_errors = np.linalg.norm(poses_final[:, :3] - poses_target[:, :3], axis=1)
    # Orientation error (quaternion geodesic distance)
    dots = np.sum(poses_target[:, 3:] * poses_final[:, 3:], axis=1)
    orientation_errors = 2 * np.arccos(np.clip(np.abs(dots), -1.0, 1.0))
    return position_errors, orientation_errors


def detect_outliers(data, factor=1.5):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return (data < lower_bound) | (data > upper_bound)


with h5py.File(DATASET_PATH, "r") as f:
    data = [
        {
            "scene_id": int(scene_key.split("_")[1]),
            "object_name": obj_name,
            "object_volume": np.prod(obj_data["size"][:]),
            "primitive_name": primitive_name,
            "primitive_type": primitive_name.split("_")[0],
            "feasible": bool(prim_data["feasible"][()]),
            "final_pose": prim_data["final_pose"][:],
            "target_pose": prim_data["target_pose"][:],
        }
        for scene_key in f.keys()
        for obj_name, obj_data in f[scene_key]["objects"].items()
        for primitive_name, prim_data in obj_data["primitives"].items()
    ]

df = pd.DataFrame(data)
print(f"Loaded {len(df)} attempts from {df['scene_id'].nunique()} scenes.")


# --- 2. Feasibility Analysis ---
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Manipulation Feasibility Analysis", fontsize=16)

# Overall feasibility rate
feas_rate = df["feasible"].mean()
axes[0, 0].set_title("Overall Feasibility Rate")
axes[0, 0].pie([feas_rate, 1 - feas_rate], labels=[f"Feasible ({feas_rate:.1%})", "Infeasible"], autopct="%1.1f%%", startangle=90)

# Feasibility Rate by Primitive Type
feas_by_type = df.groupby("primitive_type")["feasible"].agg(["mean", "count"])
axes[0, 1].set_title("Feasibility Rate by Primitive Type")
axes[0, 1].bar(feas_by_type.index, feas_by_type["mean"])
axes[0, 1].bar_label(axes[0, 1].containers[0], fmt="%.2f")
axes[0, 1].set_ylabel("Success Rate")
axes[0, 1].set_ylim(0, 1)

# Feasibility Rate by Primitive
feas_by_prim = df.groupby("primitive_name")["feasible"].agg(["mean", "count"]).sort_values("mean")
axes[1, 0].set_title("Feasibility Rate by Primitive")
axes[1, 0].barh(feas_by_prim.index, feas_by_prim["mean"])
axes[1, 0].set_xlabel("Success Rate")
axes[1, 0].set_xlim(0, 1)

# Feasibility Rate by Object Size
df["size_category"] = pd.cut(df["object_volume"], bins=5, labels=["XS", "S", "M", "L", "XL"])
feas_by_size = df.groupby("size_category", observed=False)["feasible"].agg(["mean", "count"])
axes[1, 1].set_title("Feasibility Rate by Object Size")
axes[1, 1].bar(feas_by_size.index, feas_by_size["mean"])
axes[1, 1].set_ylabel("Success Rate")
axes[1, 1].set_xlabel("Object Size Category")
axes[1, 1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_OUTPUT_PATH, "feasibility_analysis.png"))

# --- 3. Pose Error Analysis (Raw and Filtered) ---
df_feasible = df[df["feasible"]].copy()
pos_errors, ori_errors = compute_pose_errors(df_feasible)
df_feasible["position_error"] = pos_errors
df_feasible["orientation_error_deg"] = np.degrees(ori_errors)

pos_outliers = detect_outliers(pos_errors, factor=2.5)
ori_outliers = detect_outliers(ori_errors, factor=2.5)
outlier_mask = pos_outliers | ori_outliers
df_clean = df_feasible[~outlier_mask]

for is_filtered in [False, True]:
    subset_df = df_clean if is_filtered else df_feasible
    title_suffix = f"FILTERED ({outlier_mask.sum()} outliers removed)" if is_filtered else "RAW"
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"Pose Accuracy Analysis - {title_suffix}", fontsize=16)

    # Row 0: Distributions and Outlier Plot
    sns.histplot(data=subset_df, x="position_error", ax=axes[0, 0], bins=50)
    axes[0, 0].set_title("Position Error Distribution")
    axes[0, 0].set_xlabel("Position Error (m)")

    sns.histplot(data=subset_df, x="orientation_error_deg", ax=axes[0, 1], bins=50)
    axes[0, 1].set_title("Orientation Error Distribution")
    axes[0, 1].set_xlabel("Orientation Error (degrees)")

    # Outlier/Correlation Plot (only show on raw data plot)
    if not is_filtered:
        sns.scatterplot(
            data=df_feasible,
            x="position_error",
            y="orientation_error_deg",
            hue=np.where(outlier_mask, "Outlier", "Normal"),
            palette={"Normal": "blue", "Outlier": "red"},
            alpha=0.6,
            ax=axes[0, 2],
        )
        axes[0, 2].set_title("Outlier Identification")
    else:
        sns.scatterplot(data=subset_df, x="position_error", y="orientation_error_deg", alpha=0.6, ax=axes[0, 2])
        axes[0, 2].set_title("Position vs Orientation Error")

    # Row 1: Boxplots by primitive type
    sns.boxplot(data=subset_df, x="primitive_type", y="position_error", ax=axes[1, 0])
    axes[1, 0].set_title("Position Error by Primitive Type")
    axes[1, 0].set_ylabel("Position Error (m)")

    sns.boxplot(data=subset_df, x="primitive_type", y="orientation_error_deg", ax=axes[1, 1])
    axes[1, 1].set_title("Orientation Error by Primitive Type")
    axes[1, 1].set_ylabel("Orientation Error (degrees)")

    # Mean error by specific primitive
    prim_pos_errors = subset_df.groupby("primitive_name")["position_error"].mean().sort_values()
    axes[1, 2].barh(prim_pos_errors.index, prim_pos_errors.values)
    axes[1, 2].set_title("Mean Position Error by Primitive")
    axes[1, 2].set_xlabel("Position Error (m)")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_OUTPUT_PATH, f"pose_error_{'filtered' if is_filtered else 'raw'}.png"))


# --- 4. Primitive Correlation Analysis ---
# Pivot the table to have primitives as columns and feasibility as values
df_pivot = df.pivot_table(index=["scene_id", "object_name"], columns="primitive_name", values="feasible").astype(int)

# Calculate the correlation matrix
corr_matrix = df_pivot.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.title("Correlation Matrix of Primitive Feasibility", fontsize=16)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_OUTPUT_PATH, "primitive_correlation_heatmap.png"))
