import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("TkAgg")
plt.style.use("seaborn-v0_8")

DATASET_PATH = "dataset-0000-2000.h5"


def compute_pose_errors(df_feasible: pd.DataFrame):
    poses_target = np.vstack(df_feasible["target_pose"].to_numpy())
    poses_final = np.vstack(df_feasible["final_pose"].to_numpy())

    target_pos = poses_target[:, :3]
    final_pos = poses_final[:, :3]
    target_quat = poses_target[:, 3:]
    final_quat = poses_final[:, 3:]

    position_errors = np.linalg.norm(final_pos - target_pos, axis=1)
    orientation_errors = 2 * np.arccos(np.clip(np.abs(np.sum(target_quat * final_quat, axis=1)), -1.0, 1.0))
    return position_errors, orientation_errors


def detect_outliers(data, factor=1.5):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return (data < lower_bound) | (data > upper_bound)


data = []
with h5py.File(DATASET_PATH, "r") as f:
    for scene_key in f.keys():
        scene_id = int(scene_key.split("_")[1])
        objects_group = f[scene_key]["objects"]

        for obj_name in objects_group.keys():
            obj_data = objects_group[obj_name]
            obj_size = obj_data["size"][:]

            primitives_group = obj_data["primitives"]
            for primitive_name in primitives_group.keys():
                prim_data = primitives_group[primitive_name]

                data.append(
                    {
                        "scene_id": scene_id,
                        "object_volume": np.prod(obj_size),
                        "primitive_name": primitive_name,
                        "primitive_type": primitive_name.split("_")[0],
                        "feasible": bool(prim_data["feasible"][()]),
                        "final_pose": prim_data["final_pose"][:],
                        "target_pose": prim_data["target_pose"][:],
                    }
                )

df = pd.DataFrame(data)
print(f"Loaded {len(df)} manipulation attempts from {df["scene_id"].nunique()} scenes")

# 1. Feasibility Analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Feasibility Analysis", fontsize=16)

# Overall feasibility rate
feas_rate = df["feasible"].mean()
axes[0, 0].set_title("Overall Feasibility Rate")
axes[0, 0].pie([feas_rate, 1 - feas_rate], labels=["Feasible", "Not Feasible"], autopct="%1.1f%%", startangle=90)

# Feasibility Rate by Primitive Type
feasibility_by_type = df.groupby("primitive_type")["feasible"].agg(["mean", "count"])
axes[0, 1].set_title("Feasibility Rate by Primitive Type")
axes[0, 1].bar(feasibility_by_type.index, feasibility_by_type["mean"])
axes[0, 1].bar_label(axes[0, 1].containers[0], fmt="%.2f")
axes[0, 1].set_ylabel("Success Rate")
axes[0, 1].set_ylim(0, 1)

# Feasibility Rate by Primitive
feas_by_prim = df.groupby("primitive_name")["feasible"].agg(["mean", "count"]).sort_values("mean")
y_pos = np.arange(len(feas_by_prim))
axes[1, 0].set_title("Feasibility Rate by Primitive")
axes[1, 0].barh(y_pos, feas_by_prim["mean"])
axes[1, 0].set_yticks(y_pos)
axes[1, 0].set_yticklabels(feas_by_prim.index)
axes[1, 0].set_xlabel("Success Rate")
axes[1, 0].set_xlim(0, 1)

# Feasibility Rate by Object Size
df["size_category"] = pd.cut(df["object_volume"], bins=5, labels=["XS", "S", "M", "L", "XL"])
feas_by_size = df.groupby("size_category", observed=False)["feasible"].agg(["mean", "count"])
axes[1, 1].set_title("Feasibility Rate by Object Size")
axes[1, 1].bar(range(len(feas_by_size)), feas_by_size["mean"])
axes[1, 1].set_xticks(range(len(feas_by_size)))
axes[1, 1].set_xticklabels(feas_by_size.index)
axes[1, 1].set_ylabel("Success Rate")
axes[1, 1].set_xlabel("Object Size Category")
axes[1, 1].set_ylim(0, 1)

plt.tight_layout()
plt.show()

# 2. Raw Pose Error Analysis
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Pose Accuracy Analysis - RAW DATA", fontsize=16)

df_feasible = df[df["feasible"]].copy()
pos_errors, ori_errors = compute_pose_errors(df_feasible)

# Position error distribution
axes[0, 0].hist(pos_errors, bins=50, alpha=0.7, edgecolor="black")
axes[0, 0].set_title("Position Error Distribution")
axes[0, 0].set_xlabel("Position Error (m)")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].axvline(np.mean(pos_errors), color="red", label=f"Mean: {np.mean(pos_errors):.4f}m")
axes[0, 0].axvline(np.median(pos_errors), color="green", label=f"Median: {np.median(pos_errors):.4f}m")
axes[0, 0].legend()

# Orientation error distribution
axes[0, 1].hist(np.degrees(ori_errors), bins=50, alpha=0.7, edgecolor="black")
axes[0, 1].set_title("Orientation Error Distribution")
axes[0, 1].set_xlabel("Orientation Error (degrees)")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].axvline(np.degrees(np.mean(ori_errors)), color="red", label=f"Mean: {np.degrees(np.mean(ori_errors)):.1f}°")
axes[0, 1].axvline(np.degrees(np.median(ori_errors)), color="green", label=f"Median: {np.degrees(np.median(ori_errors)):.1f}°")
axes[0, 1].legend()

# Analyze outliers
pos_outliers = detect_outliers(pos_errors, factor=2.0)
ori_outliers = detect_outliers(ori_errors, factor=2.0)

# Outlier identification plot
colors = ["red" if outlier else "blue" for outlier in pos_outliers]
axes[0, 2].set_title("Outlier Identification")
axes[0, 2].scatter(pos_errors, np.degrees(ori_errors), c=colors, alpha=0.6)
axes[0, 2].set_xlabel("Position Error (m)")
axes[0, 2].set_ylabel("Orientation Error (degrees)")
axes[0, 2].legend(["Normal", "Position Outlier"], loc="upper right")

# Position error by primitive type
df_feasible["position_error"] = pos_errors
df_feasible["orientation_error"] = np.degrees(ori_errors)

sns.boxplot(data=df_feasible, x="primitive_type", y="position_error", ax=axes[1, 0])
axes[1, 0].set_title("Position Error by Primitive Type ")
axes[1, 0].set_ylabel("Position Error (m)")

# Orientation error by primitive type
sns.boxplot(data=df_feasible, x="primitive_type", y="orientation_error", ax=axes[1, 1])
axes[1, 1].set_title("Orientation Error by Primitive Type ")
axes[1, 1].set_ylabel("Orientation Error (degrees)")

# Error correlation
axes[1, 2].scatter(pos_errors, ori_errors, alpha=0.6)
axes[1, 2].set_xlabel("Position Error (m)")
axes[1, 2].set_ylabel("Orientation Error (rad)")
axes[1, 2].set_title("Position vs Orientation Error")
corr_coef = np.corrcoef(pos_errors, ori_errors)[0, 1]
axes[1, 2].text(0.05, 0.95, f"Correlation: {corr_coef:.3f}", transform=axes[1, 2].transAxes, bbox=dict(facecolor="wheat"))

plt.tight_layout()
plt.show()

# 3. Filtered Pose Error Analysis
outlier_mask = pos_outliers | ori_outliers
clean_pos_errors = pos_errors[~outlier_mask]
clean_ori_errors = ori_errors[~outlier_mask]
df_clean = df_feasible[~outlier_mask].copy()
df_clean["position_error"] = clean_pos_errors
df_clean["orientation_error"] = np.degrees(clean_ori_errors)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f"Pose Accuracy Analysis - FILTERED DATA ({outlier_mask.sum()} outliers removed)", fontsize=16)

# Position error distribution (clean)
axes[0, 0].hist(clean_pos_errors, bins=30, alpha=0.7, edgecolor="black")
axes[0, 0].set_title("Position Error Distribution")
axes[0, 0].set_xlabel("Position Error (m)")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].axvline(np.mean(clean_pos_errors), color="red", label=f"Mean: {np.mean(clean_pos_errors):.4f}m")
axes[0, 0].axvline(np.median(clean_pos_errors), color="green", label=f"Median: {np.median(clean_pos_errors):.4f}m")
axes[0, 0].legend()

# Orientation error distribution (clean)
axes[0, 1].hist(np.degrees(clean_ori_errors), bins=30, alpha=0.7, edgecolor="black")
axes[0, 1].set_title("Orientation Error Distribution")
axes[0, 1].set_xlabel("Orientation Error (degrees)")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].axvline(np.degrees(np.mean(clean_ori_errors)), color="red", label=f"Mean: {np.degrees(np.mean(clean_ori_errors)):.1f}°")
axes[0, 1].axvline(np.degrees(np.median(clean_ori_errors)), color="green", label=f"Median: {np.degrees(np.median(clean_ori_errors)):.1f}°")
axes[0, 1].legend()

# Position error by primitive type (clean)
sns.boxplot(data=df_clean, x="primitive_type", y="position_error", ax=axes[0, 2])
axes[0, 2].set_title("Position Error by Primitive Type")
axes[0, 2].set_ylabel("Position Error (m)")

# Orientation error by primitive type (clean)
sns.boxplot(data=df_clean, x="primitive_type", y="orientation_error", ax=axes[1, 0])
axes[1, 0].set_title("Orientation Error by Primitive Type")
axes[1, 0].set_ylabel("Orientation Error (degrees)")

# Position error by specific primitive (clean)
prim_pos_errors = df_clean.groupby("primitive_name")["position_error"].mean().sort_values()
axes[1, 1].set_title("Mean Position Error by Primitive")
axes[1, 1].barh(range(len(prim_pos_errors)), prim_pos_errors.values)
axes[1, 1].set_yticks(range(len(prim_pos_errors)))
axes[1, 1].set_yticklabels(prim_pos_errors.index)
axes[1, 1].set_xlabel("Position Error (m)")

# Error correlation (clean)
axes[1, 2].set_title("Position vs Orientation Error")
axes[1, 2].scatter(clean_pos_errors, clean_ori_errors, alpha=0.6)
axes[1, 2].set_xlabel("Position Error (m)")
axes[1, 2].set_ylabel("Orientation Error (rad)")
clean_corr_coef = np.corrcoef(clean_pos_errors, clean_ori_errors)[0, 1]
axes[1, 2].text(0.05, 0.95, f"Correlation: {clean_corr_coef:.3f}", transform=axes[1, 2].transAxes, bbox=dict(facecolor="lightgreen"))

plt.tight_layout()
plt.show()
