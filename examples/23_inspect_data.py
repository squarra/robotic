import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

DATASET_PATH = "dataset.h5"


def load_dataset(dataset_path):
    data = []
    with h5py.File(dataset_path, "r") as f:
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
                            "object_name": obj_name,
                            "object_size": obj_size,
                            "object_volume": np.prod(obj_size),
                            "primitive_name": primitive_name,
                            "primitive_type": primitive_name.split("_")[0],
                            "feasible": bool(prim_data["feasible"][()]),
                            "final_pose": prim_data["final_pose"][:],
                            "target_pose": prim_data["target_pose"][:],
                        }
                    )
    return pd.DataFrame(data)


def quaternion_distance(q1, q2):
    """Compute angular distance between two quaternions in radians."""
    return 2 * np.arccos(np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0))


def compute_pose_errors(df_feasible):
    """Compute position and orientation errors for feasible manipulations."""
    position_errors = []
    orientation_errors = []

    for _, row in df_feasible.iterrows():
        target_pos = row["target_pose"][:3]
        final_pos = row["final_pose"][:3]
        target_quat = row["target_pose"][3:]
        final_quat = row["final_pose"][3:]

        position_errors.append(np.linalg.norm(final_pos - target_pos))
        orientation_errors.append(quaternion_distance(target_quat, final_quat))

    return np.array(position_errors), np.array(orientation_errors)


def detect_outliers(data, method="iqr", factor=1.5):
    """Detect outliers using IQR or percentile method."""
    if method == "iqr":
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        return (data < lower_bound) | (data > upper_bound)
    elif method == "percentile":
        lower_bound = np.percentile(data, factor)
        upper_bound = np.percentile(data, 100 - factor)
        return (data < lower_bound) | (data > upper_bound)
    elif method == "zscore":
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        return z_scores > factor


def analyze_outliers(df_feasible, position_errors, orientation_errors):
    """Analyze outlier characteristics."""
    pos_outliers = detect_outliers(position_errors, method="iqr", factor=2.0)
    ori_outliers = detect_outliers(orientation_errors, method="iqr", factor=2.0)

    df_with_errors = df_feasible.copy()
    df_with_errors["position_error"] = position_errors
    df_with_errors["orientation_error"] = orientation_errors
    df_with_errors["pos_outlier"] = pos_outliers
    df_with_errors["ori_outlier"] = ori_outliers

    return pos_outliers, ori_outliers


def create_visualizations(df: pd.DataFrame):
    """Create comprehensive visualizations of the dataset."""

    # 1. Feasibility Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Manipulation Feasibility Analysis", fontsize=16)

    # Overall feasibility rate
    feasibility_rate = df["feasible"].mean()
    axes[0, 0].pie(
        [feasibility_rate, 1 - feasibility_rate],
        labels=[f"Feasible ({feasibility_rate:.1%})", f"Not Feasible ({1 - feasibility_rate:.1%})"],
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[0, 0].set_title("Overall Feasibility Rate")

    # Feasibility by primitive type
    feasibility_by_type = df.groupby("primitive_type")["feasible"].agg(["mean", "count"])
    axes[0, 1].bar(feasibility_by_type.index, feasibility_by_type["mean"])
    axes[0, 1].set_title("Feasibility Rate by Primitive Type")
    axes[0, 1].set_ylabel("Success Rate")
    axes[0, 1].set_ylim(0, 1)
    for i, (idx, row) in enumerate(feasibility_by_type.iterrows()):
        axes[0, 1].text(i, row["mean"] + 0.02, f"{row['mean']:.1%}\n(n={row['count']})", ha="center", va="bottom")

    # Feasibility by specific primitive
    feasibility_by_prim = df.groupby("primitive_name")["feasible"].agg(["mean", "count"])
    feasibility_by_prim = feasibility_by_prim.sort_values("mean", ascending=True)

    y_pos = np.arange(len(feasibility_by_prim))
    axes[1, 0].barh(y_pos, feasibility_by_prim["mean"])
    axes[1, 0].set_yticks(y_pos)
    axes[1, 0].set_yticklabels(feasibility_by_prim.index)
    axes[1, 0].set_xlabel("Success Rate")
    axes[1, 0].set_title("Feasibility Rate by Primitive")
    axes[1, 0].set_xlim(0, 1)

    # Feasibility vs object size
    df["size_category"] = pd.cut(df["object_volume"], bins=5, labels=["XS", "S", "M", "L", "XL"])
    feasibility_by_size = df.groupby("size_category", observed=False)["feasible"].agg(["mean", "count"])
    axes[1, 1].bar(range(len(feasibility_by_size)), feasibility_by_size["mean"])
    axes[1, 1].set_xticks(range(len(feasibility_by_size)))
    axes[1, 1].set_xticklabels(feasibility_by_size.index)
    axes[1, 1].set_title("Feasibility Rate by Object Size")
    axes[1, 1].set_ylabel("Success Rate")
    axes[1, 1].set_xlabel("Object Size Category")
    axes[1, 1].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

    # 2. Raw Pose Error Analysis (including outliers)
    df_feasible = df[df["feasible"]].copy()
    position_errors, orientation_errors = compute_pose_errors(df_feasible)

    # Analyze outliers
    pos_outliers, ori_outliers = analyze_outliers(df_feasible, position_errors, orientation_errors)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Pose Accuracy Analysis - RAW DATA (Including Outliers)", fontsize=16)

    # Position error distribution
    axes[0, 0].hist(position_errors, bins=50, alpha=0.7, edgecolor="black")
    axes[0, 0].set_title("Position Error Distribution (Raw)")
    axes[0, 0].set_xlabel("Position Error (m)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].axvline(np.mean(position_errors), color="red", linestyle="--", label=f"Mean: {np.mean(position_errors):.4f}m")
    axes[0, 0].axvline(np.median(position_errors), color="green", linestyle="--", label=f"Median: {np.median(position_errors):.4f}m")
    axes[0, 0].legend()

    # Orientation error distribution
    axes[0, 1].hist(np.degrees(orientation_errors), bins=50, alpha=0.7, edgecolor="black")
    axes[0, 1].set_title("Orientation Error Distribution (Raw)")
    axes[0, 1].set_xlabel("Orientation Error (degrees)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].axvline(
        np.degrees(np.mean(orientation_errors)), color="red", linestyle="--", label=f"Mean: {np.degrees(np.mean(orientation_errors)):.1f}°"
    )
    axes[0, 1].axvline(
        np.degrees(np.median(orientation_errors)), color="green", linestyle="--", label=f"Median: {np.degrees(np.median(orientation_errors)):.1f}°"
    )
    axes[0, 1].legend()

    # Outlier identification plot
    colors = ["red" if outlier else "blue" for outlier in pos_outliers]
    axes[0, 2].scatter(position_errors, np.degrees(orientation_errors), c=colors, alpha=0.6)
    axes[0, 2].set_xlabel("Position Error (m)")
    axes[0, 2].set_ylabel("Orientation Error (degrees)")
    axes[0, 2].set_title("Outlier Identification")
    axes[0, 2].legend(["Normal", "Position Outlier"], loc="upper right")

    # Position error by primitive type (raw)
    df_feasible["position_error"] = position_errors
    df_feasible["orientation_error"] = np.degrees(orientation_errors)

    sns.boxplot(data=df_feasible, x="primitive_type", y="position_error", ax=axes[1, 0])
    axes[1, 0].set_title("Position Error by Primitive Type (Raw)")
    axes[1, 0].set_ylabel("Position Error (m)")

    # Orientation error by primitive type (raw)
    sns.boxplot(data=df_feasible, x="primitive_type", y="orientation_error", ax=axes[1, 1])
    axes[1, 1].set_title("Orientation Error by Primitive Type (Raw)")
    axes[1, 1].set_ylabel("Orientation Error (degrees)")

    # Error correlation (raw)
    axes[1, 2].scatter(position_errors, orientation_errors, alpha=0.6)
    axes[1, 2].set_xlabel("Position Error (m)")
    axes[1, 2].set_ylabel("Orientation Error (rad)")
    axes[1, 2].set_title("Position vs Orientation Error (Raw)")
    corr_coef = np.corrcoef(position_errors, orientation_errors)[0, 1]
    axes[1, 2].text(0.05, 0.95, f"Correlation: {corr_coef:.3f}", transform=axes[1, 2].transAxes, bbox=dict(boxstyle="round", facecolor="wheat"))

    plt.tight_layout()
    plt.show()

    # 3. Filtered Pose Error Analysis (outliers removed)
    # Filter outliers using IQR method
    outlier_mask = pos_outliers | ori_outliers
    clean_pos_errors = position_errors[~outlier_mask]
    clean_ori_errors = orientation_errors[~outlier_mask]
    df_clean = df_feasible[~outlier_mask].copy()
    df_clean["position_error"] = clean_pos_errors
    df_clean["orientation_error"] = np.degrees(clean_ori_errors)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Pose Accuracy Analysis - FILTERED DATA ({outlier_mask.sum()} outliers removed)", fontsize=16)

    # Position error distribution (clean)
    axes[0, 0].hist(clean_pos_errors, bins=30, alpha=0.7, edgecolor="black")
    axes[0, 0].set_title("Position Error Distribution (Filtered)")
    axes[0, 0].set_xlabel("Position Error (m)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].axvline(np.mean(clean_pos_errors), color="red", linestyle="--", label=f"Mean: {np.mean(clean_pos_errors):.4f}m")
    axes[0, 0].axvline(np.median(clean_pos_errors), color="green", linestyle="--", label=f"Median: {np.median(clean_pos_errors):.4f}m")
    axes[0, 0].legend()

    # Orientation error distribution (clean)
    axes[0, 1].hist(np.degrees(clean_ori_errors), bins=30, alpha=0.7, edgecolor="black")
    axes[0, 1].set_title("Orientation Error Distribution (Filtered)")
    axes[0, 1].set_xlabel("Orientation Error (degrees)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].axvline(
        np.degrees(np.mean(clean_ori_errors)), color="red", linestyle="--", label=f"Mean: {np.degrees(np.mean(clean_ori_errors)):.1f}°"
    )
    axes[0, 1].axvline(
        np.degrees(np.median(clean_ori_errors)), color="green", linestyle="--", label=f"Median: {np.degrees(np.median(clean_ori_errors)):.1f}°"
    )
    axes[0, 1].legend()

    # Position error by primitive type (clean)
    sns.boxplot(data=df_clean, x="primitive_type", y="position_error", ax=axes[0, 2])
    axes[0, 2].set_title("Position Error by Primitive Type (Filtered)")
    axes[0, 2].set_ylabel("Position Error (m)")

    # Orientation error by primitive type (clean)
    sns.boxplot(data=df_clean, x="primitive_type", y="orientation_error", ax=axes[1, 0])
    axes[1, 0].set_title("Orientation Error by Primitive Type (Filtered)")
    axes[1, 0].set_ylabel("Orientation Error (degrees)")

    # Position error by specific primitive (clean)
    prim_pos_errors = df_clean.groupby("primitive_name")["position_error"].mean().sort_values()
    axes[1, 1].barh(range(len(prim_pos_errors)), prim_pos_errors.values)
    axes[1, 1].set_yticks(range(len(prim_pos_errors)))
    axes[1, 1].set_yticklabels(prim_pos_errors.index)
    axes[1, 1].set_title("Mean Position Error by Primitive (Filtered)")
    axes[1, 1].set_xlabel("Position Error (m)")

    # Error correlation (clean)
    axes[1, 2].scatter(clean_pos_errors, clean_ori_errors, alpha=0.6)
    axes[1, 2].set_xlabel("Position Error (m)")
    axes[1, 2].set_ylabel("Orientation Error (rad)")
    axes[1, 2].set_title("Position vs Orientation Error (Filtered)")
    clean_corr_coef = np.corrcoef(clean_pos_errors, clean_ori_errors)[0, 1]
    axes[1, 2].text(
        0.05, 0.95, f"Correlation: {clean_corr_coef:.3f}", transform=axes[1, 2].transAxes, bbox=dict(boxstyle="round", facecolor="lightgreen")
    )

    plt.tight_layout()
    plt.show()


df = load_dataset(DATASET_PATH)
print(f"Loaded {len(df)} manipulation attempts from {df['scene_id'].nunique()} scenes")
create_visualizations(df)
