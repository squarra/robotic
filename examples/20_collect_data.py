import h5py
import numpy as np
from tqdm import trange

from robotic.manipulation import Manipulation
from robotic.scenario import PandaScenario

DATASET_PATH = "dataset.h5"
NUM_SCENES = 1
START_SEED = 0
SLICES = 10  # fewer slices = faster but less accurate
INCREMENTAL_SLICES = True

primitives = Manipulation.primitives
num_primitives = len(primitives)


with h5py.File(DATASET_PATH, "w") as f:
    f.attrs["primitives"] = np.array(primitives, dtype=h5py.string_dtype(encoding="utf-8"))

    for seed in trange(START_SEED, START_SEED + NUM_SCENES, desc="Collecting data"):
        config = PandaScenario()
        config.add_boxes(density=500, seed=seed)

        images, depths, seg_ids = config.compute_images_depths_and_seg_ids()

        dp_group = f.create_group(f"dp_{seed:04d}")
        dp_group.create_dataset("cam_poses", data=config.cam_poses)
        dp_group.create_dataset("images", data=images, compression="gzip", chunks=True)
        dp_group.create_dataset("depths", data=depths, compression="gzip", chunks=True)
        dp_group.create_dataset("seg_ids", data=seg_ids, compression="gzip", chunks=True)

        num_objects = len(config.man_frames)

        obj_ids = np.zeros((num_objects), dtype=np.int32)
        poses = np.zeros((num_objects, 7), dtype=np.float32)
        sizes = np.zeros((num_objects, 3), dtype=np.float32)
        target_poses = np.zeros((num_objects, 7), dtype=np.float32)
        feasibles = np.zeros((num_objects, num_primitives), dtype=np.float32)
        final_poses = np.zeros((num_objects, num_primitives, 7), dtype=np.float32)

        for oi, obj in enumerate(config.man_frames):
            frame = config.getFrame(obj)
            obj_ids[oi] = frame.ID
            poses[oi] = frame.getRelativePose()
            sizes[oi] = frame.getSize()[:3]

            target_pose = config.sample_target_pose(obj, seed=seed)
            target_poses[oi] = target_pose
            config.add_marker(target_pose)

            for pi, primitive in enumerate(primitives):
                man = Manipulation(config, obj, slices=SLICES)
                getattr(man, primitive)()
                man.target_pose(target_pose)
                feasible = man.solve().feasible
                if feasible and INCREMENTAL_SLICES:
                    man = Manipulation(config, obj, slices=SLICES * 2)
                    getattr(man, primitive)()
                    man.target_pose(target_pose)
                    feasible = man.solve().feasible
                man.view(pause=True, txt=f"{obj}, {primitive}={feasible}")
                if feasible:
                    man.simulate(view=True)
                feasibles[oi][pi] = feasible
                final_poses[oi][pi] = man.config.getFrame(obj).getRelativePose()

            config.remove_markers()

        dp_group.create_dataset("obj_ids", data=obj_ids)
        dp_group.create_dataset("poses", data=poses)
        dp_group.create_dataset("sizes", data=sizes)
        dp_group.create_dataset("target_poses", data=target_poses)
        dp_group.create_dataset("feasibles", data=feasibles)
        dp_group.create_dataset("final_poses", data=final_poses)
