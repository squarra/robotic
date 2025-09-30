from multiprocessing import Pool

import h5py
import numpy as np
from tqdm import trange

from robotic.manipulation import Manipulation
from robotic.scenario import PandaScenario

DATASET_PATH = "dataset.h5"
NUM_SCENES = 100
START_SEED = 0
SLICES = 10  # fewer slices = faster but less accurate
INCREMENTAL_SLICES = True

offset_directions = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]
num_offsets = len(offset_directions)
primitives = Manipulation.primitives
num_primitives = len(primitives)
config = PandaScenario()


def solve_primitive(args):
    oi, obj, ti, target_pose, pi = args

    man = Manipulation(config, obj, slices=SLICES)
    getattr(man, primitives[pi])()
    man.target_pose(target_pose)
    feasible = man.solve()
    if feasible and INCREMENTAL_SLICES:
        man = Manipulation(config, obj, slices=SLICES * 2)
        getattr(man, primitive)()
        man.target_pose(target_pose)
        feasible = man.solve().feasible
    if feasible:
        man.simulate(view=False)

    final_pose = man.config.getFrame(obj).getRelativePose()
    return oi, ti, pi, feasible, final_pose


with h5py.File(DATASET_PATH, "w") as f:
    f.attrs["primitives"] = np.array(primitives, dtype=h5py.string_dtype(encoding="utf-8"))

    for seed in trange(START_SEED, START_SEED + NUM_SCENES, desc="Collecting data"):
        config.delete_man_frames()
        config.add_boxes(seed=seed)

        num_objects = len(config.man_frames)

        poses = np.zeros((num_objects, 7), dtype=np.float32)
        sizes = np.zeros((num_objects, 3), dtype=np.float32)
        target_poses = np.zeros((num_objects, num_offsets, 7), dtype=np.float32)
        feasibles = np.zeros((num_objects, num_offsets, num_primitives), dtype=np.int8)
        final_poses = np.zeros((num_objects, num_offsets, num_primitives, 7), dtype=np.float32)

        jobs = []
        for oi, obj in enumerate(config.man_frames):
            frame = config.getFrame(obj)
            poses[oi] = frame.getRelativePose()
            sizes[oi] = frame.getSize()[:3]

            for ti, direction in enumerate(offset_directions):
                target_pos = config.sample_target_pos(obj, direction, seed=seed)
                target_pose = np.concatenate([target_pos, frame.getRelativeQuaternion()])
                target_poses[oi][ti] = target_pose

                for pi, primitive in enumerate(primitives):
                    jobs.append((oi, obj, ti, target_pose, pi))

        with Pool() as pool:
            for oi, ti, pi, feasible, final_pose in pool.map(solve_primitive, jobs):
                feasibles[oi, ti, pi] = feasible
                final_poses[oi, ti, pi] = final_pose

        images, depths, seg_ids = config.compute_images_depths_and_seg_ids()
        dp = f.create_group(f"dp_{seed:04d}")
        dp.create_dataset("cam_poses", data=config.cam_poses)
        dp.create_dataset("images", data=images, compression="gzip", chunks=True)
        dp.create_dataset("depths", data=depths, compression="gzip", chunks=True)
        dp.create_dataset("seg_ids", data=seg_ids, compression="gzip", chunks=True)
        dp.create_dataset("poses", data=poses)
        dp.create_dataset("sizes", data=sizes)
        dp.create_dataset("target_poses", data=target_poses)
        dp.create_dataset("feasibles", data=feasibles)
        dp.create_dataset("final_poses", data=final_poses)
