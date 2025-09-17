from multiprocessing import Pool

import h5py
import numpy as np
import tqdm

from robotic.manipulation import Manipulation
from robotic.scenario import PandaScenario

DATASET_PATH = "dataset-flat.h5"
NUM_SCENES = 1500
START_SEED = 0
SLICES = 10  # fewer slices = faster but less accurate
POS_OFFSET = 0.05  # move 5cm along the push/grasp axis

primitives = Manipulation.primitives
config = PandaScenario()


def solve_primitive(args):
    oi, obj, pi, primitive_name = args

    man = Manipulation(config, obj, slices=SLICES)
    frame = man.config.getFrame(obj)

    _, primitive_dim, primitive_dir = primitive_name.split("_")
    dim = {"x": 0, "y": 1, "z": 2}[primitive_dim]
    dir = {"pos": 1, "neg": -1}[primitive_dir]
    offset_local = np.zeros(3)
    offset_local[dim] = POS_OFFSET * dir
    offset_world = frame.getRotationMatrix() @ offset_local
    target_pos = frame.getRelativePosition() + offset_world
    target_pose = np.concatenate([target_pos, frame.getRelativeQuaternion()])

    man.target_pose(target_pose)
    getattr(man, primitive_name)()
    ret = man.solve()
    if ret.feasible:
        man.simulate(view=False)

    return oi, pi, target_pose, ret.feasible, frame.getRelativePose()


with h5py.File(DATASET_PATH, "w") as f:
    f.attrs["primitives"] = np.array(primitives, dtype=h5py.string_dtype(encoding="utf-8"))

    for i in tqdm.trange(NUM_SCENES, desc="Collecting data"):
        seed = START_SEED + i
        config.delete_man_frames()
        config.add_boxes_to_scene(seed=seed)

        images, depths, seg_ids = config.compute_images_depths_and_seg_ids()

        dp_group = f.create_group(f"datapoint_{seed:04d}")
        dp_group.create_dataset("camera_positions", data=config.camera_positions)
        dp_group.create_dataset("images", data=images, compression="gzip", chunks=True)
        dp_group.create_dataset("depths", data=depths, compression="gzip", chunks=True)
        dp_group.create_dataset("seg_ids", data=seg_ids, compression="gzip", chunks=True)

        num_objects = len(config.man_frames)
        num_primitives = len(primitives)

        poses = np.zeros((num_objects, 7), dtype=np.float32)
        sizes = np.zeros((num_objects, 3), dtype=np.float32)
        target_poses = np.zeros((num_objects, num_primitives, 7), dtype=np.float32)
        feasibles = np.zeros((num_objects, num_primitives), dtype=np.int8)
        final_poses = np.zeros((num_objects, num_primitives, 7), dtype=np.float32)

        jobs = []
        for oi, obj in enumerate(config.man_frames):
            frame = config.getFrame(obj)
            poses[oi] = frame.getRelativePose()
            sizes[oi] = frame.getSize()[:3]

            for pi, primitive in enumerate(primitives):
                jobs.append((oi, obj, pi, primitive))

        with Pool() as pool:
            results = pool.map(solve_primitive, jobs)

        for result_tuple in results:
            oi, pi, target_pose, feasible, final_pose = result_tuple
            target_poses[oi, pi] = target_pose
            feasibles[oi, pi] = feasible
            final_poses[oi, pi] = final_pose

        dp_group.create_dataset("poses", data=poses)
        dp_group.create_dataset("sizes", data=sizes)
        dp_group.create_dataset("target_poses", data=target_poses)
        dp_group.create_dataset("feasibles", data=feasibles)
        dp_group.create_dataset("final_poses", data=final_poses)
