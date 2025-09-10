"""
This example shows how to use multiprocessing when solve() is the main
bottleneck (see 22_collect_data_timed.py). A global config object is
updated per scene, and for each manipulation primitive a Worker is
spawned to call solve(). Since writing to HDF5 must occur in the main
thread, results are collected and written at the end.
"""

from multiprocessing import Pool

import h5py
import numpy as np
import tqdm

from robotic.manipulation import Manipulation
from robotic.scenario import PandaScenario

DATASET_PATH = "dataset.h5"
NUM_SCENES = 2000
SLICES = 10  # fewer slices = more speed
POS_OFFSET = 0.05  # move 5cm along the push/grasp axis

config = PandaScenario()


def solve_primitive(args):
    obj, primitive_name = args
    man = Manipulation(config, obj, slices=SLICES)
    frame = man.config.getFrame(obj)
    if "push" in primitive_name:
        _, primitive_dim, primitive_dir = primitive_name.split("_")
        axis = {"x": 0, "y": 1, "z": 2}[primitive_dim]
        direction = {"pos": 1, "neg": -1}[primitive_dir]
        offset_local = np.zeros(3)
        offset_local[axis] = POS_OFFSET * direction
    elif "grasp" in primitive_name:
        _, primitive_dim = primitive_name.split("_")
        axis = {"x": 0, "y": 1, "z": 2}[primitive_dim]
        offset_local = np.zeros(3)
        offset_local[axis] = POS_OFFSET

    offset_world = frame.getRotationMatrix() @ offset_local
    target_pos = frame.getRelativePosition() + offset_world
    target_pose = np.concatenate([target_pos, frame.getRelativeQuaternion()])
    man.target_pose(target_pose)

    getattr(man, primitive_name)()
    ret = man.solve()
    if ret.feasible:
        man.simulate(view=False)
    final_pose = frame.getRelativePose()
    return obj, primitive_name, target_pose, ret.feasible, final_pose


with h5py.File(DATASET_PATH, "w") as f:
    for scene_id in tqdm.trange(NUM_SCENES):
        config.delete_man_frames()
        config.add_boxes_to_scene(seed=scene_id)

        images, depths, seg_ids = config.compute_images_depths_and_seg_ids()

        dp_group = f.create_group(f"datapoint_{scene_id:04d}")
        dp_group.attrs["seed"] = scene_id
        dp_group.create_dataset("camera_positions", data=config.camera_positions)
        dp_group.create_dataset("images", data=images, compression="gzip", chunks=True)
        dp_group.create_dataset("depths", data=depths, compression="gzip", chunks=True)
        dp_group.create_dataset("seg_ids", data=seg_ids, compression="gzip", chunks=True)

        objects_group = dp_group.create_group("objects")
        jobs = []
        for obj in config.man_frames:
            obj_frame = config.getFrame(obj)
            obj_group = objects_group.create_group(obj)

            obj_group.create_dataset("pose", data=obj_frame.getRelativePose().astype(np.float32))
            obj_group.create_dataset("size", data=obj_frame.getSize()[:3].astype(np.float32))

            obj_group.create_group("primitives")
            for primitive in Manipulation.primitives:
                jobs.append((obj, primitive))

        with Pool() as pool:
            results = pool.map(solve_primitive, jobs)

        for obj, primitive_name, target_pose, feasible, final_pose in results:
            primitive_group = objects_group[obj]["primitives"].create_group(primitive_name)
            primitive_group.create_dataset("target_pose", data=target_pose.astype(np.float32))
            primitive_group.create_dataset("feasible", data=feasible)
            primitive_group.create_dataset("final_pose", data=final_pose.astype(np.float32))
