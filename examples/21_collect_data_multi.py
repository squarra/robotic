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

from robotic._robotic import setLogLevel
from robotic.manipulation import Manipulation
from robotic.scenario import PandaScenario

config = PandaScenario()
num_scenes = 1

setLogLevel(-1)


def solve_primitive(args):
    obj, primitive = args
    man = Manipulation(config, obj, slices=10)  # fewer slices = more speed
    getattr(man, primitive)()
    ret = man.solve()
    return obj, primitive, ret.feasible


with h5py.File("dataset.h5", "w") as f:
    for scene_id in tqdm.trange(num_scenes):
        config.delete_man_frames()
        config.add_boxes_to_scene((2, 10), seed=scene_id)
        depths, seg_ids = config.compute_depths_and_seg_ids()

        dp_group = f.create_group(f"datapoint_{scene_id:04d}")
        dp_group.create_dataset("depths", data=depths, compression="gzip", chunks=True)
        dp_group.create_dataset("camera_positions", data=config.camera_positions)
        man_group = dp_group.create_group("manipulations")

        jobs = []
        for obj in config.man_frames:
            man_frame = config.getFrame(obj)
            masks = (seg_ids == man_frame.ID).astype(np.uint8)

            obj_group = man_group.create_group(obj)
            obj_group.create_dataset("masks", data=masks, compression="gzip", chunks=True)
            prim_group = obj_group.create_group("primitives")

            for primitive in Manipulation.primitives:
                jobs.append((obj, primitive))

        with Pool() as pool:
            results = pool.map(solve_primitive, jobs)

        for obj, primitive_name, feasible in results:
            prim_group = man_group[obj]["primitives"]
            prim_group.create_dataset(primitive_name, data=feasible)
