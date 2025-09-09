"""
This script profiles the data collection pipeline to highlight runtime
bottlenecks. Each stage (scene setup, image/depth computation, I/O, solve,
and simulate) is timed and summarized at the end.

The solve() calls are usually the dominant cost, especially with a high
number of slices. Adjust NUM_SCENES and SLICES to see how runtime scales.
"""

import time

import h5py
import numpy as np

from robotic.manipulation import Manipulation
from robotic.scenario import PandaScenario

DATASET_PATH = "dataset.h5"
SLICES = 10  # fewer slices = more speed

timing = {
    "scene_setup": 0.0,
    "images_depths": 0.0,
    "io": 0.0,
    "solve": 0.0,
    "simulate": 0.0,
    "total": 0.0,
}

config = PandaScenario()
camera_positions = config.camera_positions

with h5py.File(DATASET_PATH, "w") as f:
    t0 = time.time()

    t_start = time.time()
    config.delete_man_frames()
    config.add_boxes_to_scene(seed=42)
    timing["scene_setup"] += time.time() - t_start

    t_start = time.time()
    images, depths, seg_ids = config.compute_images_depths_and_seg_ids()
    timing["images_depths"] += time.time() - t_start

    t_start = time.time()
    dp_group = f.create_group("datapoint_0001")
    dp_group.create_dataset("camera_positions", data=camera_positions)
    dp_group.create_dataset("images", data=images, compression="gzip", chunks=True)
    dp_group.create_dataset("depths", data=depths, compression="gzip", chunks=True)
    dp_group.create_dataset("seg_ids", data=seg_ids, compression="gzip", chunks=True)
    timing["io"] += time.time() - t_start

    objects_group = dp_group.create_group("objects")
    for obj in config.man_frames:
        man_frame = config.getFrame(obj)

        obj_group = objects_group.create_group(obj)
        obj_group.create_dataset("pose", data=man_frame.getRelativePose().astype(np.float32))
        obj_group.create_dataset("size", data=man_frame.getSize()[:3].astype(np.float32))

        primitives_group = obj_group.create_group("primitives")
        for prim in Manipulation.primitives:
            man = Manipulation(config, obj, slices=SLICES)
            getattr(man, prim)()

            t_start = time.time()
            ret = man.solve()
            timing["solve"] += time.time() - t_start

            if ret.feasible:
                t_start = time.time()
                man.simulate(view=False)
                timing["simulate"] += time.time() - t_start

            prim_group = primitives_group.create_group(prim)
            prim_group.create_dataset("feasible", data=ret.feasible)
            prim_group.create_dataset("pose", data=man.config.getFrame(obj).getRelativePose())

    timing["total"] += time.time() - t0

print("\n=== Timing summary ===")
for k, v in timing.items():
    print(f"{k:15s}: {v:.3f} s")
