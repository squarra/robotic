import h5py
import numpy as np
import time

from robotic.manipulation import Manipulation
from robotic.scenario import PandaScenario

config = PandaScenario()
camera_positions = config.camera_positions
num_scenes = 2

# Cumulative timers
timing = {
    "scene_setup": 0.0,
    "depths": 0.0,
    "io": 0.0,
    "solve": 0.0,
    "total": 0.0,
}

with h5py.File("dataset.h5", "w") as f:
    for scene_id in range(num_scenes):
        t0 = time.time()

        # Scene setup
        t_start = time.time()
        config.delete_man_frames()
        config.add_boxes_to_scene((2, 10), seed=scene_id)
        timing["scene_setup"] += time.time() - t_start

        # Depth + segmentation
        t_start = time.time()
        depths, seg_ids = config.compute_depths_and_seg_ids()
        timing["depths"] += time.time() - t_start

        # HDF5 writing (depths + camera positions)
        t_start = time.time()
        dp_group = f.create_group(f"datapoint_{scene_id:04d}")
        dp_group.create_dataset("depths", data=depths, compression="gzip", chunks=True)
        dp_group.create_dataset("camera_positions", data=camera_positions)
        timing["io"] += time.time() - t_start

        # Manipulations
        man_group = dp_group.create_group("manipulations")
        for obj in config.man_frames:
            man = Manipulation(config, obj, slices=10)
            print(object.__repr__(man.config))
            man_frame = config.getFrame(obj)
            masks = (seg_ids == man_frame.ID).astype(np.uint8)

            obj_group = man_group.create_group(obj)
            obj_group.create_dataset("masks", data=masks, compression="gzip", chunks=True)

            prim_group = obj_group.create_group("primitives")
            for primitive in man.primitives:
                man.reset()
                primitive()
                t_start = time.time()
                ret = man.solve()
                timing["solve"] += time.time() - t_start
                prim_group.create_dataset(f"{primitive.__name__}_feasible", data=ret.feasible)

        timing["total"] += time.time() - t0

# Print summary
print("\n=== Timing summary ===")
for k, v in timing.items():
    print(f"{k:12s}: {v:.3f} s")
