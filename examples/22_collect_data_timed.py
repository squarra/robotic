import time

import h5py
import numpy as np

from robotic.manipulation import Manipulation
from robotic.scenario import PandaScenario

DATASET_PATH = "dataset.h5"
SEED = 0
SLICES = 10  # fewer slices = faster but less accurate
INCREMENTAL_SLICES = True

# offset_directions = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]
offset_directions = [[1, 0, 0]]
num_offsets = len(offset_directions)
primitives = Manipulation.primitives
num_primitives = len(primitives)

solve_first = []
solve_incre = []
simulate_times = []
disk_io_times = []

config = PandaScenario()
config.add_boxes(density=500, seed=SEED)
images, depths, seg_ids = config.compute_images_depths_and_seg_ids()

num_objects = len(config.man_frames)

poses = np.zeros((num_objects, 7), dtype=np.float32)
sizes = np.zeros((num_objects, 3), dtype=np.float32)
target_poses = np.zeros((num_objects, num_offsets, 7), dtype=np.float32)
feasibles = np.zeros((num_objects, num_offsets, num_primitives), dtype=np.int8)
final_poses = np.zeros((num_objects, num_offsets, num_primitives, 7), dtype=np.float32)

for oi, obj in enumerate(config.man_frames):
    frame = config.getFrame(obj)
    poses[oi] = frame.getRelativePose()
    sizes[oi] = frame.getSize()[:3]

    for ti, direction in enumerate(offset_directions):
        target_pos = config.sample_target_pos(obj, direction, seed=SEED)
        target_pose = np.concatenate([target_pos, frame.getRelativeQuaternion()])
        target_poses[oi][ti] = target_pose

        for pi, primitive in enumerate(primitives):
            man = Manipulation(config, obj, slices=SLICES)
            getattr(man, primitive)()
            man.target_pose(target_pose)

            t0 = time.perf_counter()
            feasible = man.solve().feasible
            t1 = time.perf_counter()
            dt = t1 - t0
            solve_first.append(dt)

            if feasible and INCREMENTAL_SLICES:
                man = Manipulation(config, obj, slices=SLICES * 2)
                getattr(man, primitive)()
                man.target_pose(target_pose)

                t0 = time.perf_counter()
                feasible = man.solve().feasible
                t1 = time.perf_counter()
                dt = t1 - t0
                solve_incre.append(dt)

            if feasible:
                t0 = time.perf_counter()
                man.simulate(view=False)
                t1 = time.perf_counter()
                simulate_times.append(t1 - t0)

            feasibles[oi][ti][pi] = feasible
            final_poses[oi][ti][pi] = man.config.getFrame(obj).getRelativePose()


t0 = time.perf_counter()
with h5py.File(DATASET_PATH, "w") as f:
    f.attrs["primitives"] = np.array(primitives, dtype=h5py.string_dtype(encoding="utf-8"))
    dp = f.create_group(f"dp_{SEED:04d}")
    dp.create_dataset("cam_poses", data=config.cam_poses)
    dp.create_dataset("images", data=images, compression="gzip", chunks=True)
    dp.create_dataset("depths", data=depths, compression="gzip", chunks=True)
    dp.create_dataset("seg_ids", data=seg_ids, compression="gzip", chunks=True)
    dp.create_dataset("poses", data=poses)
    dp.create_dataset("sizes", data=sizes)
    dp.create_dataset("target_poses", data=target_poses)
    dp.create_dataset("feasibles", data=feasibles)
    dp.create_dataset("final_poses", data=final_poses)
t1 = time.perf_counter()
disk_io_times.append(t1 - t0)

print("\n=== RESULTS ===")
print(f"solve() first calls: {len(solve_first)}, avg={np.mean(solve_first):.4f}s, total={np.sum(solve_first):.2f}s")
print(f"solve() incre calls: {len(solve_incre)}, avg={np.mean(solve_incre) if solve_incre else 0:.4f}s, total={np.sum(solve_incre):.2f}s")
print(f"simulate() calls:    {len(simulate_times)}, avg={np.mean(simulate_times)}")
print(f"Simulate time: avg={np.mean(simulate_times):.4f}s, total={np.sum(simulate_times):.2f}s")
print(f"Disk I/O time (group & datasets): {np.sum(disk_io_times):.4f}s")
