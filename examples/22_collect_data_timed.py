import time
from collections import defaultdict

import h5py
import numpy as np

from robotic.manipulation import Manipulation
from robotic.scenario import PandaScenario

DATASET_PATH = "dataset.h5"
SLICES = 10  # fewer slices = more speed
POS_OFFSET = 0.05  # move 5cm along the push/grasp axis
SEED = 42

timing_data = {"solve_feasible": [], "solve_infeasible": [], "simulate": [], "per_primitive": defaultdict(list), "per_object": defaultdict(float)}


total_start = time.perf_counter()

config = PandaScenario()
config.add_boxes_to_scene(seed=SEED)
camera_positions = config.camera_positions
images, depths, seg_ids = config.compute_images_depths_and_seg_ids()

with h5py.File(DATASET_PATH, "w") as f:
    dp_group = f.create_group("datapoint_0001")
    dp_group.attrs["seed"] = SEED
    dp_group.create_dataset("camera_positions", data=camera_positions)
    dp_group.create_dataset("images", data=images, compression="gzip", chunks=True)
    dp_group.create_dataset("depths", data=depths, compression="gzip", chunks=True)
    dp_group.create_dataset("seg_ids", data=seg_ids, compression="gzip", chunks=True)

    objects_group = dp_group.create_group("objects")

    solve_feasible_count = 0
    solve_infeasible_count = 0

    for obj_idx, obj in enumerate(config.man_frames):
        obj_start = time.perf_counter()
        print(f"Processing object {obj_idx + 1}/{len(config.man_frames)}: {obj}")

        obj_frame = config.getFrame(obj)
        obj_group = objects_group.create_group(obj)
        obj_group.create_dataset("pose", data=obj_frame.getRelativePose().astype(np.float32))
        obj_group.create_dataset("size", data=obj_frame.getSize()[:3].astype(np.float32))

        primitives_group = obj_group.create_group("primitives")

        for prim_idx, primitive_name in enumerate(Manipulation.primitives):
            prim_start = time.perf_counter()

            _, primitive_dim, primitive_dir = primitive_name.split("_")
            axis = {"x": 0, "y": 1, "z": 2}[primitive_dim]
            direction = {"pos": 1, "neg": -1}[primitive_dir]
            offset_local = np.zeros(3)
            offset_local[axis] = POS_OFFSET * direction
            offset_world = obj_frame.getRotationMatrix() @ offset_local
            target_pos = obj_frame.getRelativePosition() + offset_world
            target_pose = np.concatenate([target_pos, obj_frame.getRelativeQuaternion()])

            man = Manipulation(config, obj, slices=SLICES)
            man.target_pose(target_pose)

            getattr(man, primitive_name)()

            # Time solve() operation
            solve_start = time.perf_counter()
            ret = man.solve()
            solve_duration = time.perf_counter() - solve_start

            if ret.feasible:
                timing_data["solve_feasible"].append(solve_duration)
                solve_feasible_count += 1

                # Time simulate() operation
                sim_start = time.perf_counter()
                man.simulate(view=False)
                sim_duration = time.perf_counter() - sim_start
                timing_data["simulate"].append(sim_duration)
            else:
                timing_data["solve_infeasible"].append(solve_duration)
                solve_infeasible_count += 1

            final_pose = obj_frame.getRelativePose()

            # Save results
            prim_group = primitives_group.create_group(primitive_name)
            prim_group.create_dataset("target_pose", data=target_pose.astype(np.float32))
            prim_group.create_dataset("feasible", data=ret.feasible)
            prim_group.create_dataset("final_pose", data=final_pose.astype(np.float32))

            prim_duration = time.perf_counter() - prim_start
            timing_data["per_primitive"][primitive_name].append(prim_duration)

        timing_data["per_object"][obj] = time.perf_counter() - obj_start

total_duration = time.perf_counter() - total_start
print(f"Total pipeline duration: {total_duration:.2f}s")

print("\nSOLVE() PERFORMANCE:")
if timing_data["solve_feasible"]:
    feasible_times = np.array(timing_data["solve_feasible"])
    print(f"Feasible solves: {len(feasible_times)}")
    print(f"  Total time: {feasible_times.sum():.2f}s")
    print(f"  Average: {feasible_times.mean():.3f}s")
    print(f"  Min/Max: {feasible_times.min():.3f}s / {feasible_times.max():.3f}s")

if timing_data["solve_infeasible"]:
    infeasible_times = np.array(timing_data["solve_infeasible"])
    print(f"Infeasible solves: {len(infeasible_times)}")
    print(f"  Total time: {infeasible_times.sum():.2f}s")
    print(f"  Average: {infeasible_times.mean():.3f}s")
    print(f"  Min/Max: {infeasible_times.min():.3f}s / {infeasible_times.max():.3f}s")

print("\nSIMULATE() PERFORMANCE:")
if timing_data["simulate"]:
    sim_times = np.array(timing_data["simulate"])
    print(f"Simulate calls: {len(sim_times)}")
    print(f"  Total time: {sim_times.sum():.2f}s")
    print(f"  Average: {sim_times.mean():.3f}s")
    print(f"  Min/Max: {sim_times.min():.3f}s / {sim_times.max():.3f}s")

print("\nPER-PRIMITIVE PERFORMANCE:")
for primitive_name, times in timing_data["per_primitive"].items():
    times_arr = np.array(times)
    print(f"{primitive_name}: {times_arr.mean():.3f}s avg ({len(times)} calls)")

print("\nPER-OBJECT PERFORMANCE:")
for obj_name, duration in timing_data["per_object"].items():
    print(f"{obj_name}: {duration:.2f}s")

total_solve_time = sum(timing_data["solve_feasible"]) + sum(timing_data["solve_infeasible"])
total_sim_time = sum(timing_data["simulate"])
print("\nSOLVE vs SIMULATE COMPARISON:")
print(f"Total solve() time: {total_solve_time:.2f}s")
print(f"Total simulate() time: {total_sim_time:.2f}s")
