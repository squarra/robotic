import time

import numpy as np

from robotic.manipulation import Manipulation
from robotic.scenario import PandaScenario

SEED = 0
SLICES = 10  # fewer slices = faster but less accurate

config = PandaScenario()
config.add_boxes(density=500, seed=SEED)

offset_directions = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]
primitives = Manipulation.primitives
num_primitives = len(primitives)
num_objects = len(config.man_frames)

t_solve_first = []
t_solve_incre = []
t_simulate = []


for oi, obj in enumerate(config.man_frames):
    rng = np.random.default_rng(SEED)
    direction = offset_directions[rng.integers(len(offset_directions))]
    target_pos = config.sample_target_pos(obj, direction, seed=SEED)
    target_pose = np.concatenate([target_pos, config.getFrame(obj).getRelativeQuaternion()])

    for pi, primitive in enumerate(primitives):
        man = Manipulation(config, obj, slices=SLICES)
        getattr(man, primitive)()
        man.target_pose(target_pose)

        t0 = time.perf_counter()
        feasible = man.solve().feasible
        t1 = time.perf_counter()
        dt = t1 - t0
        t_solve_first.append(dt)

        if feasible:
            man = Manipulation(config, obj, slices=SLICES * 2)
            getattr(man, primitive)()
            man.target_pose(target_pose)

            t0 = time.perf_counter()
            feasible = man.solve().feasible
            t1 = time.perf_counter()
            dt = t1 - t0
            t_solve_incre.append(dt)

        if feasible:
            t0 = time.perf_counter()
            man.simulate(view=False)
            t1 = time.perf_counter()
            t_simulate.append(t1 - t0)

print("=== RESULTS ===")
print(f"solve() first calls: {len(t_solve_first)}, avg={np.mean(t_solve_first):.4f}s, total={np.sum(t_solve_first):.2f}s")
print(f"solve() incre calls: {len(t_solve_incre)}, avg={np.mean(t_solve_incre):.4f}s, total={np.sum(t_solve_incre):.2f}s")
print(f"simulate() calls:    {len(t_simulate)}, avg={np.mean(t_simulate)}s, total={np.sum(t_simulate):.2f}s")
