from multiprocessing import Pool

import h5py
import numpy as np

from robotic import ST
from robotic._robotic import JT
from robotic.manipulation import Manipulation
from robotic.scenario import PandaScenario


def solve_primitive(args):
    obj, primitive_name, target_pos = args
    man = Manipulation(config, obj, slices=10)
    man_frame = config.getFrame(obj)
    man.reset()
    man.target_pos(target_pos)
    man.target_ori(man_frame.getRelativeQuaternion())
    getattr(man, primitive_name)()
    ret = man.solve()
    return obj, primitive_name, ret.feasible


config = PandaScenario()
box_size = [0.15, 0.06, 0.06, 0.005]
box_z = (config.getFrame("table").getSize()[2] / 2) + (box_size[2] / 2)
config.addFrame("box1", "table").setJoint(JT.rigid).setShape(ST.ssBox, box_size).setRelativePosition([0.4, 0.4, box_z]).setContact(1)
config.addFrame("box2", "table").setJoint(JT.rigid).setShape(ST.ssBox, box_size).setRelativePosition([0.1, 0.1, box_z]).setContact(1)
config.addFrame("box3", "table").setJoint(JT.rigid).setShape(ST.ssBox, box_size).setRelativePosition([0.4, 0.1, box_z]).setContact(1)
config.addFrame("box4", "table").setJoint(JT.rigid).setShape(ST.ssBox, box_size).setRelativePosition([0.1, 0.4, box_z]).setContact(1)

camera_positions = config.camera_positions
images, seg_ids = config.compute_images_and_seg_ids(grayscale=True)

target_pos = np.array([0.5, 0.4, box_z])
with h5py.File("dataset.h5", "w") as f:
    dp_group = f.create_group("datapoint_0002")
    dp_group.create_dataset("images", data=images)
    dp_group.create_dataset("camera_positions", data=camera_positions)
    dp_group.create_dataset("target_pos", data=target_pos)

    man_group = dp_group.create_group("manipulations")

    jobs = []
    for obj in config.man_frames:
        man = Manipulation(config, obj, slices=10)
        man_frame = config.getFrame(obj)
        masks = (seg_ids == man_frame.ID).astype(np.uint8)
        obj_group = man_group.create_group(obj)
        obj_group.create_dataset("masks", data=masks)
        prim_group = obj_group.create_group("primitives")
        for primitive in man.primitives:
            jobs.append((obj, primitive.__name__, target_pos))

    with Pool() as pool:
        results = pool.map(solve_primitive, jobs)

    for obj, primitive_name, feasible in results:
        prim_group = man_group[obj]["primitives"]
        prim_group.create_dataset(f"{primitive_name}_feasible", data=feasible)
