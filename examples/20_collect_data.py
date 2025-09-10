import h5py
import numpy as np

from robotic.manipulation import Manipulation
from robotic.scenario import PandaScenario

DATASET_PATH = "dataset.h5"
SLICES = 10  # fewer slices = more speed
POS_OFFSET = 0.05  # move 5cm along the push/grasp axis
SEED = 0

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
    for obj in config.man_frames:
        obj_frame = config.getFrame(obj)
        obj_group = objects_group.create_group(obj)
        obj_group.create_dataset("pose", data=obj_frame.getRelativePose().astype(np.float32))
        obj_group.create_dataset("size", data=obj_frame.getSize()[:3].astype(np.float32))

        primitives_group = obj_group.create_group("primitives")
        for primitive_name in Manipulation.primitives:
            man = Manipulation(config, obj, slices=SLICES)
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
            offset_world = obj_frame.getRotationMatrix() @ offset_local
            target_pos = obj_frame.getRelativePosition() + offset_world
            target_pose = np.concatenate([target_pos, obj_frame.getRelativeQuaternion()])
            man.target_pose(target_pose)

            getattr(man, primitive_name)()
            ret = man.solve()
            if ret.feasible:
                man.simulate(view=False)
            final_pose = obj_frame.getRelativePose()

            prim_group = primitives_group.create_group(primitive_name)
            prim_group.create_dataset("target_pose", data=target_pose.astype(np.float32))
            prim_group.create_dataset("feasible", data=ret.feasible)
            prim_group.create_dataset("final_pose", data=final_pose.astype(np.float32))
