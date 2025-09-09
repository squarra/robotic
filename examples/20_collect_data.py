import h5py
import numpy as np

from robotic.manipulation import Manipulation
from robotic.scenario import PandaScenario

DATASET_PATH = "dataset.h5"
SLICES = 10  # fewer slices = more speed

config = PandaScenario()
config.add_boxes_to_scene(seed=42)

camera_positions = config.camera_positions
images, depths, seg_ids = config.compute_images_depths_and_seg_ids()

with h5py.File(DATASET_PATH, "w") as f:
    dp_group = f.create_group("datapoint_0001")
    dp_group.create_dataset("camera_positions", data=camera_positions)
    dp_group.create_dataset("images", data=images, compression="gzip", chunks=True)
    dp_group.create_dataset("depths", data=depths, compression="gzip", chunks=True)
    dp_group.create_dataset("seg_ids", data=seg_ids, compression="gzip", chunks=True)

    objects_group = dp_group.create_group("objects")
    for obj in config.man_frames:
        man_frame = config.getFrame(obj)
        obj_group = objects_group.create_group(obj)
        obj_group.create_dataset("pose", data=man_frame.getRelativePose().astype(np.float32))
        obj_group.create_dataset("size", data=man_frame.getSize()[:3].astype(np.float32))

        primitives_group = obj_group.create_group("primitives")
        for prim in Manipulation.primitives:
            man = Manipulation(config, obj, slices=SLICES)
            if "push" in prim:
                _, primitive_dim, primitive_dir = prim.split("_")
                axis = {"x": 0, "y": 1, "z": 2}[primitive_dim]
                direction = {"pos": 1, "neg": -1}[primitive_dir]
                offset = np.zeros(3)
                offset[axis] = 0.05 * direction  # push 5cm in the specified direction
                offset = man_frame.getRotationMatrix() @ offset
                target_pos = man_frame.getRelativePosition() + offset
                man.target_pos(target_pos)

            getattr(man, prim)()
            ret = man.solve()
            man.view(pause=True, txt=f"{prim}={ret.feasible}")
            if ret.feasible:
                man.simulate(view=False)

            prim_group = primitives_group.create_group(prim)
            prim_group.create_dataset("feasible", data=ret.feasible)
            prim_group.create_dataset("pose", data=man.config.getFrame(obj).getRelativePose())
