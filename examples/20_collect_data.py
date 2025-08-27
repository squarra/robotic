import h5py
import numpy as np

from robotic.manipulation import Manipulation
from robotic.scenario import PandaScenario

config = PandaScenario()
config.add_boxes_to_scene(seed=42)

camera_positions = config.camera_positions
images, seg_ids = config.compute_images_and_seg_ids(grayscale=True)

with h5py.File("dataset.h5", "w") as f:
    dp_group = f.create_group("datapoint_0001")

    dp_group.create_dataset("images", data=images)
    dp_group.create_dataset("camera_positions", data=camera_positions)

    man_group = dp_group.create_group("manipulations")
    for obj in config.man_frames:
        man_frame = config.getFrame(obj)
        obj_group = man_group.create_group(obj)
        obj_group.create_dataset("quat", data=man_frame.getRelativeQuaternion().astype(np.float32))
        obj_group.create_dataset("masks", data=(seg_ids == man_frame.ID).astype(np.uint8))
        prim_group = obj_group.create_group("primitives")
        for primitive in Manipulation.primitives:
            man = Manipulation(config, obj, slices=10)
            getattr(man, primitive)()
            ret = man.solve()
            prim_group.create_dataset(f"{primitive}", data=ret.feasible)
