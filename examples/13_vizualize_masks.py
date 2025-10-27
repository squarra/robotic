import matplotlib.pyplot as plt
import numpy as np

from robotic import ST
from robotic._robotic import JT
from robotic.scenario import PandaScenario

config = PandaScenario()
box_size = [0.15, 0.06, 0.06, 0.005]
box_z = (config.getFrame("table").getSize()[2] / 2) + (box_size[2] / 2)
config.addFrame("box1", "table").setJoint(JT.rigid).setShape(ST.ssBox, box_size).setRelativePosition([0.4, 0.4, box_z]).setContact(1)
config.addFrame("box2", "table").setJoint(JT.rigid).setShape(ST.ssBox, box_size).setRelativePosition([0.1, 0.1, box_z]).setContact(1)
config.addFrame("box3", "table").setJoint(JT.rigid).setShape(ST.ssBox, box_size).setRelativePosition([0.4, 0.1, box_z]).setContact(1)
config.addFrame("box4", "table").setJoint(JT.rigid).setShape(ST.ssBox, box_size).setRelativePosition([0.1, 0.4, box_z]).setContact(1)

images, _, seg_ids = config.compute_images_depths_and_seg_ids()
n_images = len(images)

for obj in config.man_frames:
    man_frame = config.getFrame(obj)
    masks = (seg_ids == man_frame.ID).astype(np.uint8)
    fig, axes = plt.subplots(n_images, 2, figsize=(12, 3 * n_images))
    for i in range(n_images):
        axes[i, 0].imshow(images[i], cmap="gray")
        axes[i, 1].imshow(masks[i], cmap="gray")
    plt.tight_layout()
    plt.show()
