from robotic._robotic import JT, ST
from robotic.manipulation import Manipulation
from robotic.scenario import PandaScenario

config = PandaScenario()
box_size = [0.15, 0.06, 0.06, 0.005]
box_z = (config.getFrame("table").getSize()[2] / 2) + (box_size[2] / 2)
box = (
    config.addFrame("box1", "table")
    .setJoint(JT.rigid)
    .setShape(ST.ssBox, box_size)
    .setRelativePosition([0, 0.4, box_z])
    .setContact(1)
    .setMass(0.1)
    .setAttributes({"friction": 0.5})
)

man = Manipulation(config, box.name, slices=20)
man.grasp_obj_y()
target_pos = box.getRelativePosition() + [0.1, 0.0, 0.0]
man.target_pos_up_z(target_pos)
ret = man.solve()
man.view(pause=True, txt=str(ret.feasible))
if ret.feasible:
    man.simulate(view=False)
    print(man.config.getFrame("box1").getRelativePosition())
