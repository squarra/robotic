from robotic import ST
from robotic._robotic import JT
from robotic.manipulation import Manipulation
from robotic.scenario import PandaScenario

config = PandaScenario()
box_size = [0.06, 0.06, 0.06, 0.005]
box_z = (config.getFrame("table").getSize()[2] / 2) + (box_size[2] / 2)
config.addFrame("box", "table").setJoint(JT.rigid).setShape(ST.ssBox, box_size).setRelativePosition([0.0, 0.2, 0.2]).setContact(1)
config.add_markers()

for primitive_name in Manipulation.primitives:
    if "push" in primitive_name:
        continue
    man = Manipulation(config, "box", slices=10)
    getattr(man, primitive_name)()
    man.target_pos(config.getFrame("box").getRelativePosition())
    man.target_quat(config.getFrame("box").getRelativeQuaternion())
    ret = man.solve()
    man.view(pause=True, txt=f"{primitive_name}={ret.feasible}")
