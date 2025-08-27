from robotic._robotic import FS, JT, OT, ST
from robotic.manipulation import Manipulation
from robotic.scenario import PandaScenario

config = PandaScenario()
box_size = [0.15, 0.06, 0.06, 0.005]
box_z = (config.getFrame("table").getSize()[2] / 2) + (box_size[2] / 2)
box = config.addFrame("box1", "table").setJoint(JT.rigid).setShape(ST.ssBox, box_size).setRelativePosition([0.4, 0.4, box_z]).setContact(1)

for primitive in Manipulation.primitives:
    man = Manipulation(config, box.name, slices=20)
    getattr(man, primitive)()
    # man.target_pos(box.getRelativePosition() + [0.1, 0.1, 0.0])
    ret = man.solve()
    man.view(pause=True, txt=f"{primitive}={ret.feasible}")
    man.simulate()
