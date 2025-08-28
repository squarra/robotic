from robotic.manipulation import Manipulation
from robotic.scenario import PandaScenario

config = PandaScenario()
config.add_boxes_to_scene(box_size_range=(0.02, 0.08), seed=42)

for obj in config.man_frames:
    for primitive in Manipulation.primitives:
        man = Manipulation(config, obj, slices=20)
        getattr(man, primitive)()
        man.target_pos(config.getFrame(obj).getRelativePosition())
        man.target_quat(config.getFrame(obj).getRelativeQuaternion())
        ret = man.solve()
        man.view(pause=True, txt=f"{primitive}={ret.feasible}")
        if ret.feasible:
            man.simulate(times=1.0)
