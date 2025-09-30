import unittest

import numpy as np

from robotic.manipulation import Manipulation
from robotic.scenario import PandaScenario

from robotic.helpers import VIEW


def create_config_with_box(box_size: np.typing.ArrayLike):
    config = PandaScenario()
    box_z = (config.table.getSize()[2] / 2) + (box_size[2] / 2)
    config.add_box("box", [*box_size, 0.005], [0, 0.3, box_z])
    return config, box_z


class TestManipulations(unittest.TestCase):
    def test_rotations(self):
        def test_quat(quat: np.typing.ArrayLike):
            man = Manipulation(config, "box", slices=10)
            man.lift_x()
            man.target_pose([0.2, 0.3, box_z, *quat])
            feasible = man.solve().feasible
            if VIEW:
                man.view(pause=True, txt=f"quat={quat}, feasible={feasible}")
            self.assertTrue(feasible)

        config, box_z = create_config_with_box([0.06, 0.06, 0.06])
        test_quat([1.0, 0.0, 0.0, 0.0])
        test_quat([0.0, 0.0, 0.0, 1.0])
        test_quat([0.707, 0.0, 0.0, 0.707])
        # test_quat([-0.707, 0.0, 0.0, 0.707])

    def test_big_obj(self):
        return
        config, box_z = create_config_with_box([0.12, 0.12, 0.12])
        man = Manipulation(config, "box", slices=10)
        man.lift_x()
        man.target_pose([0.2, 0.3, box_z, 1.0, 0.0, 0.0, 0.0])
        feasible = man.solve().feasible
        if VIEW:
            man.view(pause=True, txt=str(feasible))
        self.assertFalse(feasible)


if __name__ == "__main__":
    unittest.main()
