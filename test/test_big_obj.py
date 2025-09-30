import unittest

from robotic.manipulation import Manipulation
from robotic.scenario import PandaScenario

config = PandaScenario()
box_size = [0.12, 0.12, 0.12, 0.005]
box_z = (config.table.getSize()[2] / 2) + (box_size[2] / 2)
box = config.add_box("box", box_size, [0, 0.3, box_z])


class TestManipulations(unittest.TestCase):
    def test_push(self):
        man = Manipulation(config, "box", slices=10)
        man.push_x_pos()
        man.target_pos([0.1, 0.3, box_z])
        self.assertTrue(man.solve().feasible)

    def test_grasp(self):
        man = Manipulation(config, "box", slices=10)
        man.lift_x_pos()
        man.target_pos([0.1, 0.3, box_z])
        self.assertFalse(man.solve().feasible)

    def test_pull(self):
        man = Manipulation(config, "box", slices=10)
        man.pull_x_pos()
        man.target_pos([0.1, 0.3, box_z])
        self.assertFalse(man.solve().feasible)


if __name__ == "__main__":
    unittest.main()
