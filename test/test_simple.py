import unittest

from robotic.manipulation import Manipulation
from robotic.scenario import PandaScenario

config = PandaScenario()
box_size = [0.06, 0.15, 0.06, 0.005]
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
        man.grasp_x_pos()
        man.target_pos([0.1, 0.3, box_z])
        self.assertTrue(man.solve().feasible)

    def test_pull(self):
        man = Manipulation(config, "box", slices=10)
        man.pull_x_pos()
        man.target_pos([0.1, 0.3, box_z])
        self.assertTrue(man.solve().feasible)

    def test_all_simle(self):
        config = PandaScenario()
        box_size = [0.06, 0.06, 0.06, 0.005]
        box_z = (config.getFrame("table").getSize()[2] / 2) + (box_size[2] / 2)
        config.add_box("box", box_size, [0.0, 0.2, box_z])

        for primitive_name in Manipulation.primitives:
            man = Manipulation(config, "box", slices=10)
            getattr(man, primitive_name)()
            man.target_pos(config.getFrame("box").getRelativePosition())
            man.target_quat(config.getFrame("box").getRelativeQuaternion())
            self.assertTrue(man.solve().feasible)


if __name__ == "__main__":
    unittest.main()
