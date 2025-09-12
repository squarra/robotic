import unittest

import numpy as np

from robotic._robotic import ST
from robotic.manipulation import Manipulation
from robotic.scenario import PandaScenario

config = PandaScenario()
box = config.add_box("box", [0.06, 0.15, 0.06, 0.005], [0, 0.3, 0.08])
half_sqrt2 = np.sqrt(2) / 2


class TestManipulations(unittest.TestCase):
    def test_push_x_pos(self):
        box.setShape(ST.ssBox, [0.06, 0.15, 0.06, 0.005])
        man = Manipulation(config, "box", slices=10)
        man.push_x_pos()
        man.target_pos([0.1, 0.3, 0.08])
        self.assertTrue(man.solve().feasible)
        man = Manipulation(config, "box", slices=10)
        man.push_x_pos()
        man.target_pos([-0.1, 0.3, 0.08])
        self.assertFalse(man.solve().feasible)

    def test_push_x_neg(self):
        box.setShape(ST.ssBox, [0.06, 0.15, 0.06, 0.005])
        man = Manipulation(config, "box", slices=10)
        man.push_x_neg()
        man.target_pos([-0.1, 0.3, 0.08])
        self.assertTrue(man.solve().feasible)
        man = Manipulation(config, "box", slices=10)
        man.push_x_neg()
        man.target_pos([0.1, 0.3, 0.08])
        self.assertFalse(man.solve().feasible)

    def test_push_y_pos(self):
        box.setShape(ST.ssBox, [0.15, 0.06, 0.06, 0.005])
        man = Manipulation(config, "box", slices=10)
        man.push_y_pos()
        man.target_pos([0.0, 0.4, 0.08])
        self.assertTrue(man.solve().feasible)
        man = Manipulation(config, "box", slices=10)
        man.push_y_pos()
        man.target_pos([0.0, 0.2, 0.08])
        self.assertFalse(man.solve().feasible)

    def test_push_y_neg(self):
        box.setShape(ST.ssBox, [0.15, 0.06, 0.06, 0.005])
        man = Manipulation(config, "box", slices=10)
        man.push_y_neg()
        man.target_pos([0.0, 0.2, 0.08])
        self.assertTrue(man.solve().feasible)
        man = Manipulation(config, "box", slices=10)
        man.push_y_neg()
        man.target_pos([0.0, 0.4, 0.08])
        self.assertFalse(man.solve().feasible)

    def test_push_z_pos(self):
        box.setShape(ST.ssBox, [0.15, 0.06, 0.06, 0.005])
        box.setRelativeQuaternion([half_sqrt2, half_sqrt2, 0, 0])
        man = Manipulation(config, "box", slices=10)
        man.push_z_pos()
        man.target_pos([0.0, 0.25, 0.08])
        self.assertTrue(man.solve().feasible)
        man = Manipulation(config, "box", slices=10)
        man.push_z_pos()
        man.target_pos([0.0, 0.35, 0.08])
        self.assertFalse(man.solve().feasible)

    def test_push_z_neg(self):
        box.setShape(ST.ssBox, [0.15, 0.06, 0.06, 0.005])
        box.setRelativeQuaternion([half_sqrt2, half_sqrt2, 0, 0])
        man = Manipulation(config, "box", slices=10)
        man.push_z_neg()
        man.target_pos([0.0, 0.35, 0.08])
        self.assertTrue(man.solve().feasible)
        man = Manipulation(config, "box", slices=10)
        man.push_z_neg()
        man.target_pos([0.0, 0.25, 0.08])
        self.assertFalse(man.solve().feasible)


if __name__ == "__main__":
    unittest.main()
