import time

import numpy as np

from robotic import FS, JT, KOMO, OT, ControlMode, NLP_Solver, Simulation, SimulationEngine
from robotic.helpers import DEBUG
from robotic.scenario import PandaScenario

gripper = "gripper"
table = "table"


class Manipulation:
    primitives = [
        "push_x_pos",
        "push_y_pos",
        "push_z_pos",
        "push_x_neg",
        "push_y_neg",
        "push_z_neg",
        "grasp_x_pos",
        "grasp_x_neg",
        "grasp_y_pos",
        "grasp_y_neg",
        "grasp_z_pos",
        "grasp_z_neg",
    ]

    def __init__(self, scenario: PandaScenario, obj: str, slices=1):
        self.scenario = scenario
        self.komo = KOMO(scenario, 2.0, slices, 1, True)
        self.komo.addControlObjective([], 0, 1e-2)
        self.komo.addControlObjective([], 1, 1e-1)
        self.komo.addObjective([], FS.jointLimits, [], OT.ineq, [1e0])
        self.komo.addObjective([], FS.accumulatedCollisions, [], OT.eq, [1e0])
        self.komo.addQuaternionNorms()

        self.obj = obj
        self.slices = slices
        self.config = self.komo.getConfig()
        self.action = None

        for obj in self.scenario.man_frames:
            self.komo.addObjective([], FS.distance, ["palm", obj], OT.ineq, scale=[1e1], target=[-0.001])

    def _push_obj(self, dim: int, dir: int):
        self.action = "push"
        if dim == 0:
            joint = JT.transX
        elif dim == 1:
            joint = JT.transY
        elif dim == 2:
            joint = JT.transZ
        else:
            raise ValueError
        self.komo.addFrameDof("obj_trans", table, joint, False, self.obj)
        self.komo.addRigidSwitch(1.0, ["obj_trans", self.obj])

        y_axis = np.eye(3)[dim]
        xz_plane = np.delete(np.eye(3), dim, axis=0)
        relative_gripper_contact_pos = 0.5 * self.get_bbox(self.obj)[dim] + 0.025

        # gripper position
        pre_target = -dir * (relative_gripper_contact_pos + 0.01)
        self.komo.addObjective([0.8], FS.positionRel, [gripper, self.obj], OT.eq, scale=y_axis * 1e1, target=[pre_target])
        target = -dir * relative_gripper_contact_pos
        self.komo.addObjective([1.0, 2.0], FS.positionRel, [gripper, self.obj], OT.eq, scale=y_axis * 1e1, target=[target])
        self.komo.addObjective([1.0, 2.0], FS.positionRel, [gripper, self.obj], OT.eq, scale=xz_plane * 1e1)
        # gripper orientation
        self.komo.addObjective([1.0, 2.0], FS.vectorZ, [gripper], OT.eq, [1e0], [0, 0, 1])
        self.komo.addObjective([1.0, 2.0], FS.scalarProductYX, [gripper, self.obj], OT.eq, scale=[1e0], target=[y_axis[0]])
        self.komo.addObjective([1.0, 2.0], FS.scalarProductYY, [gripper, self.obj], OT.eq, scale=[1e0], target=[y_axis[1]])
        self.komo.addObjective([1.0, 2.0], FS.scalarProductYZ, [gripper, self.obj], OT.eq, scale=[1e0], target=[y_axis[2]])
        # allow movement in only one direction
        self.komo.addObjective([1.0, 2.0], FS.position, [self.obj], OT.ineq, scale=-dir * y_axis * 1e0, target=[0], order=1)

    def push_x_pos(self):
        self._push_obj(0, 1)

    def push_x_neg(self):
        self._push_obj(0, -1)

    def push_y_pos(self):
        self._push_obj(1, 1)

    def push_y_neg(self):
        self._push_obj(1, -1)

    def push_z_pos(self):
        self._push_obj(2, 1)

    def push_z_neg(self):
        self._push_obj(2, -1)

    def _grasp_obj(self, dim: int, dir: int):
        self.action = "grasp"
        products = [FS.scalarProductXX, FS.scalarProductXY, FS.scalarProductXZ]

        self.komo.addFrameDof("obj_trans", gripper, JT.free, True, self.obj)
        self.komo.addRigidSwitch(1.0, ["obj_trans", self.obj])

        x_axis = np.eye(3)[dim]
        yz_plane = np.delete(np.eye(3), dim, axis=0)

        # gripper position
        target = 0.5 * self.get_bbox(self.obj) - 0.02
        self.komo.addObjective([0.7, 1.0], FS.positionRel, [gripper, self.obj], OT.eq, scale=x_axis * 1e1)
        self.komo.addObjective([1.0], FS.positionRel, [gripper, self.obj], OT.ineq, scale=yz_plane * 1e0, target=[target])
        self.komo.addObjective([1.0], FS.positionRel, [gripper, self.obj], OT.ineq, scale=yz_plane * (-1e0), target=[-target])
        self.komo.addObjective([1.0], FS.positionRel, [gripper, self.obj], OT.eq, scale=[1e-1])
        # gripper orientation
        self.komo.addObjective([0.7, 1.0], products[dim], [gripper, self.obj], OT.eq, scale=[1e0], target=[dir])

    def grasp_x_pos(self):
        self._grasp_obj(0, 1)

    def grasp_x_neg(self):
        self._grasp_obj(0, -1)

    def grasp_y_pos(self):
        self._grasp_obj(1, 1)

    def grasp_y_neg(self):
        self._grasp_obj(1, -1)

    def grasp_z_pos(self):
        self._grasp_obj(2, 1)

    def grasp_z_neg(self):
        self._grasp_obj(2, -1)

    def target_pos_up_axis(self, dim: int, dir: int, align: FS, pos: np.typing.ArrayLike):
        """Places the object on top of the table at a certain position with the specified up axis"""
        self.komo.addObjective([2.0], align, [self.obj], OT.eq, [1e1], target=[0, 0, dir])
        pos[2] = (self.get_bbox(self.obj)[dim] / 2.0) + (self.scenario.table.getSize()[2] / 2.0)
        self.target_pos(pos)

    def target_pos_up_x(self, pos: np.typing.ArrayLike):
        self.target_pos_up_axis(0, 1, FS.vectorX, pos)

    def target_pos_up_x_neg(self, pos: np.typing.ArrayLike):
        self.target_pos_up_axis(0, -1, FS.vectorX, pos)

    def target_pos_up_y(self, pos: np.typing.ArrayLike):
        self.target_pos_up_axis(1, 1, FS.vectorY, pos)

    def target_pos_up_y_neg(self, pos: np.typing.ArrayLike):
        self.target_pos_up_axis(1, -1, FS.vectorY, pos)

    def target_pos_up_z(self, pos: np.typing.ArrayLike):
        self.target_pos_up_axis(2, 1, FS.vectorZ, pos)

    def target_pos_up_z_neg(self, pos: np.typing.ArrayLike):
        self.target_pos_up_axis(2, -1, FS.vectorZ, pos)

    def target_pos(self, pos: np.typing.ArrayLike):
        self.komo.addObjective([2.0], FS.positionRel, [self.obj, table], OT.eq, [1e1], target=pos)

    def target_quat(self, ori: np.typing.ArrayLike):
        self.komo.addObjective([2.0], FS.quaternionRel, [self.obj, table], OT.eq, scale=[1e1], target=ori)

    def target_pose(self, pose: np.typing.ArrayLike):
        self.komo.addObjective([2.0], FS.poseRel, [self.obj, table], OT.eq, scale=[1e1], target=pose)

    def solve(self):
        sol = NLP_Solver(self.komo.nlp(), DEBUG.value)
        sol.setOptions(damping=1e-1, stopTolerance=1e-3, lambdaMax=100.0, stopInners=20, stopEvals=200)
        return sol.solve(verbose=DEBUG.value)

    def view(self, **kwargs):
        self.komo.view(**kwargs)

    def get_grasp_pose(self) -> np.ndarray:
        return self.komo.getFrame(gripper, 1).getPose()

    def get_grasp_transform(self) -> np.ndarray:
        return self.komo.getFrame(gripper, 1).getTransform()

    def get_bbox(self, frame_name):
        vertices = self.config.getFrame(frame_name).getMesh()[0]

        if vertices is None or not isinstance(vertices, np.ndarray) or vertices.ndim == 0:
            print(f"Warning: No vertices found for {frame_name}")
            return np.array([0.1, 0.1, 0.1])

        return np.max(vertices, axis=0) - np.min(vertices, axis=0)

    def simulate(self, view=True, times=1.0, tau=5e-3):
        sim = Simulation(self.config, SimulationEngine.physx, verbose=DEBUG.value)
        sim_steps = int(times // tau)
        splits = np.split(self.komo.getPath(), [self.slices])

        sim.setSplineRef(path=splits[0], times=[times])
        for _ in range(sim_steps):
            sim.step([], tau, ControlMode.spline)
            if view:
                time.sleep(tau)
                self.config.view()

        if self.action == "grasp":
            sim.moveGripper(gripper, 0.0)
        else:
            sim.moveGripper(gripper, 0.02)
        while not sim.gripperIsDone(gripper):
            sim.step([], tau, ControlMode.spline)
            if view:
                time.sleep(tau)
                self.config.view()

        sim.setSplineRef(path=splits[1], times=[1])
        for _ in range(sim_steps):
            sim.step([], tau, ControlMode.spline)
            if view:
                time.sleep(tau)
                self.config.view()
