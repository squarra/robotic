import time

import numpy as np

from robotic import FS, JT, KOMO, OT, ControlMode, NLP_Solver, Simulation, SimulationEngine
from robotic._robotic import ST
from robotic.helpers import DEBUG
from robotic.scenario import PandaScenario

gripper = "gripper"
table = "table"
palm = "palm"


class Manipulation:
    primitives = ["push_x_pos", "push_x_neg", "push_y_pos", "push_y_neg", "grasp_x_pos", "grasp_y_pos", "pull_x_pos", "pull_y_pos"]

    def __init__(self, scenario: PandaScenario, obj: str, phases=3.0, slices=1):
        self.scenario = scenario
        self.komo = KOMO(scenario, phases, slices, 2, True)
        self.komo.addControlObjective([], 0, 1e-2)
        self.komo.addControlObjective([], 1, 1e-1)
        self.komo.addObjective([], FS.jointLimits, [], OT.ineq, [1e0])
        self.collision_objective = self.komo.addObjective([], FS.accumulatedCollisions, [], OT.eq, [1e0])
        self.komo.addQuaternionNorms()

        self.obj = obj
        self.phases = phases
        self.slices = slices
        self.config = self.komo.getConfig()
        self.action = None

        for frame in self.scenario.man_frames:
            if frame == self.obj or self.scenario.getFrame(frame).getShapeType() == ST.marker:
                continue
            self.komo.addObjective([], FS.distance, [frame, palm], OT.ineq, scale=[1e1], target=[-0.01])
            self.komo.addObjective([], FS.distance, [frame, "finger1"], OT.ineq, scale=[1e1], target=[-0.01])
            self.komo.addObjective([], FS.distance, [frame, "finger2"], OT.ineq, scale=[1e1], target=[-0.01])
            self.komo.addObjective([], FS.distance, [frame, self.obj], OT.ineq, scale=[1e1], target=[-0.05])

    def _push_obj(self, dim: int, dir: int):
        self.action = "push"
        joints = [JT.transX, JT.transY, JT.transZ]

        self.komo.addFrameDof("obj_trans", table, joints[dim], False, self.obj)
        self.komo.addRigidSwitch(1.0, ["obj_trans", self.obj])

        y_axis = np.eye(3)[dim]
        x_axis = np.eye(3)[1 - dim]  # assuming z up
        # xz_plane = np.delete(np.eye(3), dim, axis=0)
        relative_gripper_contact_pos = 0.5 * self.get_bbox(self.obj)[dim] + 0.025

        # remove collision objective for the time of pushing
        self.komo.removeObjective(self.collision_objective)
        self.komo.addObjective([0.0, 0.8], FS.accumulatedCollisions, [], OT.eq, [1e0])
        self.komo.addObjective([2.2, 3.0], FS.accumulatedCollisions, [], OT.eq, [1e0])

        # gripper position
        pre_target = -dir * (relative_gripper_contact_pos + 0.03)
        self.komo.addObjective([0.8], FS.positionRel, [gripper, self.obj], OT.eq, scale=y_axis * 1e1, target=[pre_target])
        target = -dir * relative_gripper_contact_pos
        self.komo.addObjective([1.0, 2.0], FS.positionRel, [gripper, self.obj], OT.eq, scale=y_axis * 1e1, target=[target])
        self.komo.addObjective([1.0, 2.0], FS.positionRel, [gripper, self.obj], OT.eq, scale=x_axis * 1e1)
        self.komo.addObjective([0.8, 2.0], FS.positionRel, [gripper, table], OT.eq, scale=[0.0, 0.0, 1e1], target=[0.07])
        # gripper orientation
        self.komo.addObjective([0.8, 2.0], FS.vectorZ, [gripper], OT.eq, [1e-1], [0, 0, 1])
        self.komo.addObjective([0.8, 2.0], FS.scalarProductYX, [gripper, self.obj], OT.eq, scale=[1e0], target=[y_axis[0]])
        self.komo.addObjective([0.8, 2.0], FS.scalarProductYY, [gripper, self.obj], OT.eq, scale=[1e0], target=[y_axis[1]])
        self.komo.addObjective([0.8, 2.0], FS.scalarProductYZ, [gripper, self.obj], OT.eq, scale=[1e0], target=[y_axis[2]])
        # allow movement in only one direction
        y_axis_world = self.config.getFrame(self.obj).getRotationMatrix() @ y_axis  # for some reason this needs to be absolute
        self.komo.addObjective([1.0, 2.0], FS.position, [self.obj], OT.ineq, scale=-dir * y_axis_world * 1e0, target=[0], order=1)
        # keep gripper orientation and restrict movement for stability
        self.komo.addObjective([2.0, 3.0], FS.quaternionRel, [gripper, table], OT.eq, [1e0], target=[], order=1)
        self.komo.addObjective([2.0, 3.0], FS.positionRel, [gripper, self.obj], OT.eq, scale=x_axis * 1e0, target=[0])
        # retract 5cm
        post_target = -dir * (relative_gripper_contact_pos + 0.05)
        self.komo.addObjective([2.5, 3.0], FS.positionRel, [gripper, self.obj], OT.eq, scale=y_axis * 1e0, target=[post_target])

    def push_x_pos(self):
        self._push_obj(0, 1)

    def push_x_neg(self):
        self._push_obj(0, -1)

    def push_y_pos(self):
        self._push_obj(1, 1)

    def push_y_neg(self):
        self._push_obj(1, -1)

    def _grasp_obj(self, dim: int, dir: int):
        self.action = "grasp"
        products = [FS.scalarProductXX, FS.scalarProductXY, FS.scalarProductXZ]

        self.komo.addFrameDof("obj_free", gripper, JT.free, True, self.obj)
        self.komo.addRigidSwitch(1.0, ["obj_free", self.obj])

        x_axis = np.eye(3)[dim]
        yz_plane = np.delete(np.eye(3), dim, axis=0)

        # gripper position
        target = 0.5 * self.get_bbox(self.obj) - 0.02
        self.komo.addObjective([0.6, 1.0], FS.positionRel, [gripper, self.obj], OT.eq, scale=x_axis * 1e1)
        self.komo.addObjective([0.9, 1.0], FS.positionRel, [gripper, self.obj], OT.ineq, scale=yz_plane * 1e0, target=[target])
        self.komo.addObjective([0.9, 1.0], FS.positionRel, [gripper, self.obj], OT.ineq, scale=-yz_plane * 1e0, target=[-target])
        # encourage well centered grasps
        self.komo.addObjective([1.0], FS.positionRel, [gripper, self.obj], OT.eq, scale=[1e-2])
        # gripper orientation
        self.komo.addObjective([0.6, 1.0], products[dim], [gripper, self.obj], OT.eq, scale=[1e0], target=[dir])
        # retract gracefully
        self.komo.addObjective([2.0, 3.0], products[dim], [gripper, self.obj], OT.eq, scale=[1e1], target=[1])
        self.komo.addObjective([2.0, 3.0], FS.positionRel, [gripper, self.obj], OT.eq, scale=x_axis, target=[0])

        self.komo.addFrameDof("obj_free_after", table, JT.free, True, self.obj)
        self.komo.addRigidSwitch(2.0, ["obj_free_after", self.obj])

    def grasp_x_pos(self):
        self._grasp_obj(0, 1)

    def grasp_y_pos(self):
        self._grasp_obj(1, 1)

    def _pull_obj(self, dim: int, dir: int):
        self.action = "pull"
        products = [FS.scalarProductXX, FS.scalarProductXY, FS.scalarProductXZ]

        self.komo.addFrameDof("gripper_obj_free", gripper, JT.free, True, self.obj)
        self.komo.addRigidSwitch(1.0, ["gripper_obj_free", self.obj])

        x_axis = np.eye(3)[dim]
        yz_plane = np.delete(np.eye(3), dim, axis=0)

        # this is stolen from _grasp_obj()
        target = 0.5 * self.get_bbox(self.obj) - 0.02
        self.komo.addObjective([0.5, 1.0], FS.positionRel, [gripper, self.obj], OT.eq, scale=x_axis * 1e1)
        self.komo.addObjective([0.9, 1.0], FS.positionRel, [gripper, self.obj], OT.ineq, scale=yz_plane * 1e0, target=[target])
        self.komo.addObjective([0.9, 1.0], FS.positionRel, [gripper, self.obj], OT.ineq, scale=-yz_plane * 1e0, target=[-target])
        self.komo.addObjective([0.5, 1.0], products[dim], [gripper, self.obj], OT.eq, scale=[1e0], target=[dir])
        # make sure object is not lifted
        self.komo.addObjective([1.0, 2.0], FS.position, [self.obj], OT.eq, scale=[0, 0, 1e1], target=[0], order=1)
        # object stays upright
        self.komo.addObjective([1.0, 2.0], FS.scalarProductZZ, [table, self.obj], OT.eq, scale=[1e1], target=[1])
        # retract gracefully
        self.komo.addObjective([2.0, 3.0], products[dim], [gripper, self.obj], OT.eq, scale=[1e1], target=[1])
        self.komo.addObjective([2.0, 3.0], FS.positionRel, [gripper, self.obj], OT.eq, scale=x_axis, target=[0])

    def pull_x_pos(self):
        self._pull_obj(0, 1)

    def pull_y_pos(self):
        self._pull_obj(1, 1)

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
        self.komo.addObjective([2.0, 3.0], FS.positionRel, [self.obj, table], OT.eq, [1e1], target=pos)

    def target_quat(self, ori: np.typing.ArrayLike):
        self.komo.addObjective([2.0, 3.0], FS.quaternionRel, [self.obj, table], OT.eq, scale=[1e1], target=ori)

    def target_pose(self, pose: np.typing.ArrayLike):
        self.komo.addObjective([2.0, 3.0], FS.poseRel, [self.obj, table], OT.eq, scale=[1e1], target=pose)

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
        splits = np.split(self.komo.getPath(), int(self.phases))

        sim.setSplineRef(path=splits[0], times=[times])
        for _ in range(sim_steps):
            sim.step([], tau, ControlMode.spline)
            if view:
                time.sleep(tau)
                self.config.view()

        if self.action == "push":
            sim.moveGripper(gripper, 0.02)
        else:
            sim.moveGripper(gripper, 0.0)
        while not sim.gripperIsDone(gripper):
            sim.step([], tau, ControlMode.spline)
            if view:
                time.sleep(tau)
                self.config.view()

        sim.setSplineRef(path=splits[1], times=[times])
        for _ in range(sim_steps):
            sim.step([], tau, ControlMode.spline)
            if view:
                time.sleep(tau)
                self.config.view()

        sim.moveGripper(gripper, 0.08)
        while not sim.gripperIsDone(gripper):
            sim.step([], tau, ControlMode.spline)
            if view:
                time.sleep(tau)
                self.config.view()

        sim.setSplineRef(path=splits[2], times=[times])
        for _ in range(sim_steps):
            sim.step([], tau, ControlMode.spline)
            if view:
                time.sleep(tau)
                self.config.view()
