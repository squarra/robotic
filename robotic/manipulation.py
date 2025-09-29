import time

import numpy as np

from robotic import FS, JT, KOMO, OT, ControlMode, NLP_Solver, Simulation, SimulationEngine
from robotic._robotic import ST
from robotic.helpers import DEBUG
from robotic.scenario import PandaScenario

gripper = "gripper"
table = "table"
palm = "palm"


class Manipulation(KOMO):
    primitives = ["push_x_pos", "push_x_neg", "push_y_pos", "push_y_neg", "grasp_x_pos", "grasp_y_pos", "pull_x_pos", "pull_y_pos"]

    def __init__(self, scenario: PandaScenario, obj: str, slices=10):
        self.scenario = scenario
        self.phases = 3.0
        self.slices = slices
        super().__init__(scenario, self.phases, self.slices, 1, True)
        self.config = self.getConfig()

        self.addControlObjective([], 0, 1e-2)
        self.addControlObjective([], 1, 1e-1)
        self.addObjective([], FS.jointLimits, [], OT.ineq, [1e0])
        self.collision_objective = self.addObjective([], FS.accumulatedCollisions, [], OT.eq, [1e0])
        self.addQuaternionNorms()

        self.obj = obj
        self.action = None

        for frame in self.scenario.man_frames:
            if frame == self.obj or self.scenario.getFrame(frame).getShapeType() == ST.marker:
                continue
            self.addObjective([], FS.distance, [frame, palm], OT.ineq, scale=[1e1], target=[-0.01])
            self.addObjective([], FS.distance, [frame, "finger1"], OT.ineq, scale=[1e1], target=[-0.01])
            self.addObjective([], FS.distance, [frame, "finger2"], OT.ineq, scale=[1e1], target=[-0.01])
            # avoid collisions during manipulation
            self.addObjective([1.2, 1.8], FS.distance, [frame, self.obj], OT.ineq, scale=[1e1], target=[-0.01])

    def set_slices(self, slices=10):
        self.slices = slices
        self.setTiming(self.phases, self.slices, 1.0, 1)

    def _push_obj(self, dim: int, dir: int):
        self.action = "push"
        joints = [JT.transX, JT.transY]
        products = [FS.scalarProductYY, FS.scalarProductYX]

        x_axis = np.eye(3)[dim]  # obj push axis
        y_axis = np.eye(3)[1 - dim]  # lateral to obj push axis
        z_axis = np.array([0.0, 0.0, 1.0])  # up axis

        obj_bbox = self.get_bbox(self.obj)
        target_x = (0.5 * obj_bbox[dim]) + 0.025

        # remove collision objective for the time of pushing
        self.removeObjective(self.collision_objective)
        self.addObjective([0.0, 0.7], FS.accumulatedCollisions, [], OT.eq, [1e0])
        self.addObjective([2.3, 3.0], FS.accumulatedCollisions, [], OT.eq, [1e0])

        # approach
        pre_target_x = -dir * (target_x + 0.03)
        pre_target_z = (0.5 * obj_bbox[2]) + 0.01
        self.addObjective([0.7], FS.positionRel, [gripper, self.obj], OT.eq, scale=x_axis * 1e0, target=[pre_target_x])
        self.addObjective([0.7], FS.positionRel, [gripper, self.obj], OT.eq, scale=z_axis * 1e0, target=[pre_target_z])
        self.addObjective([0.7, 1.0], FS.positionRel, [gripper, self.obj], OT.eq, scale=y_axis * 1e0, target=[0])
        self.addObjective([0.7, 1.0], products[dim], [gripper, self.obj], OT.eq, scale=[1e0], target=[0])
        self.addObjective([0.7, 1.0], FS.vectorZ, [gripper], OT.eq, scale=[1e-1], target=[0.0, 0.0, 1.0])
        # contact
        self.addObjective([1.0], FS.positionRel, [gripper, table], OT.eq, scale=z_axis, target=[0.07])
        self.addFrameDof("obj_trans", table, joints[dim], False, self.obj)
        self.addRigidSwitch(1.0, ["obj_trans", self.obj])
        # push
        self.addObjective([1.0, 2.0], FS.positionRel, [gripper, self.obj], OT.eq, scale=x_axis * 1e0, target=[-dir * target_x])
        self.addObjective([1.0, 2.0], FS.positionRel, [gripper, self.obj], OT.eq, scale=y_axis * 1e0, target=[0], order=1)
        self.addObjective([1.0, 2.0], FS.positionRel, [gripper, table], OT.eq, scale=z_axis * 1e0, target=[0], order=1)
        self.addObjective([1.0, 2.0], FS.quaternionRel, [gripper, self.obj], OT.eq, scale=[1e0], target=[], order=1)
        # allow movement in only one direction
        x_axis_world = self.config.getFrame(self.obj).getRotationMatrix() @ x_axis  # for some reason this needs to be absolute
        self.addObjective([1.0, 2.0], FS.position, [self.obj], OT.ineq, scale=-dir * x_axis_world * 1e0, target=[0], order=1)
        # release
        post_target_x = -dir * (target_x + 0.05)
        self.addObjective([2.3, 3.0], FS.positionRel, [gripper, self.obj], OT.eq, scale=x_axis * 1e0, target=[post_target_x])
        self.addObjective([2.0, 3.0], FS.positionRel, [gripper, self.obj], OT.eq, scale=y_axis * 1e0, target=[0], order=1)
        self.addObjective([2.0, 3.0], FS.positionRel, [gripper, table], OT.eq, scale=z_axis * 1e0, target=[0], order=1)
        self.addObjective([2.0, 3.0], FS.quaternionRel, [gripper, self.obj], OT.eq, scale=[1e0], target=[], order=1)

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

        x_axis = np.eye(3)[dim]
        y_axis = np.eye(3)[1 - dim]
        z_axis = np.array([0.0, 0.0, 1.0])

        # approach
        target = 0.5 * self.get_bbox(self.obj) - 0.025
        pre_target_z = target[2] + 0.1
        self.addObjective([0.7], FS.positionRel, [gripper, self.obj], OT.eq, scale=z_axis * 1e0, target=[pre_target_z])
        self.addObjective([0.7, 1.0], FS.positionRel, [gripper, self.obj], OT.eq, scale=x_axis * 1e0)
        self.addObjective([0.7, 1.0], FS.positionRel, [gripper, self.obj], OT.ineq, scale=y_axis * 1e0, target=[target])
        self.addObjective([0.7, 1.0], FS.positionRel, [gripper, self.obj], OT.ineq, scale=-y_axis * 1e0, target=[-target])
        self.addObjective([0.7, 1.0], products[dim], [gripper, self.obj], OT.eq, scale=[1e0], target=[dir])
        # encourage well centered grasps
        self.addObjective([0.7, 1.0], FS.positionRel, [gripper, self.obj], OT.eq, scale=y_axis * 1e-2, target=[0])
        # grasp
        self.addFrameDof("obj_free", gripper, JT.free, True, self.obj)
        self.addRigidSwitch(1.0, ["obj_free", self.obj])
        self.addObjective([1.0], FS.positionRel, [gripper, self.obj], OT.ineq, scale=z_axis * 1e0, target=[target])
        self.addObjective([1.0], FS.positionRel, [gripper, self.obj], OT.ineq, scale=-z_axis * 1e0, target=[-target])
        # release
        self.addFrameDof("obj_free_after", table, JT.free, True, self.obj)
        self.addRigidSwitch(2.0, ["obj_free_after", self.obj])
        self.addObjective([2.0, 3.0], products[dim], [gripper, self.obj], OT.eq, scale=[1e0], target=[1])
        self.addObjective([2.0, 3.0], FS.positionRel, [gripper, self.obj], OT.eq, scale=x_axis * 1e0, target=[0])

    def grasp_x_pos(self):
        self._grasp_obj(0, 1)

    def grasp_y_pos(self):
        self._grasp_obj(1, 1)

    def _pull_obj(self, dim: int, dir: int):
        self.action = "pull"
        products = [FS.scalarProductXX, FS.scalarProductXY]

        x_axis = np.eye(3)[dim]
        y_axis = np.eye(3)[1 - dim]
        z_axis = np.eye(3)[2]

        obj_bbox = self.get_bbox(self.obj)
        max_margin = 0.025
        gripper_margin_y = min(max_margin, 0.5 * obj_bbox[1])
        gripper_margin_z = min(max_margin, 0.5 * obj_bbox[2])

        # approach
        pre_target_z = (0.5 * obj_bbox[2]) + 0.1
        target_y = (0.5 * obj_bbox[1]) - gripper_margin_y
        self.addObjective([0.7], FS.positionRel, [gripper, self.obj], OT.eq, scale=z_axis * 1e0, target=[pre_target_z])
        self.addObjective([0.7, 1.0], FS.positionRel, [gripper, self.obj], OT.eq, scale=x_axis * 1e0, target=[0])
        self.addObjective([0.7, 1.0], FS.positionRel, [gripper, self.obj], OT.ineq, scale=y_axis * 1e0, target=[target_y])
        self.addObjective([0.7, 1.0], FS.positionRel, [gripper, self.obj], OT.ineq, scale=-y_axis * 1e0, target=[-target_y])
        self.addObjective([0.7, 1.0], products[dim], [gripper, self.obj], OT.eq, scale=[1e0], target=[dir])
        # contact
        target_z = (0.5 * obj_bbox[2]) - gripper_margin_z
        self.addObjective([1.0], FS.positionRel, [gripper, self.obj], OT.ineq, scale=z_axis * 1e0, target=[target_z])
        self.addObjective([1.0], FS.positionRel, [gripper, self.obj], OT.ineq, scale=-z_axis * 1e0, target=[-target_z])
        self.addFrameDof("obj_free", gripper, JT.free, True, self.obj)
        self.addRigidSwitch(1.0, ["obj_free", self.obj])
        # pull
        self.addObjective([1.0, 2.0], FS.position, [self.obj], OT.eq, scale=z_axis * 1e0, target=[0], order=1)
        # release
        self.addFrameDof("obj_free_after", table, JT.free, True, self.obj)
        self.addRigidSwitch(2.0, ["obj_free_after", self.obj])
        self.addObjective([2.0, 3.0], FS.positionRel, [gripper, self.obj], OT.eq, scale=x_axis * 1e0, target=[0], order=1)
        self.addObjective([2.0, 3.0], FS.quaternionRel, [gripper, self.obj], OT.eq, scale=[1e0], target=[], order=1)

    def pull_x_pos(self):
        self._pull_obj(0, 1)

    def pull_y_pos(self):
        self._pull_obj(1, 1)

    def target_pos_up_axis(self, dim: int, dir: int, align: FS, pos: np.typing.ArrayLike):
        """Places the object on top of the table at a certain position with the specified up axis"""
        self.addObjective([2.0], align, [self.obj], OT.eq, [1e1], target=[0, 0, dir])
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
        self.addObjective([2.0, 3.0], FS.positionRel, [self.obj, table], OT.eq, [1e1], target=pos)

    def target_quat(self, ori: np.typing.ArrayLike):
        self.addObjective([2.0, 3.0], FS.quaternionRel, [self.obj, table], OT.eq, scale=[1e1], target=ori)

    def target_pose(self, pose: np.typing.ArrayLike):
        self.addObjective([2.0, 3.0], FS.poseRel, [self.obj, table], OT.eq, scale=[1e1], target=pose)

    def solve(self):
        sol = NLP_Solver(self.nlp(), DEBUG.value)
        sol.setOptions(damping=1e-1, stopTolerance=1e-3, lambdaMax=100.0, stopInners=20, stopEvals=200)
        return sol.solve(verbose=DEBUG.value)

    def get_grasp_pose(self) -> np.ndarray:
        return self.getFrame(gripper, 1).getPose()

    def get_grasp_transform(self) -> np.ndarray:
        return self.getFrame(gripper, 1).getTransform()

    def get_bbox(self, frame_name):
        vertices = self.config.getFrame(frame_name).getMesh()[0]

        if vertices is None or not isinstance(vertices, np.ndarray) or vertices.ndim == 0:
            print(f"Warning: No vertices found for {frame_name}")
            return np.array([0.1, 0.1, 0.1])

        return np.max(vertices, axis=0) - np.min(vertices, axis=0)

    def simulate(self, view=True, times=1.0, tau=5e-3):
        sim = Simulation(self.config, SimulationEngine.physx, verbose=DEBUG.value)
        sim_steps = int(times // tau)
        splits = np.split(self.getPath(), int(self.phases))

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

        if self.action != "push":
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
