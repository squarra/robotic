import time

import numpy as np

from robotic import FS, JT, KOMO, OT, CameraView, Config, ControlMode, NLP_Solver, Simulation, SimulationEngine, depthImage2PointCloud
from robotic.helpers import DEBUG

gripper = "l_gripper"
table = "table"
obj = "obj"


class Manipulation:
    def __init__(self, C: Config, slices=1, homing_scale=1e-2, velocity_scale=1e-1, collisions=True, joint_limits=True, quaternion_norms=False):
        self.initial_C = C
        self.slices = slices
        self.reset(slices, homing_scale, velocity_scale, collisions, joint_limits, quaternion_norms)

        self.cameras = {0: "cameraTop", 1: "cameraWrist"}
        self.action = None

    def reset(self, slices=1, homing_scale=1e-2, velocity_scale=1e-1, collisions=True, joint_limits=True, quaternion_norms=False):
        self.komo = KOMO(self.initial_C, 2.0, slices, 1, collisions)
        self.komo.set_viewer(self.initial_C.get_viewer())
        self.komo.addControlObjective([], 0, homing_scale)
        self.komo.addControlObjective([], 1, velocity_scale)

        if collisions:
            self.komo.addObjective([], FS.accumulatedCollisions, [], OT.eq, [1e0])
        if joint_limits:
            self.komo.addObjective([], FS.jointLimits, [], OT.ineq, [1e0])
        if quaternion_norms:
            self.komo.addQuaternionNorms()

    def pull_obj(self):
        self.action = "pull"
        self.komo.addFrameDof("obj_trans", gripper, JT.free, True, obj)
        self.komo.addRigidSwitch(1.0, ["obj_trans", obj])

        helper_end = "_pull_end"
        self.komo.addFrameDof(helper_end, table, JT.transXYPhi, True, obj)
        self.komo.addObjective([1.0], FS.vectorZ, [gripper], OT.eq, [1e1], np.array([0, 0, 1]))
        self.komo.addObjective([2.0], FS.vectorZ, [gripper], OT.eq, [1e1], np.array([0, 0, 1]))
        self.komo.addObjective([1.0], FS.vectorZ, [obj], OT.eq, [1e1], np.array([0, 0, 1]))
        self.komo.addObjective([2.0], FS.vectorZ, [obj], OT.eq, [1e1], np.array([0, 0, 1]))
        self.komo.addObjective([2.0], FS.positionDiff, [obj, helper_end], OT.eq, [1e1])
        self.komo.addObjective([1.0], FS.positionRel, [gripper, obj], OT.eq, 1e1 * np.eye(3)[:2], np.array([0, 0, 0]))
        self.komo.addObjective([1.0], FS.negDistance, [gripper, obj], OT.eq, [1e1], [-0.005])

    def _push_obj(self, dim: int, dir: int):
        self.action = "push"
        self.komo.addFrameDof("obj_trans", table, JT.transXY, False, obj)
        self.komo.addRigidSwitch(1.0, ["obj_trans", obj])

        y_axis = np.eye(3)[dim]
        xz_plane = np.delete(np.eye(3), dim, axis=0)

        # gripper position
        target = -dir * (0.5 * self.get_bbox(obj)[dim] + 0.025)
        self.komo.addObjective([1.0, 2.0], FS.positionRel, [gripper, obj], OT.eq, scale=y_axis * 1e1, target=[target])
        self.komo.addObjective([1.0, 2.0], FS.positionRel, [gripper, obj], OT.eq, scale=xz_plane * 1e1)
        # gripper orientation
        self.komo.addObjective([1.0, 2.0], FS.vectorZ, [gripper], OT.eq, [1e0], [0, 0, 1])
        self.komo.addObjective([1.0, 2.0], FS.scalarProductYX, [gripper, obj], OT.eq, [1e0], [y_axis[0]])
        self.komo.addObjective([1.0, 2.0], FS.scalarProductYY, [gripper, obj], OT.eq, [1e0], [y_axis[1]])
        self.komo.addObjective([1.0, 2.0], FS.scalarProductYZ, [gripper, obj], OT.eq, [1e0], [y_axis[2]])

    def push_obj_x(self):
        self._push_obj(0, 1)

    def push_obj_y(self):
        self._push_obj(1, 1)

    def push_obj_z(self):
        self._push_obj(2, 1)

    def push_obj_x_neg(self):
        self._push_obj(0, -1)

    def push_obj_y_neg(self):
        self._push_obj(1, -1)

    def push_obj_z_neg(self):
        self._push_obj(2, -1)

    def _grasp_obj(self, dim: int, align: tuple):
        self.action = "grasp"
        self.komo.addFrameDof("obj_trans", gripper, JT.free, True, obj)
        self.komo.addRigidSwitch(1.0, ["obj_trans", obj])

        x_axis = np.eye(3)[dim]
        yz_plane = np.delete(np.eye(3), dim, axis=0)

        # gripper position
        target = 0.5 * self.get_bbox(obj) - 0.02
        self.komo.addObjective([1.0], FS.positionRel, [gripper, obj], OT.eq, scale=x_axis * 1e1)
        self.komo.addObjective([1.0], FS.positionRel, [gripper, obj], OT.ineq, scale=yz_plane * 1e1, target=[target])
        self.komo.addObjective([1.0], FS.positionRel, [gripper, obj], OT.ineq, scale=yz_plane * (-1e1), target=[-target])
        # gripper orientation
        self.komo.addObjective([0.8, 1.0], align[0], [gripper, obj], OT.eq, [1e0])
        self.komo.addObjective([0.8, 1.0], align[1], [gripper, obj], OT.eq, [1e0])

    def grasp_obj_x(self):
        self._grasp_obj(0, [FS.scalarProductXY, FS.scalarProductXZ])

    def grasp_obj_y(self):
        self._grasp_obj(1, [FS.scalarProductXX, FS.scalarProductXZ])

    def grasp_obj_z(self):
        self._grasp_obj(2, [FS.scalarProductXX, FS.scalarProductXY])

    def target_pos(self, pos: list[float]):
        self.komo.addObjective([2.0], FS.positionRel, [obj, table], OT.eq, [1e1], target=pos)

    def target_ori(self, ori: list[float]):
        self.komo.addObjective([2.0], FS.quaternionRel, [obj, table], OT.eq, scale=[1e1], target=ori)

    def solve(self):
        sol = NLP_Solver(self.komo.nlp(), DEBUG.value)
        sol.setOptions(damping=1e-1, stopTolerance=1e-3, lambdaMax=100.0, stopInners=20, stopEvals=200)
        return sol.solve(verbose=DEBUG.value)

    def get_cam(self, cam_id=0):
        cam = CameraView(self.config)
        cam.setCamera(self.config.getFrame(self.cameras[cam_id]))
        return cam

    def get_rgbd(self, cam_id=0) -> tuple[np.ndarray, np.ndarray]:
        return self.get_cam(cam_id).computeImageAndDepth(self.config)

    def get_pcl(self, cam_id=0) -> np.ndarray:
        cam = self.get_cam(cam_id)
        return depthImage2PointCloud(cam.computeImageAndDepth(self.config)[1], cam.getFxycxy())

    def view_pcl(self, pcl: np.ndarray, cam_id=0):
        self.config.addFrame("pcl", self.cameras[cam_id]).setPointCloud(pcl, [255, 0, 0])
        self.view(cam_id)
        self.config.delFrame("pcl")

    def view(self, cam_id=-1):
        if cam_id in self.cameras:
            self.config.get_viewer().setCamera(self.config.getFrame(self.cameras[cam_id]))
        self.config.view()

    def view_komo(self, cam_id=-1, **kwargs):
        if cam_id in self.cameras:
            self.komo.get_viewer().setCamera(self.config.getFrame(self.cameras[cam_id]))
        self.komo.view(**kwargs)

    def get_segmentation_img(self, cam_id=0):
        return self.get_cam(cam_id).computeSegmentationImage()

    def get_grasp_pose(self) -> np.ndarray:
        return self.komo.getFrame(gripper, 1).getPose()

    def get_grasp_transform(self) -> np.ndarray:
        return self.komo.getFrame(gripper, 1).getTransform()

    @property
    def config(self):
        return self.komo.getConfig()

    @property
    def primitives(self):
        return [
            self.pull_obj,
            self.push_obj_x,
            self.push_obj_y,
            self.push_obj_z,
            self.push_obj_x_neg,
            self.push_obj_y_neg,
            self.push_obj_z_neg,
            self.grasp_obj_x,
            self.grasp_obj_y,
            self.grasp_obj_z,
        ]

    def get_bbox(self, frame_name):
        vertices = self.config.getFrame(frame_name).getMesh()[0]

        if vertices is None or not isinstance(vertices, np.ndarray) or vertices.ndim == 0:
            print(f"Warning: No vertices found for {frame_name}")
            return np.array([0.1, 0.1, 0.1])

        return np.max(vertices, axis=0) - np.min(vertices, axis=0)

    def simulate(self, times=1.0, tau=5e-3):
        sim = Simulation(self.config, SimulationEngine.physx, verbose=DEBUG.value)
        sim_steps = int(times // tau)
        splits = np.split(self.komo.getPath(), [self.slices])

        sim.setSplineRef(path=splits[0], times=[times])
        for _ in range(sim_steps):
            sim.step([], tau, ControlMode.spline)
            time.sleep(tau)
            self.config.view()

        if self.action in ["push", "grasp"]:
            sim.moveGripper(gripper, 0.0)
            while not sim.gripperIsDone(gripper):
                sim.step([], tau, ControlMode.spline)
                time.sleep(tau)
                self.config.view()

        sim.setSplineRef(path=splits[1], times=[1])
        for _ in range(sim_steps):
            sim.step([], tau, ControlMode.spline)
            time.sleep(tau)
            self.config.view()
