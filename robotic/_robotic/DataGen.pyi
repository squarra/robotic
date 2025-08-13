import _robotic

class RndStableConfigs:
    """A generator of random stable configurations"""
    def __init__(self) -> None:
        """__init__(self: _robotic.DataGen.RndStableConfigs) -> None"""
    def getSample(self, config: _robotic.Scenario, supports: StringA) -> bool:
        """getSample(self: _robotic.DataGen.RndStableConfigs, config: _robotic.Config, supports: StringA) -> bool

        sample a random configuration - displayed, access via config passed at construction
        """
    def report(self) -> None:
        """report(self: _robotic.DataGen.RndStableConfigs) -> None

        info on newton steps -per- feasible sample
        """
    def setOptions(self, verbose: int = ..., frictionCone_mu: float = ...) -> RndStableConfigs:
        """setOptions(self: _robotic.DataGen.RndStableConfigs, verbose: int = 1, frictionCone_mu: float = 0.8) -> _robotic.DataGen.RndStableConfigs

        set options
        """

class ShapenetGrasps:
    """A generator of random grasps on random shapenet objects"""
    def __init__(self) -> None:
        """__init__(self: _robotic.DataGen.ShapenetGrasps) -> None"""
    def displaySamples(self, samples: arr, context: uintA, scores: arr = ...) -> None:
        """displaySamples(self: _robotic.DataGen.ShapenetGrasps, samples: arr, context: uintA, scores: arr = array(0.0078125)) -> None

        (batch interface) displays all samples
        """
    def evaluateGrasp(self) -> arr:
        """evaluateGrasp(self: _robotic.DataGen.ShapenetGrasps) -> arr

        (direct interface) return scores of grasp candidate (min(scores)<0. means fail)
        """
    def evaluateSample(self, sample: arr, context: int) -> arr:
        """evaluateSample(self: _robotic.DataGen.ShapenetGrasps, sample: arr, context: int) -> arr

        (batch interface) returns scores for a single sample - this (row) are numbers where a single 'negative' means fail
        """
    def getConfig(self) -> _robotic.Scenario:
        """getConfig(self: _robotic.DataGen.ShapenetGrasps) -> _robotic.Config"""
    def getEvalGripperPoses(self) -> arr:
        """getEvalGripperPoses(self: _robotic.DataGen.ShapenetGrasps) -> arr

        return the relative gripper after each motion phase: esp. poses[1] (after closing fingers) is interesting; the later ones allow you to estimate relative motion yourself)
        """
    def getPointCloud(self) -> arr:
        """getPointCloud(self: _robotic.DataGen.ShapenetGrasps) -> arr

        (direct interface) return pcl of loaded object
        """
    def getPointNormals(self) -> arr:
        """getPointNormals(self: _robotic.DataGen.ShapenetGrasps) -> arr

        (direct interface) return point normals of the pcl of loaded object
        """
    def getSamples(self, nSamples: int) -> tuple[arr, uintA, arr]:
        """getSamples(self: _robotic.DataGen.ShapenetGrasps, nSamples: int) -> tuple[arr, uintA, arr]

        (batch interface) return three arrays: samples X, contexts Z, scores S (each row are scores for one sample - see evaluateSamples)
        """
    def loadObject(self, shape: int, rndPose: bool = ...) -> bool:
        """loadObject(self: _robotic.DataGen.ShapenetGrasps, shape: int, rndPose: bool = True) -> bool

        (direct interface) clear scene and load object and gripper
        """
    def resetObjectPose(self, idx: int = ..., rndOrientation: bool = ...) -> None:
        """resetObjectPose(self: _robotic.DataGen.ShapenetGrasps, idx: int = 0, rndOrientation: bool = True) -> None"""
    def sampleGraspPose(self) -> arr:
        """sampleGraspPose(self: _robotic.DataGen.ShapenetGrasps) -> arr

        (direct interface) return (relative) pose of random sampled grasp candidate
        """
    def setGraspPose(self, pose: arr, objPts: str = ...) -> None:
        """setGraspPose(self: _robotic.DataGen.ShapenetGrasps, pose: arr, objPts: str = 'obj0_pts') -> None

        (direct interface) set (relative) pose of grasp candidate
        """
    def setOptions(self, verbose: int = ..., filesPrefix=..., endShape: int = ..., startShape: int = ..., simVerbose: int = ..., optVerbose: int = ..., simTau: float = ..., gripperCloseSpeed: float = ..., moveSpeed: float = ..., pregraspNormalSdv: float = ...) -> ShapenetGrasps:
        """setOptions(self: _robotic.DataGen.ShapenetGrasps, verbose: int = 1, filesPrefix: rai::String = 'shapenet/models/', endShape: int = -1, startShape: int = 0, simVerbose: int = 0, optVerbose: int = 0, simTau: float = 0.01, gripperCloseSpeed: float = 0.001, moveSpeed: float = 0.005, pregraspNormalSdv: float = 0.2) -> _robotic.DataGen.ShapenetGrasps

        set options
        """
    def setPhysxOptions(self, verbose: int = ..., yGravity: bool = ..., angularDamping: float = ..., defaultFriction: float = ..., defaultRestitution: float = ..., motorKp: float = ..., motorKd: float = ...) -> ShapenetGrasps:
        """setPhysxOptions(self: _robotic.DataGen.ShapenetGrasps, verbose: int = 1, yGravity: bool = False, angularDamping: float = 0.1, defaultFriction: float = 1.0, defaultRestitution: float = 0.1, motorKp: float = 1000.0, motorKd: float = 100.0) -> _robotic.DataGen.ShapenetGrasps

        set options
        """

def sampleGraspCandidate(C: _robotic.Scenario, ptsFrame: str, refFrame: str, pregraspNormalSdv: float = ..., verbose: int = ...) -> arr:
    """sampleGraspCandidate(C: _robotic.Config, ptsFrame: str, refFrame: str, pregraspNormalSdv: float = 0.2, verbose: int = 1) -> arr

    sample random grasp candidates on an object represented as points
    """
