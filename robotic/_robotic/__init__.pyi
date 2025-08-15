import numpy
from . import DataGen as DataGen
from _typeshed import Incomplete
from typing import Callable, ClassVar, overload

class Actions2KOMO_Translator:
    """Actions2KOMO_Translator"""
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

class ArgWord:
    """[todo: replace by str]

    Members:

      _left

      _right

      _sequence

      _path"""
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    _left: ClassVar[ArgWord] = ...
    _path: ClassVar[ArgWord] = ...
    _right: ClassVar[ArgWord] = ...
    _sequence: ClassVar[ArgWord] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: _robotic.ArgWord, value: int) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: _robotic.ArgWord) -> int"""
    def __int__(self) -> int:
        """__int__(self: _robotic.ArgWord) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    @property
    def name(self) -> str:
        """name(self: object) -> str

        name(self: object) -> str
        """
    @property
    def value(self) -> int:
        """(arg0: _robotic.ArgWord) -> int"""

class BSpline:
    def __init__(self) -> None:
        """__init__(self: _robotic.BSpline) -> None

        non-initialized
        """
    def eval(self, sampleTimes: arr, derivative: int = ...) -> arr:
        """eval(self: _robotic.BSpline, sampleTimes: arr, derivative: int = 0) -> arr

        evaluate the spline (or its derivative) for given sampleTimes
        """
    def getBmatrix(self, sampleTimes: arr, startDuplicates: bool = ..., endDuplicates: bool = ...) -> arr:
        """getBmatrix(self: _robotic.BSpline, sampleTimes: arr, startDuplicates: bool = False, endDuplicates: bool = False) -> arr

        return the B-matrix mapping from ctrlPoints to (e.g. finer) sampleTimes (e.g. uniform linspace(0,1,T)
        """
    def getCtrlPoints(self) -> arr:
        """getCtrlPoints(self: _robotic.BSpline) -> arr"""
    def getKnots(self) -> arr:
        """getKnots(self: _robotic.BSpline) -> arr"""
    def setCtrlPoints(self, points: arr, addStartDuplicates: bool = ..., addEndDuplicates: bool = ..., setStartVel: arr = ..., setEndVel: arr = ...) -> None:
        """setCtrlPoints(self: _robotic.BSpline, points: arr, addStartDuplicates: bool = True, addEndDuplicates: bool = True, setStartVel: arr = array(0.0078125), setEndVel: arr = array(0.0078125)) -> None

        set the ctrl points, automatically duplicating them as needed at start/end, optionally setting vels at start/end
        """
    def setKnots(self, degree: int, times: arr) -> None:
        """setKnots(self: _robotic.BSpline, degree: int, times: arr) -> None

        set degree and knots by providing *times* (e.g. uniform linspace(0,1,T) -- duplicated knots at start/end and inter-time placing for even degrees is done internally
        """

class CameraView:
    """Offscreen rendering"""
    def __init__(self, config: Config, offscreen: bool = ...) -> None:
        """__init__(self: _robotic.CameraView, config: _robotic.Config, offscreen: bool = True) -> None

        constructor
        """
    def computeImageAndDepth(self, config: Config, visualsOnly: bool = ...) -> tuple:
        """computeImageAndDepth(self: _robotic.CameraView, config: _robotic.Config, visualsOnly: bool = True) -> tuple

        returns image and depth from a camera sensor; the 'config' argument needs to be the same configuration as in the constructor, but in new state
        """
    def computeSegmentationID(self, config: Config) -> uintA:
        """computeSegmentationID(self: _robotic.CameraView, config: _robotic.Config) -> uintA

        Update scene from config and return segmentation ID array
        """
    def computeSegmentationImage(self, *args, **kwargs):
        """computeSegmentationImage(self: _robotic.CameraView, config: _robotic.Config) -> Array<T>

        Update scene from config and return segmentation RGB image
        """
    def getFxycxy(self) -> arr:
        """getFxycxy(self: _robotic.CameraView) -> arr

        return the camera intrinsics f_x, f_y, c_x, c_y
        """
    def setCamera(self, *args, **kwargs):
        """setCamera(self: _robotic.CameraView, cameraFrameName: _robotic.Frame) -> rai::CameraView::Sensor

        select a camera, typically a frame that has camera info attributes
        """

class CameraViewSensor:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

class Config:
    """Core data structure to represent a kinematic configuration (essentially a tree of frames). See https://marctoussaint.github.io/robotics-course/tutorials/1a-configurations.html"""
    def __init__(self) -> None:
        """__init__(self: _robotic.Config) -> None

        initializes to an empty configuration, with no frames
        """
    def addConfigurationCopy(self, config: Config, prefix=..., tau: float = ...) -> Frame:
        """addConfigurationCopy(self: _robotic.Config, config: _robotic.Config, prefix: rai::String = '', tau: float = 1.0) -> _robotic.Frame"""
    def addFile(self, filename: str, namePrefix: str = ...) -> Frame:
        """addFile(self: _robotic.Config, filename: str, namePrefix: str = None) -> _robotic.Frame

        add the contents of the file to C
        """
    def addFrame(self, name: str, parent: str = ..., args: str = ...) -> Frame:
        """addFrame(self: _robotic.Config, name: str, parent: str = '', args: str = '') -> _robotic.Frame

        add a new frame to C; optionally make this a child to the given parent; use the Frame methods to set properties of the new frame
        """
    def addH5Object(self, framename: str, filename: str, verbose: int = ...) -> Frame:
        """addH5Object(self: _robotic.Config, framename: str, filename: str, verbose: int = 0) -> _robotic.Frame

        add the contents of the file to C
        """
    def animate(self) -> None:
        """animate(self: _robotic.Config) -> None

        displays while articulating all dofs in a row
        """
    def animateSpline(self, T: int = ...) -> None:
        """animateSpline(self: _robotic.Config, T: int = 3) -> None

        animate with random spline in limits bounding box [T=#spline points]
        """
    def asDict(self, parentsInKeys: bool = ...) -> dict:
        """asDict(self: _robotic.Config, parentsInKeys: bool = True) -> dict

        return the configuration description as a dict, e.g. for file export
        """
    def attach(self, arg0: str, arg1: str) -> None:
        """attach(self: _robotic.Config, arg0: str, arg1: str) -> None

        change the configuration by creating a rigid joint from frame1 to frame2, adopting their current relative pose. This also breaks the first joint that is parental to frame2 and reverses the topological order from frame2 to the broken joint
        """
    def checkConsistency(self) -> bool:
        """checkConsistency(self: _robotic.Config) -> bool

        internal use
        """
    def clear(self) -> None:
        """clear(self: _robotic.Config) -> None

        clear all frames and additional data; becomes the empty configuration, with no frames
        """
    def coll_totalViolation(self) -> float:
        """coll_totalViolation(self: _robotic.Config) -> float

        returns the sum of all penetrations (using FCL for broadphase; and low-level GJK/MRP for fine pair-wise distance/penetration computation)
        """
    def computeCollisions(self) -> None:
        """computeCollisions(self: _robotic.Config) -> None

        [should be obsolete; getCollision* methods auto ensure proxies] call the broadphase collision engine (SWIFT++ or FCL) to generate the list of collisions (or near proximities) between all frame shapes that have the collision tag set non-zero
        """
    def delFrame(self, frameName: str) -> None:
        """delFrame(self: _robotic.Config, frameName: str) -> None

        destroy and remove a frame from C
        """
    def eval(self, featureSymbol: FS, frames: StringA = ..., scale: arr = ..., target: arr = ..., order: int = ...) -> tuple:
        """eval(self: _robotic.Config, featureSymbol: _robotic.FS, frames: StringA = [], scale: arr = array(0.0078125), target: arr = array(0.0078125), order: int = -1) -> tuple

        evaluate a feature -- see https://marctoussaint.github.io/robotics-course/tutorials/features.html
        """
    def frame(self, frameID: int) -> Frame:
        """frame(self: _robotic.Config, frameID: int) -> _robotic.Frame

        get access to a frame by index (< getFrameDimension)
        """
    def getCollidablePairs(self) -> StringA:
        """getCollidablePairs(self: _robotic.Config) -> StringA

        returns the list of collisable pairs -- this should help debugging the 'contact' flag settings in a configuration
        """
    def getCollisions(self, belowMargin: float = ..., verbose: int = ...) -> list:
        """getCollisions(self: _robotic.Config, belowMargin: float = 0.0, verbose: int = 0) -> list

        return the results of collision computations: a list of 3 tuples with (frame1, frame2, distance). Optionally report only on distances below a margin To get really precise distances and penetrations use the FS.distance feature with the two frame names
        """
    def getFrame(self, frameName: str, warnIfNotExist: bool = ...) -> Frame:
        """getFrame(self: _robotic.Config, frameName: str, warnIfNotExist: bool = True) -> _robotic.Frame

        get access to a frame by name; use the Frame methods to set/get frame properties
        """
    def getFrameDimension(self) -> int:
        """getFrameDimension(self: _robotic.Config) -> int

        get the total number of frames
        """
    def getFrameNames(self) -> list[str]:
        """getFrameNames(self: _robotic.Config) -> list[str]

        get the list of frame names
        """
    def getFrameState(self) -> numpy.ndarray[numpy.float64]:
        """getFrameState(self: _robotic.Config) -> numpy.ndarray[numpy.float64]

        get the frame state as a n-times-7 numpy matrix, with a 7D pose per frame
        """
    def getFrames(self) -> list[Frame]:
        """getFrames(self: _robotic.Config) -> list[_robotic.Frame]"""
    def getJointDimension(self) -> int:
        """getJointDimension(self: _robotic.Config) -> int

        get the total number of degrees of freedom
        """
    def getJointIDs(self) -> uintA:
        """getJointIDs(self: _robotic.Config) -> uintA

        get indeces (which are the indices of their frames) of all joints
        """
    def getJointLimits(self) -> arr:
        """getJointLimits(self: _robotic.Config) -> arr

        get the joint limits as a n-by-2 matrix; for dofs that do not have limits defined, the entries are [0,-1] (i.e. upper limit < lower limit)
        """
    def getJointNames(self) -> StringA:
        """getJointNames(self: _robotic.Config) -> StringA

        get the list of joint names
        """
    def getJointState(self) -> arr:
        """getJointState(self: _robotic.Config) -> arr

        get the joint state as a numpy vector, optionally only for a subset of joints specified as list of joint names
        """
    def get_viewer(self) -> ConfigurationViewer:
        """get_viewer(self: _robotic.Config) -> _robotic.ConfigurationViewer"""
    def processInertias(self, recomputeInertias: bool = ..., transformToDiagInertia: bool = ...) -> None:
        """processInertias(self: _robotic.Config, recomputeInertias: bool = True, transformToDiagInertia: bool = False) -> None

        collect all inertia at root frame of links, optionally reestimate all inertias based on standard surface density, optionally relocate the link frame to the COM with diagonalized I)
        """
    def processStructure(self, pruneRigidJoints: bool = ..., reconnectToLinks: bool = ..., pruneNonContactShapes: bool = ..., pruneTransparent: bool = ...) -> None:
        """processStructure(self: _robotic.Config, pruneRigidJoints: bool = False, reconnectToLinks: bool = True, pruneNonContactShapes: bool = False, pruneTransparent: bool = False) -> None

        structurally simplify the Configuration (deleting frames, relinking to minimal tree)
        """
    def report(self) -> str:
        """report(self: _robotic.Config) -> str

        return a string with basic info (#frames, etc)
        """
    def selectJoints(self, jointNames: list[str], notThose: bool = ...) -> None:
        """selectJoints(self: _robotic.Config, jointNames: list[str], notThose: bool = False) -> None

        redefine what are considered the DOFs of this configuration: only joints listed in jointNames are considered part of the joint state and define the number of DOFs
        """
    def selectJointsBySubtree(self, root: Frame) -> None:
        """selectJointsBySubtree(self: _robotic.Config, root: _robotic.Frame) -> None"""
    @overload
    def setFrameState(self, X: list[float], frames: list[str] = ...) -> None:
        """setFrameState(*args, **kwargs)
        Overloaded function.

        1. setFrameState(self: _robotic.Config, X: list[float], frames: list[str] = []) -> None

        set the frame state, optionally only for a subset of frames specified as list of frame names

        2. setFrameState(self: _robotic.Config, X: numpy.ndarray, frames: list[str] = []) -> None

        set the frame state, optionally only for a subset of frames specified as list of frame names
        """
    @overload
    def setFrameState(self, X: numpy.ndarray, frames: list[str] = ...) -> None:
        """setFrameState(*args, **kwargs)
        Overloaded function.

        1. setFrameState(self: _robotic.Config, X: list[float], frames: list[str] = []) -> None

        set the frame state, optionally only for a subset of frames specified as list of frame names

        2. setFrameState(self: _robotic.Config, X: numpy.ndarray, frames: list[str] = []) -> None

        set the frame state, optionally only for a subset of frames specified as list of frame names
        """
    def setJointState(self, q: arr, joints: list = ...) -> None:
        """setJointState(self: _robotic.Config, q: arr, joints: list = []) -> None

        set the joint state, optionally only for a subset of joints specified as list of joint names
        """
    def setJointStateSlice(self, arg0: list[float], arg1: int) -> None:
        """setJointStateSlice(self: _robotic.Config, arg0: list[float], arg1: int) -> None"""
    def set_viewer(self, arg0: ConfigurationViewer) -> None:
        """set_viewer(self: _robotic.Config, arg0: _robotic.ConfigurationViewer) -> None"""
    def view(self, pause: bool = ..., message: str = ...) -> int:
        """view(self: _robotic.Config, pause: bool = False, message: str = None) -> int

        open a view window for the configuration
        """
    def view_close(self) -> None:
        """view_close(self: _robotic.Config) -> None

        close the view
        """
    def view_recopyMeshes(self) -> None:
        """view_recopyMeshes(self: _robotic.Config) -> None"""
    def viewer(self) -> ConfigurationViewer:
        """viewer(self: _robotic.Config) -> _robotic.ConfigurationViewer"""
    def watchFile(self, arg0: str) -> None:
        """watchFile(self: _robotic.Config, arg0: str) -> None

        launch a viewer that listents (inode) to changes of a file (made by you in an editor), and reloads, displays and animates the configuration whenever the file is changed
        """
    def write(self, *args, **kwargs):
        """write(self: _robotic.Config) -> rai::String

        return the configuration description as a str (similar to YAML), e.g. for file export
        """
    def writeCollada(self, filename: str, format: str = ...) -> None:
        """writeCollada(self: _robotic.Config, filename: str, format: str = 'collada') -> None

        write the full configuration in a collada file for export
        """
    def writeMesh(self, filename: str) -> None:
        """writeMesh(self: _robotic.Config, filename: str) -> None

        write the full configuration in a ply mesh file
        """
    def writeMeshes(self, pathPrefix, copyTextures: bool = ..., enumerateAssets: bool = ...) -> None:
        """writeMeshes(self: _robotic.Config, pathPrefix: rai::String, copyTextures: bool = True, enumerateAssets: bool = False) -> None

        write all object meshes in a directory
        """
    def writeURDF(self) -> str:
        """writeURDF(self: _robotic.Config) -> str

        write the full configuration as URDF in a string, e.g. for file export
        """

class ConfigurationViewer:
    """internal viewer handle (gl window)"""
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def focus(self, position_7d: arr, heightAbs: float = ...) -> None:
        """focus(self: _robotic.ConfigurationViewer, position_7d: arr, heightAbs: float = 1.0) -> None

        focus at a 3D position; second argument distances camara so that view window has roughly given absHeight around object
        """
    def getCamera_focalLength(self) -> float:
        """getCamera_focalLength(self: _robotic.ConfigurationViewer) -> float

        return the focal length of the view camera (only intrinsic parameter)
        """
    def getCamera_fxycxy(self) -> arr:
        """getCamera_fxycxy(self: _robotic.ConfigurationViewer) -> arr

        return (fx, fy, cx, cy): the focal length and image center in PIXEL UNITS
        """
    def getCamera_pose(self) -> arr:
        """getCamera_pose(self: _robotic.ConfigurationViewer) -> arr

        get the camera pose directly
        """
    def getDepth(self, *args, **kwargs):
        """getDepth(self: _robotic.ConfigurationViewer) -> Array<T>

        return the view's depth array (scaled to meters)
        """
    def getEventCursor(self) -> arr:
        """getEventCursor(self: _robotic.ConfigurationViewer) -> arr

        return the position and normal of the 'curser': mouse position 3D projected into scene via depth, and 3D normal of depth map -- returned as 6D vector
        """
    def getEventCursorObject(self) -> int:
        """getEventCursorObject(self: _robotic.ConfigurationViewer) -> int

        (aka mouse picking) return the frame ID (or -1) that the 'cursor' currently points at
        """
    def getEvents(self) -> StringA:
        """getEvents(self: _robotic.ConfigurationViewer) -> StringA

        return accumulated events as list of strings
        """
    def getRgb(self, *args, **kwargs):
        """getRgb(self: _robotic.ConfigurationViewer) -> Array<T>

        return the view's rgb image
        """
    def raiseWindow(self) -> None:
        """raiseWindow(self: _robotic.ConfigurationViewer) -> None

        raise the window
        """
    def savePng(self, saveVideoPath=..., count: int = ...) -> None:
        """savePng(self: _robotic.ConfigurationViewer, saveVideoPath: rai::String = 'z.vid/', count: int = -1) -> None

        saves a png image of the current view, numbered with a global counter, with the intention to make a video
        """
    def setCamera(self, camFrame: Frame) -> None:
        """setCamera(self: _robotic.ConfigurationViewer, camFrame: _robotic.Frame) -> None

        set the camera pose to a frame, and check frame attributes for intrinsic parameters (focalLength, width height)
        """
    def setCameraPose(self, pose_7d: arr) -> None:
        """setCameraPose(self: _robotic.ConfigurationViewer, pose_7d: arr) -> None

        set the camera pose directly
        """
    def setWindow(self, title: str, width: int, height: int) -> None:
        """setWindow(self: _robotic.ConfigurationViewer, title: str, width: int, height: int) -> None

        set title, width, and height
        """
    def setupEventHandler(self, blockDefaultHandler: bool) -> None:
        """setupEventHandler(self: _robotic.ConfigurationViewer, blockDefaultHandler: bool) -> None

        setup callbacks to grab window events and return them with methods below
        """
    def visualsOnly(self, _visualsOnly: bool = ...) -> None:
        """visualsOnly(self: _robotic.ConfigurationViewer, _visualsOnly: bool = True) -> None

        display only visuals (no markers/transparent/text)
        """

class ControlMode:
    """Members:

      none

      position

      velocity

      acceleration

      spline"""
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    acceleration: ClassVar[ControlMode] = ...
    none: ClassVar[ControlMode] = ...
    position: ClassVar[ControlMode] = ...
    spline: ClassVar[ControlMode] = ...
    velocity: ClassVar[ControlMode] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: _robotic.ControlMode, value: int) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: _robotic.ControlMode) -> int"""
    def __int__(self) -> int:
        """__int__(self: _robotic.ControlMode) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    @property
    def name(self) -> str:
        """name(self: object) -> str

        name(self: object) -> str
        """
    @property
    def value(self) -> int:
        """(arg0: _robotic.ControlMode) -> int"""

class FS:
    """Members:

      position

      positionDiff

      positionRel

      quaternion

      quaternionDiff

      quaternionRel

      pose

      poseDiff

      poseRel

      vectorX

      vectorXDiff

      vectorXRel

      vectorY

      vectorYDiff

      vectorYRel

      vectorZ

      vectorZDiff

      vectorZRel

      scalarProductXX

      scalarProductXY

      scalarProductXZ

      scalarProductYX

      scalarProductYY

      scalarProductYZ

      scalarProductZZ

      gazeAt

      angularVel

      accumulatedCollisions

      jointLimits

      distance

      negDistance

      oppose

      qItself

      jointState

      aboveBox

      insideBox

      pairCollision_negScalar

      pairCollision_vector

      pairCollision_normal

      pairCollision_p1

      pairCollision_p2

      standingAbove

      physics

      contactConstraints

      energy

      transAccelerations

      transVelocities

      qQuaternionNorms

      opposeCentral

      linangVel

      AlignXWithDiff

      AlignYWithDiff"""
    __members__: ClassVar[dict] = ...  # read-only
    AlignXWithDiff: ClassVar[FS] = ...
    AlignYWithDiff: ClassVar[FS] = ...
    __entries: ClassVar[dict] = ...
    aboveBox: ClassVar[FS] = ...
    accumulatedCollisions: ClassVar[FS] = ...
    angularVel: ClassVar[FS] = ...
    contactConstraints: ClassVar[FS] = ...
    distance: ClassVar[FS] = ...
    energy: ClassVar[FS] = ...
    gazeAt: ClassVar[FS] = ...
    insideBox: ClassVar[FS] = ...
    jointLimits: ClassVar[FS] = ...
    jointState: ClassVar[FS] = ...
    linangVel: ClassVar[FS] = ...
    negDistance: ClassVar[FS] = ...
    oppose: ClassVar[FS] = ...
    opposeCentral: ClassVar[FS] = ...
    pairCollision_negScalar: ClassVar[FS] = ...
    pairCollision_normal: ClassVar[FS] = ...
    pairCollision_p1: ClassVar[FS] = ...
    pairCollision_p2: ClassVar[FS] = ...
    pairCollision_vector: ClassVar[FS] = ...
    physics: ClassVar[FS] = ...
    pose: ClassVar[FS] = ...
    poseDiff: ClassVar[FS] = ...
    poseRel: ClassVar[FS] = ...
    position: ClassVar[FS] = ...
    positionDiff: ClassVar[FS] = ...
    positionRel: ClassVar[FS] = ...
    qItself: ClassVar[FS] = ...
    qQuaternionNorms: ClassVar[FS] = ...
    quaternion: ClassVar[FS] = ...
    quaternionDiff: ClassVar[FS] = ...
    quaternionRel: ClassVar[FS] = ...
    scalarProductXX: ClassVar[FS] = ...
    scalarProductXY: ClassVar[FS] = ...
    scalarProductXZ: ClassVar[FS] = ...
    scalarProductYX: ClassVar[FS] = ...
    scalarProductYY: ClassVar[FS] = ...
    scalarProductYZ: ClassVar[FS] = ...
    scalarProductZZ: ClassVar[FS] = ...
    standingAbove: ClassVar[FS] = ...
    transAccelerations: ClassVar[FS] = ...
    transVelocities: ClassVar[FS] = ...
    vectorX: ClassVar[FS] = ...
    vectorXDiff: ClassVar[FS] = ...
    vectorXRel: ClassVar[FS] = ...
    vectorY: ClassVar[FS] = ...
    vectorYDiff: ClassVar[FS] = ...
    vectorYRel: ClassVar[FS] = ...
    vectorZ: ClassVar[FS] = ...
    vectorZDiff: ClassVar[FS] = ...
    vectorZRel: ClassVar[FS] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: _robotic.FS, value: int) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: _robotic.FS) -> int"""
    def __int__(self) -> int:
        """__int__(self: _robotic.FS) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    @property
    def name(self) -> str:
        """name(self: object) -> str

        name(self: object) -> str
        """
    @property
    def value(self) -> int:
        """(arg0: _robotic.FS) -> int"""

class Frame:
    """A (coordinate) frame of a configuration, which can have a parent, and associated shape, joint, and/or inertia"""
    name: Incomplete
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def asDict(self) -> dict:
        """asDict(self: _robotic.Frame) -> dict"""
    def computeCompoundInertia(self) -> Frame:
        """computeCompoundInertia(self: _robotic.Frame) -> _robotic.Frame"""
    def convertDecomposedShapeToChildFrames(self) -> Frame:
        """convertDecomposedShapeToChildFrames(self: _robotic.Frame) -> _robotic.Frame"""
    def ensure_X(self) -> None:
        """ensure_X(self: _robotic.Frame) -> None

        Ensure the absolute pose X is up-to-date
        """
    def getAttributes(self) -> dict:
        """getAttributes(self: _robotic.Frame) -> dict

        get frame attributes
        """
    def getChildren(self) -> list[Frame]:
        """getChildren(self: _robotic.Frame) -> list[_robotic.Frame]"""
    def getJointState(self) -> arr:
        """getJointState(self: _robotic.Frame) -> arr"""
    def getJointType(self) -> JT:
        """getJointType(self: _robotic.Frame) -> _robotic.JT"""
    def getMesh(self) -> tuple:
        """getMesh(self: _robotic.Frame) -> tuple"""
    def getMeshColors(self, *args, **kwargs):
        """getMeshColors(self: _robotic.Frame) -> Array<T>"""
    def getMeshPoints(self) -> arr:
        """getMeshPoints(self: _robotic.Frame) -> arr"""
    def getMeshTriangles(self) -> uintA:
        """getMeshTriangles(self: _robotic.Frame) -> uintA"""
    def getParent(self) -> Frame:
        """getParent(self: _robotic.Frame) -> _robotic.Frame"""
    def getPose(self) -> arr:
        """getPose(self: _robotic.Frame) -> arr"""
    def getPosition(self) -> arr:
        """getPosition(self: _robotic.Frame) -> arr"""
    def getQuaternion(self) -> arr:
        """getQuaternion(self: _robotic.Frame) -> arr"""
    def getRelativePose(self) -> arr:
        """getRelativePose(self: _robotic.Frame) -> arr"""
    def getRelativePosition(self) -> arr:
        """getRelativePosition(self: _robotic.Frame) -> arr"""
    def getRelativeQuaternion(self) -> arr:
        """getRelativeQuaternion(self: _robotic.Frame) -> arr"""
    def getRelativeTransform(self) -> arr:
        """getRelativeTransform(self: _robotic.Frame) -> arr"""
    def getRotationMatrix(self) -> arr:
        """getRotationMatrix(self: _robotic.Frame) -> arr"""
    def getShapeType(self) -> ST:
        """getShapeType(self: _robotic.Frame) -> _robotic.ST"""
    def getSize(self) -> arr:
        """getSize(self: _robotic.Frame) -> arr"""
    def getTransform(self) -> arr:
        """getTransform(self: _robotic.Frame) -> arr"""
    def makeRoot(self, untilPartBreak: bool) -> None:
        """makeRoot(self: _robotic.Frame, untilPartBreak: bool) -> None"""
    def setAttributes(self, arg0: dict) -> Frame:
        """setAttributes(self: _robotic.Frame, arg0: dict) -> _robotic.Frame

        set attributes for the frame
        """
    def setColor(self, arg0: arr) -> Frame:
        """setColor(self: _robotic.Frame, arg0: arr) -> _robotic.Frame"""
    def setContact(self, arg0: int) -> Frame:
        """setContact(self: _robotic.Frame, arg0: int) -> _robotic.Frame"""
    def setConvexMesh(self, points: arr, colors=..., radius: float = ...) -> Frame:
        """setConvexMesh(self: _robotic.Frame, points: arr, colors: Array<T> = array(1, dtype=uint8), radius: float = 0.0) -> _robotic.Frame

        attach a convex mesh as shape
        """
    def setImplicitSurface(self, data, size: arr, blur: int, resample: float = ...) -> Frame:
        """setImplicitSurface(self: _robotic.Frame, data: Array<T>, size: arr, blur: int, resample: float = -1.0) -> _robotic.Frame"""
    def setJoint(self, jointType: JT, limits: arr = ..., scale: float = ..., mimic: Frame = ...) -> Frame:
        """setJoint(self: _robotic.Frame, jointType: _robotic.JT, limits: arr = array(0.0078125), scale: float = 1.0, mimic: _robotic.Frame = None) -> _robotic.Frame"""
    def setJointState(self, arg0: arr) -> Frame:
        """setJointState(self: _robotic.Frame, arg0: arr) -> _robotic.Frame"""
    def setLines(self, verts: arr, colors=..., singleConnectedLine: bool = ...) -> Frame:
        """setLines(self: _robotic.Frame, verts: arr, colors: Array<T> = array(1, dtype=uint8), singleConnectedLine: bool = False) -> _robotic.Frame

        attach lines as shape
        """
    def setMass(self, mass: float, inertiaMatrix: arr = ...) -> Frame:
        """setMass(self: _robotic.Frame, mass: float, inertiaMatrix: arr = array(0.0078125)) -> _robotic.Frame"""
    def setMesh(self, vertices: arr, triangles: uintA, colors=..., cvxParts: uintA = ...) -> Frame:
        """setMesh(self: _robotic.Frame, vertices: arr, triangles: uintA, colors: Array<T> = array(1, dtype=uint8), cvxParts: uintA = array(0, dtype=uint32)) -> _robotic.Frame

        attach a mesh shape
        """
    def setMeshAsLines(self, arg0: list[float]) -> None:
        """setMeshAsLines(self: _robotic.Frame, arg0: list[float]) -> None"""
    def setMeshFile(self, filename, scale: float = ...) -> Frame:
        """setMeshFile(self: _robotic.Frame, filename: rai::String, scale: float = 1.0) -> _robotic.Frame

        attach a mesh shape from a file
        """
    def setParent(self, parent: Frame, keepAbsolutePose_and_adaptRelativePose: bool = ..., checkForLoop: bool = ...) -> Frame:
        """setParent(self: _robotic.Frame, parent: _robotic.Frame, keepAbsolutePose_and_adaptRelativePose: bool = False, checkForLoop: bool = False) -> _robotic.Frame"""
    def setPointCloud(self, points: arr, colors=..., normals: arr = ...) -> Frame:
        """setPointCloud(self: _robotic.Frame, points: arr, colors: Array<T> = array(1, dtype=uint8), normals: arr = array(0.0078125)) -> _robotic.Frame

        attach a point cloud shape
        """
    def setPose(self, arg0: arr) -> Frame:
        """setPose(self: _robotic.Frame, arg0: arr) -> _robotic.Frame"""
    def setPoseByText(self, arg0: str) -> Frame:
        """setPoseByText(self: _robotic.Frame, arg0: str) -> _robotic.Frame"""
    def setPosition(self, arg0: arr) -> Frame:
        """setPosition(self: _robotic.Frame, arg0: arr) -> _robotic.Frame"""
    def setQuaternion(self, arg0: arr) -> Frame:
        """setQuaternion(self: _robotic.Frame, arg0: arr) -> _robotic.Frame"""
    def setRelativePose(self, arg0: arr) -> Frame:
        """setRelativePose(self: _robotic.Frame, arg0: arr) -> _robotic.Frame"""
    def setRelativePoseByText(self, arg0: str) -> Frame:
        """setRelativePoseByText(self: _robotic.Frame, arg0: str) -> _robotic.Frame"""
    def setRelativePosition(self, arg0: arr) -> Frame:
        """setRelativePosition(self: _robotic.Frame, arg0: arr) -> _robotic.Frame"""
    def setRelativeQuaternion(self, arg0: arr) -> Frame:
        """setRelativeQuaternion(self: _robotic.Frame, arg0: arr) -> _robotic.Frame"""
    def setRelativeRotationMatrix(self, arg0: arr) -> Frame:
        """setRelativeRotationMatrix(self: _robotic.Frame, arg0: arr) -> _robotic.Frame"""
    def setRotationMatrix(self, arg0: arr) -> Frame:
        """setRotationMatrix(self: _robotic.Frame, arg0: arr) -> _robotic.Frame"""
    def setShape(self, type: ST, size: arr) -> Frame:
        """setShape(self: _robotic.Frame, type: _robotic.ST, size: arr) -> _robotic.Frame"""
    def setTensorShape(self, data, size: arr) -> Frame:
        """setTensorShape(self: _robotic.Frame, data: Array<T>, size: arr) -> _robotic.Frame"""
    def setTextureFile(self, image_filename, texCoords: arr = ...) -> Frame:
        """setTextureFile(self: _robotic.Frame, image_filename: rai::String, texCoords: arr = array(0.0078125)) -> _robotic.Frame

        set the texture of the mesh of a shape
        """
    def transformToDiagInertia(self, *args, **kwargs):
        """transformToDiagInertia(self: _robotic.Frame, arg0: bool) -> rai::Transformation"""
    def unLink(self) -> Frame:
        """unLink(self: _robotic.Frame) -> _robotic.Frame"""
    @property
    def ID(self) -> int:
        """the unique ID of the frame, which is also its index in lists/arrays (e.g. when the frameState is returned as matrix) (readonly)
        (self: _robotic.Frame) -> int
        """

class JT:
    """Members:

      none

      hingeX

      hingeY

      hingeZ

      transX

      transY

      transZ

      transXY

      trans3

      transXYPhi

      transYPhi

      universal

      rigid

      quatBall

      phiTransXY

      XBall

      free

      generic

      tau"""
    __members__: ClassVar[dict] = ...  # read-only
    XBall: ClassVar[JT] = ...
    __entries: ClassVar[dict] = ...
    free: ClassVar[JT] = ...
    generic: ClassVar[JT] = ...
    hingeX: ClassVar[JT] = ...
    hingeY: ClassVar[JT] = ...
    hingeZ: ClassVar[JT] = ...
    none: ClassVar[JT] = ...
    phiTransXY: ClassVar[JT] = ...
    quatBall: ClassVar[JT] = ...
    rigid: ClassVar[JT] = ...
    tau: ClassVar[JT] = ...
    trans3: ClassVar[JT] = ...
    transX: ClassVar[JT] = ...
    transXY: ClassVar[JT] = ...
    transXYPhi: ClassVar[JT] = ...
    transY: ClassVar[JT] = ...
    transYPhi: ClassVar[JT] = ...
    transZ: ClassVar[JT] = ...
    universal: ClassVar[JT] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: _robotic.JT, value: int) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: _robotic.JT) -> int"""
    def __int__(self) -> int:
        """__int__(self: _robotic.JT) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    @property
    def name(self) -> str:
        """name(self: object) -> str

        name(self: object) -> str
        """
    @property
    def value(self) -> int:
        """(arg0: _robotic.JT) -> int"""

class KOMO:
    """A framework to define manipulation problems (IK, path optimization, sequential manipulation) as Nonlinear Mathematical Program (NLP). The actual NLP_Solver class is separate. (KOMO = k-order Markov Optimization) -- see https://marctoussaint.github.io/robotics-course/tutorials/1c-komo.html"""
    @overload
    def __init__(self) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: _robotic.KOMO) -> None

        [deprecated] please use the other constructor

        2. __init__(self: _robotic.KOMO, config: _robotic.Config, phases: float, slicesPerPhase: int, kOrder: int, enableCollisions: bool) -> None

        constructor
        * config: the configuration, which is copied once (for IK) or many times (for waypoints/paths) to be the optimization variable
        * phases: the number P of phases (which essentially defines the real-valued interval [0,P] over which objectives can be formulated)
        * slicesPerPhase: the discretizations per phase -> in total we have phases*slicesPerPhases configurations which form the path and over which we optimize
        * kOrder: the 'Markov-order', i.e., maximal tuple of configurations over which we formulate features (e.g. take finite differences)
        * enableCollisions: if True, KOMO runs a broadphase collision check (using libFCL) in each optimization step -- only then accumulative collision/penetration features will correctly evaluate to non-zero. But this is costly.
        """
    @overload
    def __init__(self, config: Config, phases: float, slicesPerPhase: int, kOrder: int, enableCollisions: bool) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: _robotic.KOMO) -> None

        [deprecated] please use the other constructor

        2. __init__(self: _robotic.KOMO, config: _robotic.Config, phases: float, slicesPerPhase: int, kOrder: int, enableCollisions: bool) -> None

        constructor
        * config: the configuration, which is copied once (for IK) or many times (for waypoints/paths) to be the optimization variable
        * phases: the number P of phases (which essentially defines the real-valued interval [0,P] over which objectives can be formulated)
        * slicesPerPhase: the discretizations per phase -> in total we have phases*slicesPerPhases configurations which form the path and over which we optimize
        * kOrder: the 'Markov-order', i.e., maximal tuple of configurations over which we formulate features (e.g. take finite differences)
        * enableCollisions: if True, KOMO runs a broadphase collision check (using libFCL) in each optimization step -- only then accumulative collision/penetration features will correctly evaluate to non-zero. But this is costly.
        """
    def addControlObjective(self, times: arr, order: int, scale: float = ..., target: arr = ..., deltaFromSlice: int = ..., deltaToSlice: int = ...) -> Objective:
        """addControlObjective(self: _robotic.KOMO, times: arr, order: int, scale: float = 1.0, target: arr = array(0.0078125), deltaFromSlice: int = 0, deltaToSlice: int = 0) -> Objective


        * times: (as for `addObjective`) the phase-interval in which this objective holds; [] means all times
        * order: Do we penalize the jointState directly (order=0: penalizing sqr distance to qHome, order=1: penalizing sqr distances between consecutive configurations (velocities), order=2: penalizing accelerations across 3 configurations)
        * scale: as usual, but modulated by a factor 'sqrt(delta t)' that somehow ensures total control costs in approximately independent of the choice of stepsPerPhase
        """
    def addFrameDof(self, name: str, parent: str, jointType: JT, stable: bool, originFrameName: str = ..., originFrame: Frame = ...) -> Frame:
        """addFrameDof(self: _robotic.KOMO, name: str, parent: str, jointType: _robotic.JT, stable: bool, originFrameName: str = None, originFrame: _robotic.Frame = None) -> _robotic.Frame

        complicated...
        """
    def addModeSwitch(self, times: arr, newMode: SY, frames: StringA, firstSwitch: bool = ...) -> None:
        """addModeSwitch(self: _robotic.KOMO, times: arr, newMode: _robotic.SY, frames: StringA, firstSwitch: bool = True) -> None"""
    def addObjective(self, times: arr, feature: FS, frames: StringA, type: ObjectiveType, scale: arr = ..., target: arr = ..., order: int = ...) -> None:
        """addObjective(self: _robotic.KOMO, times: arr, feature: _robotic.FS, frames: StringA, type: ObjectiveType, scale: arr = array(0.0078125), target: arr = array(0.0078125), order: int = -1) -> None

        central method to define objectives in the KOMO NLP:
        * times: the time intervals (subset of configurations in a path) over which this feature is active (irrelevant for IK)
        * feature: the feature symbol (see advanced `Feature` tutorial)
        * frames: the frames for which the feature is computed, given as list of frame names
        * type: whether this is a sum-of-squares (sos) cost, or eq or ineq constraint
        * scale: the matrix(!) by which the feature is multiplied
        * target: the offset which is substracted from the feature (before scaling)
        """
    def addQuaternionNorms(self, times: arr = ..., scale: float = ..., hard: bool = ...) -> None:
        """addQuaternionNorms(self: _robotic.KOMO, times: arr = array(0.0078125), scale: float = 3.0, hard: bool = True) -> None"""
    def addRigidSwitch(self, times: float, frames: StringA, noJumpStart: bool = ...) -> None:
        """addRigidSwitch(self: _robotic.KOMO, times: float, frames: StringA, noJumpStart: bool = True) -> None"""
    def addTimeOptimization(self) -> None:
        """addTimeOptimization(self: _robotic.KOMO) -> None"""
    def clearObjectives(self) -> None:
        """clearObjectives(self: _robotic.KOMO) -> None"""
    def getConfig(self) -> Config:
        """getConfig(self: _robotic.KOMO) -> _robotic.Config"""
    def getFeatureNames(self) -> StringA:
        """getFeatureNames(self: _robotic.KOMO) -> StringA

        (This is to be passed to the NLP_Solver when needed.) returns a long list of features (per time slice!)
        """
    def getForceInteractions(self) -> list:
        """getForceInteractions(self: _robotic.KOMO) -> list"""
    def getFrame(self, frameName: str, phaseTime: float) -> Frame:
        """getFrame(self: _robotic.KOMO, frameName: str, phaseTime: float) -> _robotic.Frame"""
    def getFrameState(self, arg0: int) -> arr:
        """getFrameState(self: _robotic.KOMO, arg0: int) -> arr"""
    def getPath(self, dofIndices: uintA = ...) -> arr:
        """getPath(self: _robotic.KOMO, dofIndices: uintA = array(0, dtype=uint32)) -> arr

        get path for selected dofs (default: all original config dofs)
        """
    def getPathFrames(self) -> arr:
        """getPathFrames(self: _robotic.KOMO) -> arr"""
    def getPathTau(self) -> arr:
        """getPathTau(self: _robotic.KOMO) -> arr"""
    def getPath_qAll(self) -> arrA:
        """getPath_qAll(self: _robotic.KOMO) -> arrA"""
    def getSubProblem(self, phase: int) -> tuple:
        """getSubProblem(self: _robotic.KOMO, phase: int) -> tuple

        return a tuple of (configuration, start q0, end q1) for given phase of this komo problem
        """
    def getT(self) -> int:
        """getT(self: _robotic.KOMO) -> int"""
    def get_viewer(self) -> ConfigurationViewer:
        """get_viewer(self: _robotic.KOMO) -> _robotic.ConfigurationViewer"""
    def info_objectiveErrorTraces(self) -> arr:
        """info_objectiveErrorTraces(self: _robotic.KOMO) -> arr

        return a TxO, for O objectives
        """
    def info_objectiveNames(self) -> StringA:
        """info_objectiveNames(self: _robotic.KOMO) -> StringA

        return a array of O strings, for O objectives
        """
    def info_sliceCollisions(self, *args, **kwargs):
        """info_sliceCollisions(self: _robotic.KOMO, t: int, belowMargin: float) -> rai::String

        return string info of collosions belowMargin in slice t
        """
    def info_sliceErrors(self, *args, **kwargs):
        """info_sliceErrors(self: _robotic.KOMO, t: int, errorTraces: arr) -> rai::String

        return string info of objectives and errors in slice t -- needs errorTraces as input
        """
    def initOrg(self) -> None:
        """initOrg(self: _robotic.KOMO) -> None"""
    def initPhaseWithDofsPath(self, t_phase: int, dofIDs: uintA, path: arr, autoResamplePath: bool = ...) -> None:
        """initPhaseWithDofsPath(self: _robotic.KOMO, t_phase: int, dofIDs: uintA, path: arr, autoResamplePath: bool = False) -> None"""
    def initRandom(self, verbose: int = ...) -> None:
        """initRandom(self: _robotic.KOMO, verbose: int = 0) -> None"""
    def initWithConstant(self, q: arr) -> None:
        """initWithConstant(self: _robotic.KOMO, q: arr) -> None"""
    def initWithPath(self, q: arr) -> None:
        """initWithPath(self: _robotic.KOMO, q: arr) -> None"""
    def initWithWaypoints(self, waypoints: arrA, waypointSlicesPerPhase: int = ..., interpolate: bool = ..., qHomeInterpolate: float = ..., verbose: int = ...) -> uintA:
        """initWithWaypoints(self: _robotic.KOMO, waypoints: arrA, waypointSlicesPerPhase: int = 1, interpolate: bool = False, qHomeInterpolate: float = 0.0, verbose: int = -1) -> uintA"""
    def nlp(self) -> NLP:
        """nlp(self: _robotic.KOMO) -> NLP

        return the problem NLP
        """
    def report(self, *args, **kwargs):
        """report(self: _robotic.KOMO, specs: bool = False, listObjectives: bool = True, plotOverTime: bool = False) -> rai::Graph

        returns a dict with full list of features, optionally also on problem specs and plotting costs/violations over time
        """
    def setConfig(self, config: Config, enableCollisions: bool) -> None:
        """setConfig(self: _robotic.KOMO, config: _robotic.Config, enableCollisions: bool) -> None

        [deprecated] please set directly in constructor
        """
    def setTiming(self, phases: float, slicesPerPhase: int, durationPerPhase: float, kOrder: int) -> None:
        """setTiming(self: _robotic.KOMO, phases: float, slicesPerPhase: int, durationPerPhase: float, kOrder: int) -> None

        [deprecated] please set directly in constructor
        """
    def set_viewer(self, arg0: ConfigurationViewer) -> None:
        """set_viewer(self: _robotic.KOMO, arg0: _robotic.ConfigurationViewer) -> None"""
    def updateRootObjects(self, config: Config) -> None:
        """updateRootObjects(self: _robotic.KOMO, config: _robotic.Config) -> None

        update root frames (without parents) within all KOMO configurations
        """
    def view(self, pause: bool = ..., txt: str = ...) -> int:
        """view(self: _robotic.KOMO, pause: bool = False, txt: str = None) -> int"""
    def view_close(self) -> None:
        """view_close(self: _robotic.KOMO) -> None"""
    def view_play(self, pause: bool = ..., txt: str = ..., delay: float = ..., saveVideoPath: str = ...) -> int:
        """view_play(self: _robotic.KOMO, pause: bool = False, txt: str = None, delay: float = 0.1, saveVideoPath: str = None) -> int"""
    def view_slice(self, t: int, pause: bool = ...) -> int:
        """view_slice(self: _robotic.KOMO, t: int, pause: bool = False) -> int"""

class KOMO_Objective:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

class LGP_Tool:
    """Tools to compute things (and solve) a Task-and-Motion Planning problem formulated as Logic-Geometric Program"""
    def __init__(self, arg0: Config, arg1: TAMP_Provider, arg2: Actions2KOMO_Translator) -> None:
        """__init__(self: _robotic.LGP_Tool, arg0: _robotic.Config, arg1: _robotic.TAMP_Provider, arg2: _robotic.Actions2KOMO_Translator) -> None

        initialization
        """
    def getSolvedKOMO(self) -> KOMO:
        """getSolvedKOMO(self: _robotic.LGP_Tool) -> _robotic.KOMO

        return the solved KOMO object (including its continuous solution) of current solution
        """
    def getSolvedPlan(self) -> StringAA:
        """getSolvedPlan(self: _robotic.LGP_Tool) -> StringAA

        return list of discrete decisions of current solution
        """
    def get_fullMotionProblem(self, initWithWaypoints: bool) -> KOMO:
        """get_fullMotionProblem(self: _robotic.LGP_Tool, initWithWaypoints: bool) -> _robotic.KOMO

        return the (unsolved) KOMO object corresponding to the full joint motion problem spanning all steps
        """
    def get_piecewiseMotionProblem(self, phase: int, fixEnd: bool) -> KOMO:
        """get_piecewiseMotionProblem(self: _robotic.LGP_Tool, phase: int, fixEnd: bool) -> _robotic.KOMO

        return the (unsolved) KOMO object corresponding to the k-th piece of the current solution
        """
    def solve(self, verbose: int = ...) -> None:
        """solve(self: _robotic.LGP_Tool, verbose: int = 1) -> None

        compute new solution
        """
    def solveFullMotion(self, verbose: int = ...) -> KOMO:
        """solveFullMotion(self: _robotic.LGP_Tool, verbose: int = 1) -> _robotic.KOMO

        solve full motion of current solution and return the (solved) KOMO object
        """
    def solvePiecewiseMotions(self, verbose: int = ...) -> arrA:
        """solvePiecewiseMotions(self: _robotic.LGP_Tool, verbose: int = 1) -> arrA

        solve full motion of current solution and return the (solved) KOMO object
        """
    def view_close(self) -> None:
        """view_close(self: _robotic.LGP_Tool) -> None"""
    def view_solved(self, pause: bool) -> int:
        """view_solved(self: _robotic.LGP_Tool, pause: bool) -> int

        view last computed solution
        """

class NLP:
    """A Nonlinear Mathematical Program (bindings to the c++ object - distinct from the python template nlp.NLP"""
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def checkHessian(self, x: arr, tolerance: float) -> bool:
        """checkHessian(self: _robotic.NLP, x: arr, tolerance: float) -> bool"""
    def checkJacobian(self, x: arr, tolerance: float, featureNames: StringA = ...) -> bool:
        """checkJacobian(self: _robotic.NLP, x: arr, tolerance: float, featureNames: StringA = []) -> bool"""
    def evaluate(self, arg0: arr) -> tuple[arr, arr]:
        """evaluate(self: _robotic.NLP, arg0: arr) -> tuple[arr, arr]

        query the NLP at a point $x$; returns the tuple $(phi,J)$, which is the feature vector and its Jacobian; features define cost terms, sum-of-square (sos) terms, inequalities, and equalities depending on 'getFeatureTypes'
        """
    def getBounds(self) -> arr:
        """getBounds(self: _robotic.NLP) -> arr

        returns the tuple $(b_{lo},b_{up})$, where both vectors are of same dimensionality of $x$ (or size zero, if there are no bounds)
        """
    def getDimension(self) -> int:
        """getDimension(self: _robotic.NLP) -> int

        return the dimensionality of $x$
        """
    def getFHessian(self, arg0: arr) -> arr:
        """getFHessian(self: _robotic.NLP, arg0: arr) -> arr

        returns Hessian of the sum of $f$-terms
        """
    def getFeatureTypes(self) -> list[ObjectiveType]:
        """getFeatureTypes(self: _robotic.NLP) -> list[ObjectiveType]

        features (entries of $phi$) can be of one of (ry.OT.f, ry.OT.sos, ry.OT.ineq, ry.OT.eq), which means (cost, sum-of-square, inequality, equality). The total cost $f(x)$ is the sum of all f-terms plus sum-of-squares of sos-terms.
        """
    def getInitializationSample(self) -> arr:
        """getInitializationSample(self: _robotic.NLP) -> arr

        returns a sample (e.g. uniform within bounds) to initialize an optimization -- not necessarily feasible
        """
    def report(self, arg0: int) -> str:
        """report(self: _robotic.NLP, arg0: int) -> str

        displays semantic information on the last query
        """

class NLP_Factory(NLP):
    def __init__(self) -> None:
        """__init__(self: _robotic.NLP_Factory) -> None"""
    def setBounds(self, arg0: arr, arg1: arr) -> None:
        """setBounds(self: _robotic.NLP_Factory, arg0: arr, arg1: arr) -> None"""
    def setDimension(self, arg0: int) -> None:
        """setDimension(self: _robotic.NLP_Factory, arg0: int) -> None"""
    def setEvalCallback(self, arg0: Callable[[arr], tuple[arr, arr]]) -> None:
        """setEvalCallback(self: _robotic.NLP_Factory, arg0: Callable[[arr], tuple[arr, arr]]) -> None"""
    def setFeatureTypes(self, arg0) -> None:
        """setFeatureTypes(self: _robotic.NLP_Factory, arg0: Array<T>) -> None"""
    def testCallingEvalCallback(self, arg0: arr) -> tuple[arr, arr]:
        """testCallingEvalCallback(self: _robotic.NLP_Factory, arg0: arr) -> tuple[arr, arr]"""

class NLP_Sampler:
    """An interface to an NLP sampler"""
    def __init__(self, problem: NLP) -> None:
        """__init__(self: _robotic.NLP_Sampler, problem: _robotic.NLP) -> None"""
    def sample(self) -> SolverReturn:
        """sample(self: _robotic.NLP_Sampler) -> SolverReturn"""
    def setOptions(self, eps: float = ..., useCentering: bool = ..., verbose: int = ..., seedMethod=..., seedCandidates: int = ..., penaltyMu: float = ..., downhillMethod=..., downhillMaxSteps: int = ..., slackStepAlpha: float = ..., slackMaxStep: float = ..., slackRegLambda: float = ..., ineqOverstep: float = ..., downhillNoiseMethod=..., downhillRejectMethod=..., downhillNoiseSigma: float = ..., interiorMethod=..., interiorBurnInSteps: int = ..., interiorSampleSteps: int = ..., interiorNoiseMethod=..., hitRunEqMargin: float = ..., interiorNoiseSigma: float = ..., langevinTauPrime: float = ...) -> NLP_Sampler:
        """setOptions(self: _robotic.NLP_Sampler, eps: float = 0.05, useCentering: bool = True, verbose: int = 1, seedMethod: rai::String = 'uni', seedCandidates: int = 10, penaltyMu: float = 1.0, downhillMethod: rai::String = 'GN', downhillMaxSteps: int = 50, slackStepAlpha: float = 1.0, slackMaxStep: float = 0.1, slackRegLambda: float = 0.01, ineqOverstep: float = -1, downhillNoiseMethod: rai::String = 'none', downhillRejectMethod: rai::String = 'none', downhillNoiseSigma: float = 0.1, interiorMethod: rai::String = 'HR', interiorBurnInSteps: int = 0, interiorSampleSteps: int = 1, interiorNoiseMethod: rai::String = 'iso', hitRunEqMargin: float = 0.1, interiorNoiseSigma: float = 0.5, langevinTauPrime: float = -1.0) -> _robotic.NLP_Sampler

        set solver options
        """

class NLP_Solver:
    """An interface to portfolio of solvers"""
    dual: arr
    x: arr
    @overload
    def __init__(self) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: _robotic.NLP_Solver) -> None

        2. __init__(self: _robotic.NLP_Solver, problem: _robotic.NLP, verbose: int = 0) -> None
        """
    @overload
    def __init__(self, problem: NLP, verbose: int = ...) -> None:
        """__init__(*args, **kwargs)
        Overloaded function.

        1. __init__(self: _robotic.NLP_Solver) -> None

        2. __init__(self: _robotic.NLP_Solver, problem: _robotic.NLP, verbose: int = 0) -> None
        """
    def getOptions(self) -> NLP_SolverOptions:
        """getOptions(self: _robotic.NLP_Solver) -> _robotic.NLP_SolverOptions"""
    def getProblem(self) -> NLP:
        """getProblem(self: _robotic.NLP_Solver) -> _robotic.NLP

        returns the NLP problem
        """
    def getTrace_J(self) -> arr:
        """getTrace_J(self: _robotic.NLP_Solver) -> arr"""
    def getTrace_costs(self) -> arr:
        """getTrace_costs(self: _robotic.NLP_Solver) -> arr

        returns steps-times-3 array with rows (f+sos-costs, ineq, eq)
        """
    def getTrace_phi(self) -> arr:
        """getTrace_phi(self: _robotic.NLP_Solver) -> arr"""
    def getTrace_x(self) -> arr:
        """getTrace_x(self: _robotic.NLP_Solver) -> arr

        returns steps-times-n array with queries points in each row
        """
    def reportLagrangeGradients(self, featureNames: StringA = ...) -> dict:
        """reportLagrangeGradients(self: _robotic.NLP_Solver, featureNames: StringA = []) -> dict

        return dictionary of Lagrange gradients per objective
        """
    def setInitialization(self, arg0: arr) -> NLP_Solver:
        """setInitialization(self: _robotic.NLP_Solver, arg0: arr) -> _robotic.NLP_Solver"""
    def setOptions(self, verbose: int = ..., stopTolerance: float = ..., stopFTolerance: float = ..., stopGTolerance: float = ..., stopEvals: int = ..., stopInners: int = ..., stopOuters: int = ..., stepMax: float = ..., damping: float = ..., stepInc: float = ..., stepDec: float = ..., wolfe: float = ..., muInit: float = ..., muInc: float = ..., muMax: float = ..., muLBInit: float = ..., muLBDec: float = ..., lambdaMax: float = ...) -> NLP_Solver:
        """setOptions(self: _robotic.NLP_Solver, verbose: int = 1, stopTolerance: float = 0.01, stopFTolerance: float = -1.0, stopGTolerance: float = -1.0, stopEvals: int = 1000, stopInners: int = 1000, stopOuters: int = 1000, stepMax: float = 0.2, damping: float = 1.0, stepInc: float = 1.5, stepDec: float = 0.5, wolfe: float = 0.01, muInit: float = 1.0, muInc: float = 5.0, muMax: float = 10000.0, muLBInit: float = 0.1, muLBDec: float = 0.2, lambdaMax: float = -1.0) -> _robotic.NLP_Solver

        set solver options
        """
    def setProblem(self, arg0: NLP) -> NLP_Solver:
        """setProblem(self: _robotic.NLP_Solver, arg0: _robotic.NLP) -> _robotic.NLP_Solver"""
    def setPyProblem(self, arg0: object) -> None:
        """setPyProblem(self: _robotic.NLP_Solver, arg0: object) -> None"""
    def setSolver(self, arg0) -> NLP_Solver:
        """setSolver(self: _robotic.NLP_Solver, arg0: rai::OptMethod) -> _robotic.NLP_Solver"""
    def setTracing(self, arg0: bool, arg1: bool, arg2: bool, arg3: bool) -> NLP_Solver:
        """setTracing(self: _robotic.NLP_Solver, arg0: bool, arg1: bool, arg2: bool, arg3: bool) -> _robotic.NLP_Solver"""
    def solve(self, resampleInitialization: int = ..., verbose: int = ...) -> SolverReturn:
        """solve(self: _robotic.NLP_Solver, resampleInitialization: int = -1, verbose: int = -100) -> SolverReturn

        resampleInitialization=-1 means: only when not already solved
        """

class NLP_SolverOptions:
    """solver options"""
    def __init__(self) -> None:
        """__init__(self: _robotic.NLP_SolverOptions) -> None"""
    def dict(self) -> dict:
        """dict(self: _robotic.NLP_SolverOptions) -> dict"""
    def set_damping(self, arg0: float) -> NLP_SolverOptions:
        """set_damping(self: _robotic.NLP_SolverOptions, arg0: float) -> _robotic.NLP_SolverOptions"""
    def set_lambdaMax(self, arg0: float) -> NLP_SolverOptions:
        """set_lambdaMax(self: _robotic.NLP_SolverOptions, arg0: float) -> _robotic.NLP_SolverOptions"""
    def set_muInc(self, arg0: float) -> NLP_SolverOptions:
        """set_muInc(self: _robotic.NLP_SolverOptions, arg0: float) -> _robotic.NLP_SolverOptions"""
    def set_muInit(self, arg0: float) -> NLP_SolverOptions:
        """set_muInit(self: _robotic.NLP_SolverOptions, arg0: float) -> _robotic.NLP_SolverOptions"""
    def set_muLBDec(self, arg0: float) -> NLP_SolverOptions:
        """set_muLBDec(self: _robotic.NLP_SolverOptions, arg0: float) -> _robotic.NLP_SolverOptions"""
    def set_muLBInit(self, arg0: float) -> NLP_SolverOptions:
        """set_muLBInit(self: _robotic.NLP_SolverOptions, arg0: float) -> _robotic.NLP_SolverOptions"""
    def set_muMax(self, arg0: float) -> NLP_SolverOptions:
        """set_muMax(self: _robotic.NLP_SolverOptions, arg0: float) -> _robotic.NLP_SolverOptions"""
    def set_stepDec(self, arg0: float) -> NLP_SolverOptions:
        """set_stepDec(self: _robotic.NLP_SolverOptions, arg0: float) -> _robotic.NLP_SolverOptions"""
    def set_stepInc(self, arg0: float) -> NLP_SolverOptions:
        """set_stepInc(self: _robotic.NLP_SolverOptions, arg0: float) -> _robotic.NLP_SolverOptions"""
    def set_stepMax(self, arg0: float) -> NLP_SolverOptions:
        """set_stepMax(self: _robotic.NLP_SolverOptions, arg0: float) -> _robotic.NLP_SolverOptions"""
    def set_stopEvals(self, arg0: int) -> NLP_SolverOptions:
        """set_stopEvals(self: _robotic.NLP_SolverOptions, arg0: int) -> _robotic.NLP_SolverOptions"""
    def set_stopFTolerance(self, arg0: float) -> NLP_SolverOptions:
        """set_stopFTolerance(self: _robotic.NLP_SolverOptions, arg0: float) -> _robotic.NLP_SolverOptions"""
    def set_stopGTolerance(self, arg0: float) -> NLP_SolverOptions:
        """set_stopGTolerance(self: _robotic.NLP_SolverOptions, arg0: float) -> _robotic.NLP_SolverOptions"""
    def set_stopInners(self, arg0: int) -> NLP_SolverOptions:
        """set_stopInners(self: _robotic.NLP_SolverOptions, arg0: int) -> _robotic.NLP_SolverOptions"""
    def set_stopOuters(self, arg0: int) -> NLP_SolverOptions:
        """set_stopOuters(self: _robotic.NLP_SolverOptions, arg0: int) -> _robotic.NLP_SolverOptions"""
    def set_stopTolerance(self, arg0: float) -> NLP_SolverOptions:
        """set_stopTolerance(self: _robotic.NLP_SolverOptions, arg0: float) -> _robotic.NLP_SolverOptions"""
    def set_verbose(self, arg0: int) -> NLP_SolverOptions:
        """set_verbose(self: _robotic.NLP_SolverOptions, arg0: int) -> _robotic.NLP_SolverOptions"""
    def set_wolfe(self, arg0: float) -> NLP_SolverOptions:
        """set_wolfe(self: _robotic.NLP_SolverOptions, arg0: float) -> _robotic.NLP_SolverOptions"""

class OT:
    """Members:

      f

      sos

      ineq

      eq

      ineqB

      ineqP

      none"""
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    eq: ClassVar[OT] = ...
    f: ClassVar[OT] = ...
    ineq: ClassVar[OT] = ...
    ineqB: ClassVar[OT] = ...
    ineqP: ClassVar[OT] = ...
    none: ClassVar[OT] = ...
    sos: ClassVar[OT] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: _robotic.OT, value: int) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: _robotic.OT) -> int"""
    def __int__(self) -> int:
        """__int__(self: _robotic.OT) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    @property
    def name(self) -> str:
        """name(self: object) -> str

        name(self: object) -> str
        """
    @property
    def value(self) -> int:
        """(arg0: _robotic.OT) -> int"""

class OptBench_Skeleton_Handover:
    def __init__(self, arg0: ArgWord) -> None:
        """__init__(self: _robotic.OptBench_Skeleton_Handover, arg0: _robotic.ArgWord) -> None"""
    def get(self) -> NLP:
        """get(self: _robotic.OptBench_Skeleton_Handover) -> _robotic.NLP"""

class OptBench_Skeleton_Pick:
    def __init__(self, arg0: ArgWord) -> None:
        """__init__(self: _robotic.OptBench_Skeleton_Pick, arg0: _robotic.ArgWord) -> None"""
    def get(self) -> NLP:
        """get(self: _robotic.OptBench_Skeleton_Pick) -> _robotic.NLP"""

class OptBench_Skeleton_StackAndBalance:
    def __init__(self, arg0: ArgWord) -> None:
        """__init__(self: _robotic.OptBench_Skeleton_StackAndBalance, arg0: _robotic.ArgWord) -> None"""
    def get(self) -> NLP:
        """get(self: _robotic.OptBench_Skeleton_StackAndBalance) -> _robotic.NLP"""

class OptBenchmark_InvKin_Endeff:
    def __init__(self, arg0: str, arg1: bool) -> None:
        """__init__(self: _robotic.OptBenchmark_InvKin_Endeff, arg0: str, arg1: bool) -> None"""
    def get(self) -> NLP:
        """get(self: _robotic.OptBenchmark_InvKin_Endeff) -> _robotic.NLP"""

class OptMethod:
    """Members:

      none

      gradientDescent

      rprop

      LBFGS

      newton

      augmentedLag

      squaredPenalty

      logBarrier

      singleSquaredPenalty

      slackGN

      NLopt

      Ipopt

      Ceres"""
    __members__: ClassVar[dict] = ...  # read-only
    Ceres: ClassVar[OptMethod] = ...
    Ipopt: ClassVar[OptMethod] = ...
    LBFGS: ClassVar[OptMethod] = ...
    NLopt: ClassVar[OptMethod] = ...
    __entries: ClassVar[dict] = ...
    augmentedLag: ClassVar[OptMethod] = ...
    gradientDescent: ClassVar[OptMethod] = ...
    logBarrier: ClassVar[OptMethod] = ...
    newton: ClassVar[OptMethod] = ...
    none: ClassVar[OptMethod] = ...
    rprop: ClassVar[OptMethod] = ...
    singleSquaredPenalty: ClassVar[OptMethod] = ...
    slackGN: ClassVar[OptMethod] = ...
    squaredPenalty: ClassVar[OptMethod] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: _robotic.OptMethod, value: int) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: _robotic.OptMethod) -> int"""
    def __int__(self) -> int:
        """__int__(self: _robotic.OptMethod) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    @property
    def name(self) -> str:
        """name(self: object) -> str

        name(self: object) -> str
        """
    @property
    def value(self) -> int:
        """(arg0: _robotic.OptMethod) -> int"""

class Quaternion:
    def __init__(self) -> None:
        """__init__(self: _robotic.Quaternion) -> None

        non-initialized
        """
    def append(self, q: Quaternion) -> None:
        """append(self: _robotic.Quaternion, q: _robotic.Quaternion) -> None"""
    def applyOnPointArray(self, pts: arr) -> None:
        """applyOnPointArray(self: _robotic.Quaternion, pts: arr) -> None"""
    def asArr(self) -> arr:
        """asArr(self: _robotic.Quaternion) -> arr"""
    def flipSign(self) -> None:
        """flipSign(self: _robotic.Quaternion) -> None"""
    def getJacobian(self) -> arr:
        """getJacobian(self: _robotic.Quaternion) -> arr"""
    def getLog(self) -> Vector:
        """getLog(self: _robotic.Quaternion) -> Vector"""
    def getMatrix(self) -> arr:
        """getMatrix(self: _robotic.Quaternion) -> arr"""
    def getRad(self) -> float:
        """getRad(self: _robotic.Quaternion) -> float"""
    def getRollPitchYaw(self) -> arr:
        """getRollPitchYaw(self: _robotic.Quaternion) -> arr"""
    def invert(self) -> None:
        """invert(self: _robotic.Quaternion) -> None"""
    def multiply(self, f: float) -> None:
        """multiply(self: _robotic.Quaternion, f: float) -> None"""
    def normalize(self) -> None:
        """normalize(self: _robotic.Quaternion) -> None"""
    def set(self, q: arr) -> Quaternion:
        """set(self: _robotic.Quaternion, q: arr) -> _robotic.Quaternion"""
    def setDiff(self, _from: Vector, to: Vector) -> Quaternion:
        """setDiff(self: _robotic.Quaternion, from: Vector, to: Vector) -> _robotic.Quaternion"""
    def setEuler(self, euler_zxz: Vector) -> Quaternion:
        """setEuler(self: _robotic.Quaternion, euler_zxz: Vector) -> _robotic.Quaternion"""
    def setExp(self, vector_w: Vector) -> Quaternion:
        """setExp(self: _robotic.Quaternion, vector_w: Vector) -> _robotic.Quaternion"""
    def setInterpolateEmbedded(self, t: float, _from: Quaternion, to: Quaternion) -> Quaternion:
        """setInterpolateEmbedded(self: _robotic.Quaternion, t: float, from: _robotic.Quaternion, to: _robotic.Quaternion) -> _robotic.Quaternion"""
    def setInterpolateProper(self, t: float, _from: Quaternion, to: Quaternion) -> Quaternion:
        """setInterpolateProper(self: _robotic.Quaternion, t: float, from: _robotic.Quaternion, to: _robotic.Quaternion) -> _robotic.Quaternion"""
    def setMatrix(self, R: arr) -> Quaternion:
        """setMatrix(self: _robotic.Quaternion, R: arr) -> _robotic.Quaternion"""
    def setRad(self, radians: float, axis: Vector) -> Quaternion:
        """setRad(self: _robotic.Quaternion, radians: float, axis: Vector) -> _robotic.Quaternion"""
    def setRandom(self) -> Quaternion:
        """setRandom(self: _robotic.Quaternion) -> _robotic.Quaternion"""
    def setRollPitchYaw(self, roll_pitch_yaw: Vector) -> Quaternion:
        """setRollPitchYaw(self: _robotic.Quaternion, roll_pitch_yaw: Vector) -> _robotic.Quaternion"""
    def setZero(self) -> Quaternion:
        """setZero(self: _robotic.Quaternion) -> _robotic.Quaternion"""
    def sqrNorm(self) -> float:
        """sqrNorm(self: _robotic.Quaternion) -> float"""
    def __mul__(self, arg0: Quaternion) -> Quaternion:
        """__mul__(self: _robotic.Quaternion, arg0: _robotic.Quaternion) -> _robotic.Quaternion

        concatenation (quaternion multiplication) of two transforms
        """

class RRT_PathFinder:
    """todo doc"""
    def __init__(self) -> None:
        """__init__(self: _robotic.RRT_PathFinder) -> None"""
    def get_resampledPath(self, arg0: int) -> arr:
        """get_resampledPath(self: _robotic.RRT_PathFinder, arg0: int) -> arr"""
    def setExplicitCollisionPairs(self, collisionPairs: StringA) -> None:
        """setExplicitCollisionPairs(self: _robotic.RRT_PathFinder, collisionPairs: StringA) -> None

        only after setProblem
        """
    def setProblem(self, Configuration: Config) -> None:
        """setProblem(self: _robotic.RRT_PathFinder, Configuration: _robotic.Config) -> None"""
    def setStartGoal(self, starts: arr, goals: arr) -> None:
        """setStartGoal(self: _robotic.RRT_PathFinder, starts: arr, goals: arr) -> None"""
    def solve(self) -> SolverReturn:
        """solve(self: _robotic.RRT_PathFinder) -> SolverReturn"""

class ST:
    """Members:

      none

      box

      sphere

      capsule

      mesh

      cylinder

      marker

      pointCloud

      ssCvx

      ssBox

      ssCylinder

      ssBoxElip

      quad

      camera

      sdf"""
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    box: ClassVar[ST] = ...
    camera: ClassVar[ST] = ...
    capsule: ClassVar[ST] = ...
    cylinder: ClassVar[ST] = ...
    marker: ClassVar[ST] = ...
    mesh: ClassVar[ST] = ...
    none: ClassVar[ST] = ...
    pointCloud: ClassVar[ST] = ...
    quad: ClassVar[ST] = ...
    sdf: ClassVar[ST] = ...
    sphere: ClassVar[ST] = ...
    ssBox: ClassVar[ST] = ...
    ssBoxElip: ClassVar[ST] = ...
    ssCvx: ClassVar[ST] = ...
    ssCylinder: ClassVar[ST] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: _robotic.ST, value: int) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: _robotic.ST) -> int"""
    def __int__(self) -> int:
        """__int__(self: _robotic.ST) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    @property
    def name(self) -> str:
        """name(self: object) -> str

        name(self: object) -> str
        """
    @property
    def value(self) -> int:
        """(arg0: _robotic.ST) -> int"""

class SY:
    """Members:

      touch

      above

      inside

      oppose

      restingOn

      poseEq

      positionEq

      stableRelPose

      stablePose

      stable

      stableOn

      dynamic

      dynamicOn

      dynamicTrans

      quasiStatic

      quasiStaticOn

      downUp

      stableZero

      contact

      contactStick

      contactComplementary

      bounce

      push

      magic

      magicTrans

      pushAndPlace

      topBoxGrasp

      topBoxPlace

      dampMotion

      identical

      alignByInt

      makeFree

      forceBalance

      relPosY

      touchBoxNormalX

      touchBoxNormalY

      touchBoxNormalZ

      boxGraspX

      boxGraspY

      boxGraspZ

      lift

      stableYPhi

      stableOnX

      stableOnY

      end"""
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    above: ClassVar[SY] = ...
    alignByInt: ClassVar[SY] = ...
    bounce: ClassVar[SY] = ...
    boxGraspX: ClassVar[SY] = ...
    boxGraspY: ClassVar[SY] = ...
    boxGraspZ: ClassVar[SY] = ...
    contact: ClassVar[SY] = ...
    contactComplementary: ClassVar[SY] = ...
    contactStick: ClassVar[SY] = ...
    dampMotion: ClassVar[SY] = ...
    downUp: ClassVar[SY] = ...
    dynamic: ClassVar[SY] = ...
    dynamicOn: ClassVar[SY] = ...
    dynamicTrans: ClassVar[SY] = ...
    end: ClassVar[SY] = ...
    forceBalance: ClassVar[SY] = ...
    identical: ClassVar[SY] = ...
    inside: ClassVar[SY] = ...
    lift: ClassVar[SY] = ...
    magic: ClassVar[SY] = ...
    magicTrans: ClassVar[SY] = ...
    makeFree: ClassVar[SY] = ...
    oppose: ClassVar[SY] = ...
    poseEq: ClassVar[SY] = ...
    positionEq: ClassVar[SY] = ...
    push: ClassVar[SY] = ...
    pushAndPlace: ClassVar[SY] = ...
    quasiStatic: ClassVar[SY] = ...
    quasiStaticOn: ClassVar[SY] = ...
    relPosY: ClassVar[SY] = ...
    restingOn: ClassVar[SY] = ...
    stable: ClassVar[SY] = ...
    stableOn: ClassVar[SY] = ...
    stableOnX: ClassVar[SY] = ...
    stableOnY: ClassVar[SY] = ...
    stablePose: ClassVar[SY] = ...
    stableRelPose: ClassVar[SY] = ...
    stableYPhi: ClassVar[SY] = ...
    stableZero: ClassVar[SY] = ...
    topBoxGrasp: ClassVar[SY] = ...
    topBoxPlace: ClassVar[SY] = ...
    touch: ClassVar[SY] = ...
    touchBoxNormalX: ClassVar[SY] = ...
    touchBoxNormalY: ClassVar[SY] = ...
    touchBoxNormalZ: ClassVar[SY] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: _robotic.SY, value: int) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: _robotic.SY) -> int"""
    def __int__(self) -> int:
        """__int__(self: _robotic.SY) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    @property
    def name(self) -> str:
        """name(self: object) -> str

        name(self: object) -> str
        """
    @property
    def value(self) -> int:
        """(arg0: _robotic.SY) -> int"""

class Simulation:
    """A direct simulation interface to physics engines (Nvidia PhysX, Bullet) -- see https://marctoussaint.github.io/robotics-course/tutorials/simulation.html"""
    def __init__(self, C: Config, engine: SimulationEngine, verbose: int = ...) -> None:
        """__init__(self: _robotic.Simulation, C: _robotic.Config, engine: _robotic.SimulationEngine, verbose: int = 2) -> None

        create a Simulation that is associated/attached to the given configuration
        """
    def addSensor(self, *args, **kwargs):
        """addSensor(self: _robotic.Simulation, sensorName: str, width: int = 640, height: int = 360, focalLength: float = -1.0, orthoAbsHeight: float = -1.0, zRange: arr = []) -> rai::CameraView::Sensor"""
    def attach(self, _from: Frame, to: Frame) -> None:
        """attach(self: _robotic.Simulation, from: _robotic.Frame, to: _robotic.Frame) -> None"""
    def depthData2pointCloud(self, arg0: numpy.ndarray[numpy.float32], arg1: list[float]) -> numpy.ndarray[numpy.float64]:
        """depthData2pointCloud(self: _robotic.Simulation, arg0: numpy.ndarray[numpy.float32], arg1: list[float]) -> numpy.ndarray[numpy.float64]"""
    def detach(self, _from: Frame, to: Frame) -> None:
        """detach(self: _robotic.Simulation, from: _robotic.Frame, to: _robotic.Frame) -> None"""
    def getGripperWidth(self, gripperFrameName: str) -> float:
        """getGripperWidth(self: _robotic.Simulation, gripperFrameName: str) -> float"""
    def getImageAndDepth(self) -> tuple:
        """getImageAndDepth(self: _robotic.Simulation) -> tuple"""
    def getScreenshot(self, *args, **kwargs):
        """getScreenshot(self: _robotic.Simulation) -> Array<T>"""
    def getState(self) -> tuple:
        """getState(self: _robotic.Simulation) -> tuple

        returns a 4-tuple or frame state, joint state, frame velocities (linear & angular), joint velocities
        """
    def getTimeToSplineEnd(self) -> float:
        """getTimeToSplineEnd(self: _robotic.Simulation) -> float"""
    def get_frameVelocities(self) -> arr:
        """get_frameVelocities(self: _robotic.Simulation) -> arr"""
    def get_q(self) -> arr:
        """get_q(self: _robotic.Simulation) -> arr"""
    def get_qDot(self) -> arr:
        """get_qDot(self: _robotic.Simulation) -> arr"""
    def gripperIsDone(self, gripperFrameName: str) -> bool:
        """gripperIsDone(self: _robotic.Simulation, gripperFrameName: str) -> bool"""
    def moveGripper(self, gripperFrameName: str, width: float, speed: float = ...) -> None:
        """moveGripper(self: _robotic.Simulation, gripperFrameName: str, width: float, speed: float = 0.3) -> None"""
    def pushConfigurationToSimulator(self, frameVelocities: arr = ..., jointVelocities: arr = ...) -> None:
        """pushConfigurationToSimulator(self: _robotic.Simulation, frameVelocities: arr = array(0.0078125), jointVelocities: arr = array(0.0078125)) -> None

        set the simulator to the full (frame) state of the configuration
        """
    def resetSplineRef(self) -> None:
        """resetSplineRef(self: _robotic.Simulation) -> None

        reset the spline reference, i.e., clear the current spline buffer and initialize it to constant spline at current position (to which setSplineRef can append)
        """
    def resetTime(self) -> None:
        """resetTime(self: _robotic.Simulation) -> None"""
    def selectSensor(self, *args, **kwargs):
        """selectSensor(self: _robotic.Simulation, sensorName: str) -> rai::CameraView::Sensor"""
    def setSplineRef(self, path: arr, times: arr, append: bool = ...) -> None:
        """setSplineRef(self: _robotic.Simulation, path: arr, times: arr, append: bool = True) -> None

        set the spline reference to generate motion
        * path: single configuration, or sequence of spline control points
        * times: array with single total duration, or time for each control point (times.N==path.d0)
        * append: append (with zero-velocity at append), or smoothly overwrite
        """
    def setState(self, frameState: arr, jointState: arr = ..., frameVelocities: arr = ..., jointVelocities: arr = ...) -> None:
        """setState(self: _robotic.Simulation, frameState: arr, jointState: arr = array(0.0078125), frameVelocities: arr = array(0.0078125), jointVelocities: arr = array(0.0078125)) -> None"""
    def step(self, u_control: arr, tau: float = ..., u_mode: ControlMode = ...) -> None:
        """step(self: _robotic.Simulation, u_control: arr, tau: float = 0.01, u_mode: _robotic.ControlMode = <ControlMode.velocity: 2>) -> None"""

class SimulationEngine:
    """Members:

      physx

      bullet

      kinematic"""
    __members__: ClassVar[dict] = ...  # read-only
    __entries: ClassVar[dict] = ...
    bullet: ClassVar[SimulationEngine] = ...
    kinematic: ClassVar[SimulationEngine] = ...
    physx: ClassVar[SimulationEngine] = ...
    def __init__(self, value: int) -> None:
        """__init__(self: _robotic.SimulationEngine, value: int) -> None"""
    def __eq__(self, other: object) -> bool:
        """__eq__(self: object, other: object) -> bool"""
    def __hash__(self) -> int:
        """__hash__(self: object) -> int"""
    def __index__(self) -> int:
        """__index__(self: _robotic.SimulationEngine) -> int"""
    def __int__(self) -> int:
        """__int__(self: _robotic.SimulationEngine) -> int"""
    def __ne__(self, other: object) -> bool:
        """__ne__(self: object, other: object) -> bool"""
    @property
    def name(self) -> str:
        """name(self: object) -> str

        name(self: object) -> str
        """
    @property
    def value(self) -> int:
        """(arg0: _robotic.SimulationEngine) -> int"""

class Skeleton:
    def __init__(self) -> None:
        """__init__(self: _robotic.Skeleton) -> None"""
    def add(self, arg0: list) -> None:
        """add(self: _robotic.Skeleton, arg0: list) -> None"""
    def addEntry(self, timeInterval: arr, symbol: SY, frames: StringA) -> None:
        """addEntry(self: _robotic.Skeleton, timeInterval: arr, symbol: _robotic.SY, frames: StringA) -> None"""
    def addExplicitCollisions(self, collisions: StringA) -> None:
        """addExplicitCollisions(self: _robotic.Skeleton, collisions: StringA) -> None"""
    def addLiftPriors(self, lift: StringA) -> None:
        """addLiftPriors(self: _robotic.Skeleton, lift: StringA) -> None"""
    def getKomo_finalSlice(self, Configuration: Config, lenScale: float, homingScale: float, collScale: float) -> KOMO:
        """getKomo_finalSlice(self: _robotic.Skeleton, Configuration: _robotic.Config, lenScale: float, homingScale: float, collScale: float) -> _robotic.KOMO"""
    def getKomo_path(self, Configuration: Config, stepsPerPhase: int, accScale: float, lenScale: float, homingScale: float, collScale: float) -> KOMO:
        """getKomo_path(self: _robotic.Skeleton, Configuration: _robotic.Config, stepsPerPhase: int, accScale: float, lenScale: float, homingScale: float, collScale: float) -> _robotic.KOMO"""
    def getKomo_waypoints(self, Configuration: Config, lenScale: float, homingScale: float, collScale: float) -> KOMO:
        """getKomo_waypoints(self: _robotic.Skeleton, Configuration: _robotic.Config, lenScale: float, homingScale: float, collScale: float) -> _robotic.KOMO"""
    def getMaxPhase(self) -> float:
        """getMaxPhase(self: _robotic.Skeleton) -> float"""
    def getTwoWaypointProblem(self, t2: int, komoWays: KOMO) -> tuple:
        """getTwoWaypointProblem(self: _robotic.Skeleton, t2: int, komoWays: _robotic.KOMO) -> tuple"""
    def useBroadCollisions(self, enable: bool = ...) -> None:
        """useBroadCollisions(self: _robotic.Skeleton, enable: bool = True) -> None"""

class SolverReturn:
    """return of nlp solve call"""
    done: bool
    eq: float
    evals: int
    f: float
    feasible: bool
    ineq: float
    sos: float
    time: float
    x: arr
    def __init__(self) -> None:
        """__init__(self: _robotic.SolverReturn) -> None"""
    def dict(self) -> dict:
        """dict(self: _robotic.SolverReturn) -> dict"""

class TAMP_Provider:
    """TAMP_Provider"""
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""

def compiled() -> str:
    """compiled() -> str

    return a compile date+time version string
    """
def default_Actions2KOMO_Translator() -> Actions2KOMO_Translator:
    """default_Actions2KOMO_Translator() -> _robotic.Actions2KOMO_Translator"""
def default_TAMP_Provider(C: Config, lgp_config_file: str) -> TAMP_Provider:
    """default_TAMP_Provider(C: _robotic.Config, lgp_config_file: str) -> _robotic.TAMP_Provider"""
def depthImage2PointCloud(depth: numpy.ndarray[numpy.float32], fxycxy: arr) -> arr:
    """depthImage2PointCloud(depth: numpy.ndarray[numpy.float32], fxycxy: arr) -> arr

    return the point cloud from the depth image
    """
def params_add(pythondictionaryorparamstoadd: dict) -> None:
    """params_add(python dictionary or params to add: dict) -> None

    add/set parameters
    """
def params_clear() -> None:
    """params_clear() -> None

    clear all parameters
    """
def params_file(filename: str) -> None:
    """params_file(filename: str) -> None

    add parameters from a file
    """
def params_print() -> None:
    """params_print() -> None

    print the parameters
    """
def raiPath(*args, **kwargs):
    """raiPath(arg0: str) -> rai::String

    get a path relative to rai base path
    """
def setRaiPath(arg0: str) -> None:
    """setRaiPath(arg0: str) -> None

    redefine the rai (or rai-robotModels) path
    """
