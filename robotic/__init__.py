import os

from robotic._robotic import (
    FS,
    JT,
    KOMO,
    OT,
    ST,
    CameraView,
    Config,
    ControlMode,
    NLP_Solver,
    Simulation,
    SimulationEngine,
    compiled,
    depthImage2PointCloud,
    raiPath,
    setRaiPath,
)  # noqa: F401
from robotic.manipulation import Manipulation  # noqa: F401

setRaiPath(os.path.abspath(os.path.dirname(__file__)) + "/../lib/rai-robotModels")
