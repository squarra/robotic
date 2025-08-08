import os

from robotic._robotic import CameraView, Config, compiled, depthImage2PointCloud, raiPath, setRaiPath  # noqa: F401

setRaiPath(os.path.abspath(os.path.dirname(__file__)) + "/../lib/rai-robotModels")
