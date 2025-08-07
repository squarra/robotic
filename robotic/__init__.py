import os

from robotic._robotic import Config, compiled, raiPath, setRaiPath  # noqa: F401

setRaiPath(os.path.abspath(os.path.dirname(__file__)) + "/../lib/rai-robotModels")
