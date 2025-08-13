import matplotlib
import matplotlib.pyplot as plt

from robotic import ST, CameraView
from robotic._robotic import Config, raiPath
from robotic.helpers import compute_look_at_matrix

matplotlib.use("GTK3Agg")

# Create a camera frame and set its initial position and orientation
config = Config()
config.addFile(raiPath("scenarios/pandaSingle.g"))
camera = config.addFrame("camera", "table").setRelativePosition([0.0, 2.0, 0.5]).setRelativeQuaternion([0.0, 0.0, -0.707, 0.707])
camera.setAttributes({"focalLength": 1.0, "width": 500.0, "height": 500.0, "zRange": [0.01, 10.0]})
camera.setShape(ST.camera, [])  # optional: tag as camera so it can be auto‑found in the scene
config.get_viewer().setCamera(camera)
config.view(pause=True, message="Initial camera setup")

# Reposition the camera and orient it to face the table's origin (0, 0, 0)
camera.setRelativePosition([1, 2, 1])
camera.setRotationMatrix(compute_look_at_matrix(camera.getPosition(), config.getFrame("table").getPosition()))
config.get_viewer().setCamera(camera)
config.view(pause=True, message="Camera looking at table center")

# Add a marker shape to visualize the camera position
camera.setShape(ST.marker, [0.5])
config.get_viewer().setCamera(None)
config.view(pause=True, message="Camera position marker")
config.view_close()

# Capture RGB and depth images from the current camera view
cam_view = CameraView(config)
cam_view.setCamera(camera)
rgb, depth = cam_view.computeImageAndDepth(config)
fig = plt.figure()
fig.add_subplot(1, 2, 1)
plt.imshow(rgb)
fig.add_subplot(1, 2, 2)
plt.imshow(depth)
plt.show()
