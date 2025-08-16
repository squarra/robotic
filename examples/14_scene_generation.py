from robotic._robotic import ST
from robotic.scenario import PandaScenario

config = PandaScenario()
num_scenes = 10
for i in range(num_scenes):
    config.delete_man_frames()
    config.add_boxes_to_scene((2, 10), seed=i)
    for box_id, man_frame in enumerate(config.man_frames):
        config.addFrame(f"mark{box_id}", man_frame).setShape(ST.marker, [0.1])
    config.view(pause=True, message=f"Scene {i + 1}/{num_scenes}")
