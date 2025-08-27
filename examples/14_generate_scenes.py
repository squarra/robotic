from robotic.scenario import PandaScenario

config = PandaScenario()
num_scenes = 10
for i in range(num_scenes):
    config.delete_man_frames()
    config.add_boxes_to_scene((2, 10), seed=i)
    config.add_markers()
    config.view(pause=True)
