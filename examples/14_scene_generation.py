from robotic.scenario import PandaScenario

config = PandaScenario()
for i in range(10):
    config.delete_man_frames()
    config.create_random_scene((2, 10), seed=i)
    config.view(pause=True)
