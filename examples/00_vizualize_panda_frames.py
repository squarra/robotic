from robotic.scenario import PandaScenario

config = PandaScenario()

for frame in sorted(config.getFrames(), key=lambda x: x.name):
    initial_color = frame.getMeshColors()
    frame.setColor([1, 0, 0])
    config.view(pause=True, message=frame.name)
    frame.setColor([1, 1, 1])
