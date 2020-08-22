from gym.envs.registration import register

register(
    id='DeepRobot-v0',
    entry_point='Deep_Robot.envs:DeepRobotEnv',
)

register(
    id='DiscreteDeepRobot-v0',
    entry_point='Deep_Robot.envs:DiscreteDeepRobotEnv',
)

