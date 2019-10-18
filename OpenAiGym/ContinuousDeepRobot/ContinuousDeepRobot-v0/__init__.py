from gym.envs.registration import register

register(
    id='ContinuousDeepRobot-v0',
    entry_point='ContinuousDeepRobot-v0.envs:FooEnv',
)

