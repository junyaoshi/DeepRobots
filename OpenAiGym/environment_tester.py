import gym
import ContinuousDeepRobot-v0
env = gym.make('ContinuousDeepRobot-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()
