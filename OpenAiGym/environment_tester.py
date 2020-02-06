import gym
import Deep_Robot
env = gym.make('Deep_Robot:DeepRobot-v0')
env.reset()
for _ in range(1000):
    env.step(env.action_space.sample()) # take a random action
env.render()
env.close()
