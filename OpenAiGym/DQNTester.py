import gym
import Deep_Robot
import math
from math import cos, sin, pi

env = gym.make('Deep_Robot:DiscreteDeepRobot-v0')
observation = env.reset()
for i in range(1000):
  print(i)
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()
env.render()
env.close()