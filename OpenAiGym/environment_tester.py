import gym
import Deep_Robot
import math
from math import cos, sin, pi

env = gym.make('Deep_Robot:DiscreteDeepRobot-v0')
env.reset()
for t in range(1000):
    env.step((-0.5/20*sin(t/20+1),-0.5/20*sin(t/20))) # take a random action
env.render()
env.close()
