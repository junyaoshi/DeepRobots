import gym
from gym import spaces
from Robots.IdealFluidSwimmer_v1 import IdealFluidSwimmer
import numpy as np
from math import pi
from utils.learning_helper import forward_reward_function


class IdealFluidSwimmerEnv(gym.Env):
  def __init__(self):
    super(IdealFluidSwimmerEnv, self).__init__()
    self.swimmer = IdealFluidSwimmer()
    self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
    self.observation_space = spaces.Box(low=-pi, high=pi, shape=(3,))

  def step(self, action):
    reward, swimmer = forward_reward_function(robot=self.swimmer, action=action, c_x=1, c_joint=0, c_zero_x=1, c_theta=0.1)
    self.swimmer = swimmer
    observation = np.array([*self.swimmer.state])
    done = False
    info = {'episode': None}
    return observation, reward, done, info

  def reset(self):
    return np.array([*self.swimmer.reset()])