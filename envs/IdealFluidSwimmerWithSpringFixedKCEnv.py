"""
For Training with Fixed K, C Values
"""

import gym
from gym import spaces
from Robots.IdealFluidSwimmerWithSpring_v1 import IdealFluidSwimmerWithSpring
import numpy as np
from math import pi
from utils.learning_helper import forward_reward_function

# Fixed Values
K = 1000
C= 2000

class IdealFluidSwimmerWithSpringEnv(gym.Env):
    def __init__(self):
        super(IdealFluidSwimmerWithSpringEnv, self).__init__()
        self.snake_robot = IdealFluidSwimmerWithSpring()
        self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), shape=(1,))
        self.observation_space = spaces.Box(low=np.array([-pi, -pi, -10, -pi, -10]),
                                            high=np.array([pi, pi, 10, pi, 10]), shape=(5,))

    def step(self, normalized_action):
        a1ddot = normalized_action
        action = (a1ddot, K, C)
        reward, snake_robot = forward_reward_function(robot=self.snake_robot, action=action, c_x=1, c_joint=0,
                                                      c_zero_x=1, c_theta=0.1)
        self.snake_robot = snake_robot
        observation = np.array([*self.snake_robot.state])
        done = False
        info = {'episode': None}
        return observation, reward, done, info

    def reset(self):
        return np.array([*self.snake_robot.reset()])
