import gym
from gym import spaces
from Robots.IdealFluidSwimmer_v1 import IdealFluidSwimmer
import numpy as np
from math import pi
from utils.learning_helper import forward_reward_function


class IdealFluidSwimmerEnv(gym.Env):
    def __init__(self):
        super(IdealFluidSwimmerEnv, self).__init__()
        self.snake_robot = IdealFluidSwimmer()
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = spaces.Box(low=-pi, high=pi, shape=(3,))

    def step(self, action):
        reward, snake_robot = forward_reward_function(robot=self.snake_robot, action=action, c_x=1, c_joint=0,
                                                      c_zero_x=1, c_theta=0.1)
        self.snake_robot = snake_robot
        observation = np.array([*self.snake_robot.state])
        done = False
        info = {'episode': None}
        return observation, reward, done, info

    def reset(self):
        return np.array([*self.snake_robot.reset()])
