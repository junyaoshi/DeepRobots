import gym
from gym import spaces
from Robots.WheeledRobotPybullet import WheeledRobotPybullet
import numpy as np
from math import pi
from utils.learning_helper import forward_reward_function
import pybullet as p


class WheeledRobotPybulletEnv(gym.Env):
    def __init__(self, decision_interval, use_GUI, num_episode_steps=1000):
        super(WheeledRobotPybulletEnv, self).__init__()
        self.snake_robot = WheeledRobotPybullet(decision_interval=decision_interval, use_GUI=use_GUI)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = spaces.Box(low=-pi, high=pi, shape=(3,))
        self.num_episode_steps = num_episode_steps
        self.step_count = 0

    def step(self, action):

        if self.step_count >= self.num_episode_steps:
            self.random_theta_reset()

        self.step_count += 1

        # plane boundary detection
        if self.snake_robot.x < -90 or self.snake_robot.x > 90 or self.snake_robot.y < -90 or self.snake_robot.y > 90:
            self.snake_robot.reset(clear_step_count=False)

        reward, snake_robot = forward_reward_function(robot=self.snake_robot, action=action, c_x=1, c_joint=0,
                                                      c_zero_x=1, c_theta=0.1)
        self.snake_robot = snake_robot
        observation = np.array([*self.snake_robot.state])
        done = False
        info = {'episode': None}
        return observation, reward, done, info

    def reset(self, clear_step_count=True):
        if clear_step_count:
            self.step_count = 0

        return np.array([*self.snake_robot.reset()])

    def random_theta_reset(self):
        print("Performing random theta reset!")
        print("Theta before reset: {}".format(self.snake_robot.theta))
        self.step_count = 0

        random_theta = np.random.uniform(low=-np.pi, high=np.pi)
        init_orientation_euler = p.getEulerFromQuaternion(self.snake_robot.init_orientation)
        random_orientation_quat = p.getQuaternionFromEuler([
            random_theta, init_orientation_euler[1], init_orientation_euler[2]])
        self.snake_robot.set_system_params(
            self.snake_robot.init_position, random_orientation_quat, self.snake_robot.init_a1, self.snake_robot.init_a2)

        print("Theta after reset: {}".format(self.snake_robot.theta))

        return np.array([*self.snake_robot.state])
