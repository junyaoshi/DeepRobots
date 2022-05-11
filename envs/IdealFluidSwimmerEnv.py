import gym
from gym import spaces
from Robots.IdealFluidSwimmer_v3 import IdealFluidSwimmer
import numpy as np
from math import pi
from utils.learning_helper import forward_reward_function
import random


class IdealFluidSwimmerEnv(gym.Env):
    def __init__(self, reset_theta = False, num_episode_steps=500, interval_num=60, a_upper = pi/2, a_lower = -pi/2):
        super(IdealFluidSwimmerEnv, self).__init__()
        self.snake_robot = IdealFluidSwimmer(a1=0, a2=0, a_upper = a_upper, a_lower = a_lower, no_joint_limit = False)#IdealFluidSwimmer(ahead=0, atail=-5/3, no_joint_limit=False, t_interval=1)#IdealFluidSwimmer()
        self.action_space = spaces.Box(low=-4, high=4, shape=(2,))
        self.observation_space = spaces.Box(low=-pi, high=pi, shape=(3,))

        self.num_episode_steps = num_episode_steps
        self.toggle = 0 #for alternating theta test
        self.interval_num = 60
        self.step_count = 0
        self.intervals = self.gen_interval_tuples()
        self.int_step = 0
        self.reset_theta = reset_theta
        self.is_test = False
        self.did_reset = False

    def step(self, action):
        if self.is_test == False and self.reset_theta and (self.step_count != 0 and (self.step_count % self.num_episode_steps) == 0):
            self.do_random_reset()
            print('reset')
            print(self.reset_theta)
            self.did_reset = True
        else:
            self.did_reset = False

        self.step_count += 1

        reward, snake_robot = forward_reward_function(robot=self.snake_robot, action=action, c_x=1, c_joint=0,
                                                      c_zero_x=1, c_theta=0.1)
        self.snake_robot = snake_robot
        observation = np.array([*self.snake_robot.state])
        done = False
        info = {'episode': None}
        return observation, reward, done, info

    def gen_interval_tuples(self):
        interval_bounds = np.linspace(0, 2*pi, num = self.interval_num + 1)
        intervals = []
        for i in range(len(interval_bounds) - 1):
            intervals.append((interval_bounds[i], interval_bounds[i+1]))
        return intervals

    def reset(self):
        return np.array([*self.snake_robot.reset()])

    def enable_test_mode(self):
        self.is_test = True

    def disable_test_mode(self):
        self.is_test = False

    def do_random_reset(self):
        #generate random value between 0 and pi and set the env theta to it
        rand_max = self.intervals[self.int_step][1]
        rand_min = self.intervals[self.int_step][0]
        rand_theta = rand_min + random.random()*(rand_max - rand_min) 
        # obs = self.env.envs[0].reset_w_theta(rand_theta)
        self.reset_w_theta(rand_theta)
        self.int_step += 1
        if(self.int_step % self.interval_num == 0):
            self.int_step = 0
            random.shuffle(self.intervals)
        
        # if(self.toggle == False):
        #     self.reset_w_theta(pi)
        # else:
        #     self.reset_w_theta(0)
        # self.toggle = not(self.toggle)    
            #obs = self.env.envs[0].reset_w_theta(pi/2)

    def reset_w_theta(self, _theta):
        return np.array([*self.snake_robot.reset_w_theta(_theta)])

    def set_theta(self, _theta):
        return np.array([*self.snake_robot.set_theta(_theta)])
