import gym
from gym import error, spaces, utils
from gym.utils import seeding

import math
from math import cos, sin, pi
import numpy as np
import random
from scipy.integrate import quad, odeint
# SET BACKEND
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

class DiscreteDeepRobotEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    #required method for environment setup
    def __init__(self, x=0, y=0, theta=0, a1=-pi/4, a2=pi/4, link_length=2, t_interval=0.001, timestep=1, theta_range=(-pi,pi), a1_range=(-pi/2,pi/2), a2_range=(-pi/2,pi/2), a1_amount = 6, a2_amount = 6):
        """
        :param x: robot's initial x- displacement
        :param y: robot's initial y- displacement
        :param theta: robot's initial angle
        :param a1: joint angle of proximal link
        :param a2: joint angle of distal link
        :param link_length: length of every robot link
        :param t_interval: discretization of time
        :param theta_range: range of possible theta values observed
        :param a1_range: range of possible a1 values observed
        :param a2_range: range of possible a2 values observed
        :param a1_range: range of possible a1dot values for actions
        :param a2_range: range of possible a2dot values for actions
        :param a1_amount: the number of possible a1dot values in the defined interval
        :param a2_amount: the number of possible a2dot values in the defined interval
        
        """

        self.x = x
        self.y = y
        self.theta = theta
        self.theta_displacement = 0
        self.a1 = a1
        self.a2 = a2
        self.a1 = 0
        self.a2 = 0

        self.state = (self.theta, self.a1, self.a2)
        # self.body_v = (0, 0, 0)
        # self.inertial_v = (0, 0, 0)

        # constants
        self.t_interval = t_interval
        self.timestep = timestep
        self.R = link_length
        
        #for visualization
        self.x_pos = [self.x]
        self.y_pos = [self.y]
        self.thetas = [self.theta]
        self.time = [0]
        self.a1s = [self.a1]
        self.a2s = [self.a2]
        
        #for env requirements
        self.action_space = spaces.Discrete(a1_amount*a2_amount)
        
        self.a_interval = a1_range[1]-a1_range[0]#temp
        self.actionDictionary={}
        self.generateActionDictionary(a1_range,a2_range,a1_amount,a2_amount)
        # Example for using image as input:
        self.observation_space = spaces.Box(np.array([theta_range[0],a1_range[0],a2_range[0]]),np.array([theta_range[1],a1_range[1],a2_range[1]]))


    # generate state dictionary
    def generateActionDictionary(self,a1_range,a2_range,a1_amount,a2_amount):
        count = 0
        i = a1_range[0]
        j = a2_range[0]
        while (i<=a1_range[1]):
            j = a2_range[0]
            while (j<=a2_range[1]):
                if(round(i,1)!=0 and round(j,1)!=0):
                    self.actionDictionary[count]=(i,j)
                    count+=1
                j+=(a2_range[1]-a2_range[0])/(a2_amount)
            i+=(a1_range[1]-a1_range[0])/(a1_amount)

    # mutator methods
    def set_state(self, theta, a1, a2):
        self.state = (theta, a1, a2)

    # accessor methods
    def get_position(self):
        return self.x, self.y

    # helper methods
    @staticmethod
    def TeLg(theta):
        """
        :param theta: the inertial angle in radians
        :return: the lifted left action matrix given the angle
        """
        arr = np.array([[cos(theta), -sin(theta), 0],
                        [sin(theta), cos(theta), 0],
                        [0, 0, 1]])
        return arr

    @staticmethod
    def discretize(val, interval):
        '''
        :param val: input non-discretized value
        :param interval: interval for discretization
        :return: discretized value
        '''
        quotient = val / interval
        floor = math.floor(quotient)
        diff = quotient - floor
        if diff >= 0.5:
            discretized_val = (floor + 1) * interval
        else:
            discretized_val = floor * interval
        return discretized_val

    def D_inverse(self, a1, a2):
        """
        :return: the inverse of D function
        """
        R = self.R
        D = (2/R) * (-sin(a1) - sin(a1 - a2) + sin(a2))
        if(D!=0):
            return 1/D
        else:
            return float('inf')

    def A(self, a1, a2):
        """
        :return: the Jacobian matrix
        """
        R = self.R
        A = np.array([[cos(a1) + cos(a1 - a2), 1 + cos(a1)],
                      [0, 0],
                      [(2/R) * (sin(a1) + sin(a1 - a2)), (2/R) * sin(a1)]])
        return A

    def M(self, theta, a1, a2, da1, da2):
        """
        :return: the 5 * 1 dv/dt matrix: xdot, ydot, thetadot, a1dot, a2dot
        """
        da = np.array([[da1],
                       [da2]])
        f = self.D_inverse(a1, a2) * (self.TeLg(theta) @ (self.A(a1, a2) @ da))
        xdot = f[0,0]
        ydot = f[1,0]
        thetadot = f[2,0]
        M = [xdot, ydot, thetadot, da1, da2]
        return M

    def robot(self, v, t, da1, da2):
        """
        :return: function used for odeint integration
        """
        _, _, theta, a1, a2 = v
        dvdt = self.M(theta, a1, a2, da1, da2)
        return dvdt

    def perform_integration(self, action, t_interval):
        """
        :return: perform integration of ode, return the final displacements and x-velocity
        """

        if t_interval == 0:
            return self.x, self.y, self.theta, self.a1, self.a2
        a1dot, a2dot = action
        v0 = [self.x, self.y, self.theta, self.a1, self.a2]
        t = np.linspace(0, t_interval, 11)
        sol = odeint(self.robot, v0, t, args=(a1dot, a2dot))
        x, y, theta, a1, a2 = sol[-1]
        return x, y, theta, a1, a2
    
    def move(self, a1dot, a2dot, timestep=1):
        """
        Implementation of Equation 9
        given the joint velocities of the 2 controlled joints
        and the number of discretized time intervals
        move the robot accordingly
        :param a1dot: joint velocity of the proximal joint
        :param a2dot: joint velocity of the distal joint
        :param timestep: number of time intevvals
        :return: new state of the robot
        """
        action = (a1dot, a2dot)
        t = timestep * self.t_interval
        x, y, theta, a1, a2 = self.perform_integration(action, t)

        # update robot variables
        self.x = x
        self.y = y
        #self.time += t
        self.a1dot = a1dot
        self.a2dot = a2dot

        self.theta = self.rnd(self.discretize(theta, self.a_interval))

        self.enforce_theta_range()

        self.a1 = self.rnd(self.discretize(a1, self.a_interval))
        self.a2 = self.rnd(self.discretize(a2, self.a_interval))
        self.state = (self.theta, self.a1, self.a2)

        return self.state

    def enforce_theta_range(self):
        angle = self.theta
        if angle > pi:
            angle = angle % (2 * pi)
            if angle > pi:
                angle = angle - 2 * pi
        elif angle < -pi:
            angle = angle % (-2 * pi)
            if angle < -pi:
                angle = angle + 2 * pi
        self.theta = angle

    @staticmethod
    def rnd(number):
        return round(number, 8)

    def print_state(self):
        """
        print the current state
        :return: None
        """
        print('\nthe current state is: ' + str(self.state) + '\n')

    def reward_function(self,action,c_x=50, c_joint=0, c_zero_x=50, c_theta=5,penalize_joint_limit=False, reward_theta=True):
        
        old_x=self.x
        old_y=self.y
        old_theta=self.theta
        old_a1=self.a1
        old_a2=self.a2
        self.move(action[0],action[1])
        new_x=self.x
        new_y=self.y
        new_theta=self.theta
        new_a1=self.a1
        new_a2=self.a2

        x_displacement_reward = new_x - old_x
        old_as = [old_a1, old_a2]
        new_as = [new_a1, new_a2]

        joint_penalty = 0
        if penalize_joint_limit and c_joint != 0:
            for i in range(len(old_as)):
                if abs(old_as[i] - pi/2) <= 0.00001 or abs(old_as[i] + pi/2) <= 0.00001:
                    if old_as[i] == new_as[i]:
                        joint_penalty = -1
                        print('incur joint limit penalty')
        zero_x_penalty = 0
        if x_displacement_reward == 0:
            print('incur 0 x displacement penalty')
            zero_x_penalty = -1

        theta_reward = 0
        if reward_theta:
            if -pi / 4 <= new_theta <= pi / 4:
                theta_reward = 1
            else:
                theta_reward = pi / 4 - abs(new_theta)

        reward = c_x * x_displacement_reward + c_joint * joint_penalty + \
                 c_zero_x * zero_x_penalty + c_theta * theta_reward

        return reward
    
    # required methods for environment setup
    def step(self, action):
        """
        :return state, reward, episode complete, infor
        """
        self.move(self.actionDictionary[action][0],self.actionDictionary[action][1])
        reward=self.reward_function(self.actionDictionary[action])
        self.x_pos.append(self.x)
        self.y_pos.append(self.y)
        self.thetas.append(self.theta)
        self.time.append(self.time[-1]+1)
        self.a1s.append(self.a1)
        self.a2s.append(self.a2)
        return self.state,reward,False, {}
        
    def reset(self):
        self.x = 0 #robot's initial x- displacement
        self.y = 0 #robot's initial y- displacement
        self.theta = 0 #robot's initial angle
        self.theta_displacement = 0
        self.a1 = (0.5*cos(1))-0.6 #-pi/4 #joint angle of proximal link
        self.a2 = 1.1 #pi/4 #joint angle of distal link
        self.a1dot = 0
        self.a2dot = 0

        self.state = (self.theta, self.a1, self.a2)

        # constants
        self.t_interval = 0.001 #discretization of time
        self.timestep = 1
        self.R = 2 #length of every robot link
        
        #for visualization
        self.x_pos = [self.x]
        self.y_pos = [self.y]
        self.thetas = [self.theta]
        self.time = [0]
        self.a1s = [self.a1]
        self.a2s = [self.a2]
    
    def render(self, mode='human'):
        # view results
        #print('x positions are: ' + str(x_pos))
        #print('y positions are: ' + str(y_pos))
        #print('thetas are: ' + str(thetas))

        plt.plot(self.time, self.a1s)
        plt.ylabel('a1 displacements')
        plt.show()

        plt.plot(self.time, self.a2s)
        plt.ylabel('a2 displacements')
        plt.show()

        plt.plot(self.time, self.x_pos)
        plt.ylabel('x positions')
        plt.show()

        plt.plot(self.time, self.y_pos)
        plt.ylabel('y positions')
        plt.show()

        plt.plot(self.time, self.thetas)
        plt.ylabel('thetas')
        plt.show()
        plt.close()

