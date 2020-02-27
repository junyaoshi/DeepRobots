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

class DeepRobotEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, x=0, y=0, theta=0, a1=-pi/4, a2=pi/4, link_length=2, t_interval=0.001, timestep=1, theta_range=(-pi,pi), a1_range=(-pi/2,pi/2), a2_range=(-pi/2,pi/2)):
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
        :param a1dot_range: range of possible a1dot values for actions
        :param a2dot_range: range of possible a2dot values for actions
        
        """

        self.x = x
        self.y = y
        self.theta = theta
        self.theta_displacement = 0
        self.a1 = a1
        self.a2 = a2
        self.a1dot = 0
        self.a2dot = 0

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
        self.action_space = spaces.Box(np.array([a1dot_range[0], a1dot_range[1]]),np.array([a2dot_range[0], a2dot_range[1]]))
        # Example for using image as input:
        self.observation_space = spaces.Box(np.array([theta_range[0], theta_range[1]]),np.array([a1_range[0],a1_range[1]]),np.array([a2_range[0],a2_range[1]]))
        
    # mutator methods
    def set_state(self, theta, a1, a2):
        self.theta, self.a1, self.a2 = theta, a1, a2
        self.state = (theta, a1, a2)

    # accessor methods
    def get_position(self):
        return self.x, self.y

    def randomize_state(self, enforce_opposite_angle_signs=True):
        self.theta = random.uniform(-pi, pi)
        self.a1 = random.uniform(-pi/2, 0) if enforce_opposite_angle_signs else random.uniform(-pi/2, pi/2)
        self.a2 = random.uniform(0, pi/2) if enforce_opposite_angle_signs else random.uniform(-pi/2, pi/2)
        self.state = (self.theta, self.a1, self.a2)
        return self.state
    
    # helper methods
    @staticmethod
    def TeLg(theta):
        """
        :param theta: the inertial angle in radians
        :return: the lifted left action matrix given the angle
        """
        return np.array([[cos(theta), -sin(theta), 0],
                         [sin(theta), cos(theta), 0],
                         [0, 0, 1]])

    def D_inverse(self, a1, a2):
        """
        :return: the inverse of D function
        """
        R = self.R
        # print('a1 a2: ', a1, a2)
        D = (2/R) * (-sin(a1) - sin(a1 - a2) + sin(a2))
        return 1/D

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
        # print('a1 a2:', a1, a2)
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
    
    # actual movement function
    def move(self, action, timestep=1, enforce_angle_limits=True):
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
        self.timestep = timestep

        if enforce_angle_limits:
            self.check_angles()

        a1dot, a2dot = action
        # print('action: ', action)
        t = self.timestep * self.t_interval
        # print('t: ', t)
        a1 = self.a1 + a1dot * t
        a2 = self.a2 + a2dot * t
        # print('ds: ', x, y, theta, a1, a2)

        d_theta = 0

        if enforce_angle_limits:

            # print('a1: {x}, a2: {y}'.format(x=self.a1 + da1, y=self.a2+da2))

            # update integration time for each angle if necessary
            a1_t, a2_t = t, t
            if a1 < -pi/2:
                a1_t = (-pi/2 - self.a1)/a1dot
            elif a1 > 0:
                a1_t = (0 - self.a1)/a1dot
            if a2 < 0:
                a2_t = (0 - self.a2)/a2dot
            elif a2 > pi/2:
                a2_t = (pi/2 - self.a2)/a2dot

            # print('a1t: {x}, a2t: {y}'.format(x=a1_t, y= a2_t))

            # need to make 2 moves
            if abs(a1_t-a2_t) > 0.0000001:
                # print(a1_t, a2_t)
                t1 = min(a1_t, a2_t)
                old_theta = self.theta
                x, y, theta, a1, a2 = self.perform_integration(action, t1)
                d_theta += (theta - old_theta)
                self.update_params(x, y, theta, a1, a2)
                if self.a1 == 0 and self.a2 == 0:
                    # self.update_velocity_matrices(body_v1, inertial_v1, t1)
                    self.update_alpha_dots(a1dot, a2dot, t1)
                else:
                    if a2_t > a1_t:
                        t2 = a2_t - a1_t
                        action = (0, a2dot)
                        old_theta = self.theta
                        x, y, theta, a1, a2 = self.perform_integration(action, t2)
                        d_theta += (theta - old_theta)
                        self.update_params(x, y, theta, a1, a2)
                        self.update_alpha_dots(a1dot, a2dot, t1, 0, a2dot, t2)
                    else:
                        t2 = a1_t - a2_t
                        action = (a1dot, 0)
                        old_theta = self.theta
                        x, y, theta, a1, a2 = self.perform_integration(action, t2)
                        d_theta += (theta - old_theta)
                        self.update_params(x, y, theta, a1, a2)
                        self.update_alpha_dots(a1dot, a2dot, t1, a1dot, 0, t2)
                    # self.update_velocity_matrices(body_v1, inertial_v1, t1, body_v2, inertial_v2, t2)

            # only one move is needed
            else:
                # print('b')
                if t != a1_t:
                    t = a1_t
                old_theta = self.theta
                x, y, theta, a1, a2 = self.perform_integration(action, t)
                d_theta += (theta - old_theta)
                self.update_params(x, y, theta, a1, a2)
                # self.update_velocity_matrices(body_v, inertial_v, t)
                self.update_alpha_dots(a1dot, a2dot, t)
        else:
            # print('a')
            old_theta = self.theta
            x, y, theta, a1, a2 = self.perform_integration(action, t)
            d_theta += (theta - old_theta)
            self.update_params(x, y, theta, a1, a2, enforce_angle_limits=False)
            # self.update_velocity_matrices(body_v, inertial_v, t)
            self.update_alpha_dots(a1dot, a2dot, t)

        self.theta_displacement = d_theta
        self.state = (self.theta, self.a1, self.a2)

        return self.state
    
    # enforce angle range
    
    def enforce_angle_range(self, angle_name):
        if angle_name == 'theta':
            angle = self.theta
        elif angle_name == 'a1':
            angle = self.a1
        else:
            angle = self.a2
        if angle > pi:
            angle = angle % (2 * pi)
            if angle > pi:
                angle = angle - 2 * pi
        elif angle < -pi:
            angle = angle % (-2 * pi)
            if angle < -pi:
                angle = angle + 2 * pi
        if angle_name == 'theta':
            self.theta = angle
        elif angle_name == 'a1':
            self.a1 = angle
        elif angle_name == 'a2':
            self.a2 = angle
            
    def update_alpha_dots(self, a1dot1, a2dot1, t1=None, a1dot2=0, a2dot2=0, t2=None):

        c3 = 1

        # one move made
        if t2 is None:
            t2 = 0

        if t1 + t2 < self.t_interval + self.timestep:
            c3 = (t1 + t2)/(self.t_interval * self.timestep)

        if t1+t2 == 0:
            c1 = 0
            c2 = 0
        else:
            c1 = t1/(t1+t2)
            c2 = t2/(t1+t2)
        self.a1dot = (c1 * a1dot1 + c2 * a1dot2) * c3
        self.a2dot = (c1 * a2dot1 + c2 * a2dot2) * c3
    
    def update_params(self, x, y, theta, a1, a2, enforce_angle_limits=True):
        # update robot variables
        self.x = x
        self.y = y
        self.theta = theta
        self.enforce_angle_range('theta')

        self.a1 = a1
        if not enforce_angle_limits:
            self.enforce_angle_range('a1')

        self.a2 = a2
        if not enforce_angle_limits:
            self.enforce_angle_range('a2')

        self.round_angles_to_limits()

    def round_angles_to_limits(self, tolerance=0.000000001):
        if abs(self.a1-0) < tolerance:
            self.a1 = 0
        elif abs(self.a1+pi/2) < tolerance:
            self.a1 = -pi/2
        if abs(self.a2-0) < tolerance:
            self.a2 = 0
        elif abs(self.a2-pi/2) < tolerance:
            self.a2 = pi/2

    def check_angles(self):
        if self.a1 < -pi/2 or self.a1 > 0:
            raise Exception('a1 out of limit: {x}'.format(x=self.a1))
        if self.a2 < 0 or self.a2 > pi/2:
            raise Exception('a2 out of limit: {x}'.format(x=self.a2))

    def print_state(self):
        """
        print the current state
        :return: None
        """
        print('\nthe current state is: ' + str(self.state) + '\n')
     
    # reward function
    def reward_function(self,action,c_x=50, c_joint=0, c_zero_x=50, c_theta=5,penalize_joint_limit=False, reward_theta=True):
        
        old_x=self.x
        old_y=self.y
        old_theta=self.theta
        old_a1=self.a1
        old_a2=self.a2
        self.move(action)
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
        self.move(action)
        reward=self.reward_function(action)
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
        #print('x positions are: ' + str(self.x_pos))
        #print('y positions are: ' + str(self.y_pos))
        #print('thetas are: ' + str(self.thetas))

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
