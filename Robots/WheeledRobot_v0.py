"""
Robot Model File

* this version is no longer used or maintained

Type: wheeled
State space：discrete
Action space: discrete
Frame of Reference: inertial
State space singularity constraints: True

Creator: @junyaoshi
"""


import math
from math import cos, sin
from math import pi
import numpy as np
from scipy.integrate import odeint
# SET BACKEND
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt


class ThreeLinkRobot(object):

    def __init__(self, x, y, theta, a1, a2, link_length, t_interval, a_interval):
        """
        :param x: robot's initial x- displacement
        :param y: robot's initial y- displacement
        :param theta: robot's initial angle
        :param a1: joint angle of proximal link
        :param a2: joint angle of distal link
        :param link_length: length of every robot link
        :param t_interval: discretization of time
        :param a_interval: discretization of joint angle
        """

        self.x = x
        self.y = y
        self.theta = round(theta, 8)
        self.a1 = round(a1, 8)
        self.a2 = round(a2, 8)
        self.a1dot = 0
        self.a2dot = 0
        self.time = 0

        self.state = (self.theta, self.a1, self.a2)
        # self.body_v = (0, 0, 0)
        # self.inertial_v = (0, 0, 0)

        # constants
        self.t_interval = t_interval
        self.R = link_length
        self.a_interval = a_interval

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

    # def get_v(self, a1dot, a2dot):
    #     """
    #     Find the body and inertial velocity matrix of robot
    #     given controlled joint angle velocities
    #     :param a1dot: proximal joint angle velocity
    #     :param a2dot: distal joint angle velocity
    #     :return: body and inertial velocity matrix of robot
    #     """
    #     a1 = self.a1
    #     a2 = self.a2
    #     R = self.R
    #     theta = self.theta
    #     # print('a1: ', a1)
    #     # print('a2: ', a2)
    #     D = (2 / R) * (-sin(a1) - sin(a1 - a2) + sin(a2))
    #     # print('D: ', D)
    #     A = np.array([[cos(a1) + cos(a1 - a2), 1 + cos(a1)],
    #                   [0, 0],
    #                   [(2 / R) * (sin(a1) + sin(a1 - a2)), (2 / R) * sin(a1)]])
    #     adot_c = np.array([[a1dot],
    #                        [a2dot]])
    #     body_v = (1 / D) * np.matmul(A, adot_c)
    #     inertial_v = np.matmul(self.TeLg(theta), body_v)
    #
    #     return body_v, inertial_v

    def move(self, a1dot, a2dot, timestep):
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

        # body_v, inertial_v = self.get_v(a1dot, a2dot)
        #
        # # lambdafication
        # x_dot = lambda t: inertial_v[0][0]
        # y_dot = lambda t: inertial_v[1][0]
        # theta_dot = lambda t: inertial_v[2][0]
        # _a1dot = lambda t: a1dot
        # _a2dot = lambda t: a2dot
        #
        # # find the increments
        # # fix to include time intervals
        # dx, _ = integrate.quad(x_dot, 0, timestep)
        # dy, _ = integrate.quad(y_dot, 0, timestep)
        # dtheta, _ = integrate.quad(theta_dot, 0, timestep)
        # da1, _ = integrate.quad(_a1dot, 0, timestep)
        # da2, _ = integrate.quad(_a2dot, 0, timestep)

        action = (a1dot, a2dot)
        t = timestep * self.t_interval
        x, y, theta, a1, a2 = self.perform_integration(action, t)

        # testing
        # print('the increments: ')
        # print(dx, dy, dtheta, da1, da2)

        # update robot variables
        self.x = x
        self.y = y
        # self.theta = theta
        # self.a1 = a1
        # self.a2 = a2
        self.time += t
        self.a1dot = a1dot
        self.a2dot = a2dot
        # self.body_v = (body_v[0][0], body_v[1][0], body_v[2][0])
        # self.inertial_v = (inertial_v[0][0], inertial_v[1][0], inertial_v[2][0])
        # self.state = (self.theta, self.a1, self.a2)

        # discretize state variables
        # print('before: ' + str(self.state))
        self.theta = self.rnd(self.discretize(theta, self.a_interval))

        # prevent theta from going out of -pi to pi range
        self.enforce_theta_range()

        self.a1 = self.rnd(self.discretize(a1, self.a_interval))
        self.a2 = self.rnd(self.discretize(a2, self.a_interval))
        self.state = (self.theta, self.a1, self.a2)
        # print('after: ' + str(self.state))

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


if __name__ == "__main__":
    robot = ThreeLinkRobot(x=0, y=0, theta=0, a1=pi/4, a2=-pi/4, link_length=2,t_interval=1,
                           a_interval=pi/32)

    # create a robot simulation
    # robot = ThreeLinkRobot(x=0, y=0, theta=0, a1=pi/4, a2=-pi/4, link_length=2, t_interval=1, a_interval=pi/64)
    # robot.print_state()
    # x_pos = [robot.x]
    # y_pos = [robot.y]
    # thetas = [robot.theta]
    # time = [0]
    # a1 = [robot.a1]
    # a2 = [robot.a2]
    # for i in range(100):
    #     print(i)
    #     if i%2 == 0:
    #         robot.move(pi/4, -pi/4, 1)
    #     else:
    #         robot.move(-pi/4, pi/4, 1)
    #     robot.print_state()
    #     print('robot x, y:', robot.get_position())
    #     x_pos.append(robot.x)
    #     y_pos.append(robot.y)
    #     thetas.append(robot.theta)
    #     time.append(robot.time)
    #     a1.append(robot.a1)
    #     a2.append(robot.a2)
    #
    #
    # # view results
    # print('x positions are: ' + str(x_pos))
    # print('y positions are: ' + str(y_pos))
    # print('thetas are: ' + str(thetas))
    #
    # plt.plot(time, a1)
    # plt.ylabel('a1 displacements')
    # plt.show()
    #
    # plt.plot(time, a2)
    # plt.ylabel('a2 displacements')
    # plt.show()
    #
    # plt.plot(time, x_pos)
    # plt.ylabel('x positions')
    # plt.show()
    #
    # plt.plot(time, y_pos)
    # plt.ylabel('y positions')
    # plt.show()
    #
    # plt.plot(time, thetas)
    # plt.ylabel('thetas')
    # plt.show()
    # plt.close()












