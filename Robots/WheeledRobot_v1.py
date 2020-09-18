"""
Robot Model File

Type: wheeled
State space：continuous
Action space: discrete
Frame of Reference: inertial
State space singularity constraints: True

Creator: @junyaoshi
"""

import math
from math import cos, sin, pi
import numpy as np
import random
from scipy.integrate import quad, odeint
# SET BACKEND
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


class ThreeLinkRobot(object):

    def __init__(self, x=0, y=0, theta=0, a1=-pi/4, a2=pi/4, link_length=2, t_interval=0.001, timestep=1):
        """
        :param x: robot's initial x- displacement
        :param y: robot's initial y- displacement
        :param theta: robot's initial angle
        :param a1: joint angle of proximal link
        :param a2: joint angle of distal link
        :param link_length: length of every robot link
        :param t_interval: discretization of time
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

        # symbolic variable
        # self.t = symbols('t')

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

    '''
    def get_v(self, a1dot, a2dot):
        """
        Find the body and inertial velocity matrix of robot
        given controlled joint angle velocities
        :param a1dot: proximal joint angle velocity
        :param a2dot: distal joint angle velocity
        :return: body and inertial velocity matrix of robot
        """
        a1 = self.a1 + a1dot * self.t
        a2 = self.a2 + a2dot * self.t
        R = self.R
        theta = self.theta
        D = (2/R) * (-sin(a1) - sin(a1 - a2) + sin(a2))
        A = Matrix([[cos(a1) + cos(a1 - a2), 1 + cos(a1)],
                    [0, 0],
                    [(2/R) * (sin(a1) + sin(a1 - a2)), (2/R) * sin(a1)]])
        adot = Matrix([[a1dot],
                       [a2dot]])
        body_v = simplify(1/D * A * adot)
        inertial_v = simplify(self.TeLg(theta) * body_v)
        # print('body_v: {x}'.format(x=body_v.evalf()))
        # print('inertial_v: {y}'.format(y=inertial_v.evalf()))
        fx = lambdify(self.t, inertial_v[0], 'numpy')
        fy = lambdify(self.t, inertial_v[1], 'numpy')
        ftheta = lambdify(self.t, inertial_v[2], 'numpy')

        return fx, fy, ftheta
    '''



    ### CHANGE THIS TO ODEINT
    '''
    def perform_integration(self, action, fx, fy, ftheta, t):

        a1dot, a2dot = action

        # print('t: ', t)
        dx = quad(fx, 0, t)[0]
        print('dx: ', dx)
        dy = quad(fy, 0, t)[0]
        # print('dy: ', dy)
        dtheta = quad(ftheta, 0, t)[0]
        # print('dt: ', dtheta)
        da1 = t * a1dot
        # print('da1: ', da1)
        da2 = t * a2dot
        # print('da2: ', da2)

        # print('a1dot: {x}, {t}, da1: {y}'.format(x=a1dot, t=t, y=da1))

        return da1, da2, dx, dy, dtheta
    '''

    '''
    def update_velocity_matrices(self, body_v1, inertial_v1, t1=None,
                                 body_v2=Matrix([[0],[0],[0]]), inertial_v2=0, t2=None):

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
        print('a: ', body_v1, 'b: ', body_v2)
        self.body_v = (c1 * body_v1 + c2 * body_v2) * c3
        self.inertial_v = (c1 * inertial_v1 + c2 * inertial_v2) * c3
    '''

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


if __name__ == "__main__":

    # create a robot simulation
    robot = ThreeLinkRobot(t_interval=1, a1=(0.5*cos(1))-0.6, a2=1.1)

    # dx, dy, dtheta, da1, da2, xdot = robot.perform_integration((0.5, 0.5), 0.1)
    # print(dx, dy, dtheta, da1, da2)

    # robot.move((pi/2, -pi/2))
    # robot.print_state()
    # print('x: ', robot.x)
    # print('y: ', robot.y)
    # print('t: ', robot.theta)


    x_pos = [robot.x]
    y_pos = [robot.y]
    thetas = [robot.theta]
    time = [0]
    a1 = [robot.a1]
    a2 = [robot.a2]
    print('initial x y theta a1 a2: ', robot.x, robot.y, robot.theta, robot.a1, robot.a2)
    for t in range(1000):
        print(t+1, 'th iteration')
        a1dot = -0.5/20*sin(t/20+1)
        a2dot = -0.5/20*sin(t/20)
        action = (a1dot, a2dot)
        robot.move(action)
        print('action taken(a1dot, a2dot): ', action)
        print('robot x y theta a1 a2: ', robot.x, robot.y, robot.theta, robot.a1, robot.a2)
        x_pos.append(robot.x)
        y_pos.append(robot.y)
        thetas.append(robot.theta)
        time.append(t+1)
        a1.append(robot.a1)
        a2.append(robot.a2)


    # view results
    print('x positions are: ' + str(x_pos))
    print('y positions are: ' + str(y_pos))
    print('thetas are: ' + str(thetas))

    plt.plot(time, a1)
    plt.ylabel('a1 displacements')
    plt.show()

    plt.plot(time, a2)
    plt.ylabel('a2 displacements')
    plt.show()

    plt.plot(time, x_pos)
    plt.ylabel('x positions')
    plt.show()

    plt.plot(time, y_pos)
    plt.ylabel('y positions')
    plt.show()

    plt.plot(time, thetas)
    plt.ylabel('thetas')
    plt.show()
    plt.close()














