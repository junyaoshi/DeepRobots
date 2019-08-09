from math import cos, sin, pi
import numpy as np
import random
from scipy.integrate import quad, odeint
# SET BACKEND
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

class SwimmingRobot(object):

    def __init__(self, x=0.0, y=0.0, theta=0.0,
                 a1=0.0, a2=0.0,
                 a_upper=pi/2, a_lower=-pi/2,
                 link_length=2, k=1, t_interval=0.25, timestep=1):
        """
        :param x: robot's initial x- displacement
        :param y: robot's initial y- displacement
        :param theta: robot's initial angle
        :param a1: joint angle of proximal link
        :param a2: joint angle of distal link
        :param a_upper: upper joint limit, None if no limit, a_upper and a_lower must be both None in that case
        :param a_lower: lower joint limit, None if no limit, a_upper and a_lower must be both None in that case
        :param link_length: length of every robot link
        :param t_interval: discretization of time
        :param k: viscosity constant
        """

        self.x = x
        self.y = y
        self.theta = theta
        self.theta_displacement = 0 # for theta reward functions
        self.a1 = a1
        self.a2 = a2
        self.a_upper = a_upper
        self.a_lower = a_lower
        self._no_joint_limit = False

        # assert check for no joint limits
        if self.a_upper is None or self.a_lower is None:
            assert self.a_upper is None and self.a_lower is None, \
                'upper and lower joint limit must both be None if there is no joint limit'
            self._no_joint_limit = True

        self.a1dot = 0
        self.a2dot = 0
        self.time = 0

        # constants
        self.t_interval = t_interval
        self.timestep = timestep
        self.L = link_length
        self.k = k

        self.state = (self.theta, self.a1, self.a2)

    # mutator methods
    def set_state(self, theta, a1, a2):
        self.theta, self.a1, self.a2 = theta, a1, a2
        self.state = (theta, a1, a2)

    # accessor methods
    def get_position(self):
        return self.x, self.y


    def randomize_state(self):
        self.theta = random.uniform(-pi, pi)
        if self._no_joint_limit:
            self.a1 = random.uniform(-pi, pi)
            self.a2 = random.uniform(-pi, pi)
        else:
            self.a1 = random.uniform(self.a_lower, self.a_upper)
            self.a2 = random.uniform(self.a_lower, self.a_upper)
        self.state = (self.theta, self.a1, self.a2)
        return self.state

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

    def J(self, a1, a2):
        """
        :return: the Jacobian matrix A_swim given joint angles
        """
        L = self.L
        return np.array([
[4*L*(72*sin(a1) + 5*sin(2*a1) - 30*sin(a2) - 7*sin(2*a2) + 6*sin(a1 - 2*a2) + 36*sin(a1 - a2) + 12*sin(a1 + a2) + 2*sin(a1 + 2*a2) + 2*sin(2*a1 + a2) + sin(2*a1 + 2*a2))/(3*(-136*cos(a1) - 14*cos(2*a1) - 136*cos(a2) - 14*cos(2*a2) + 4*cos(a1 - 2*a2) + 8*cos(a1 - a2) - 56*cos(a1 + a2) - 12*cos(a1 + 2*a2) + cos(2*a1 - 2*a2) + 4*cos(2*a1 - a2) - 12*cos(2*a1 + a2) - 3*cos(2*a1 + 2*a2) - 282)),

 4*L*(-30*sin(a1) - 7*sin(2*a1) + 72*sin(a2) + 5*sin(2*a2) - 36*sin(a1 - a2) + 12*sin(a1 + a2) + 2*sin(a1 + 2*a2) - 6*sin(2*a1 - a2) + 2*sin(2*a1 + a2) + sin(2*a1 + 2*a2))/(408*cos(a1) + 42*cos(2*a1) + 408*cos(a2) + 42*cos(2*a2) - 12*cos(a1 - 2*a2) - 24*cos(a1 - a2) + 168*cos(a1 + a2) + 36*cos(a1 + 2*a2) - 3*cos(2*a1 - 2*a2) - 12*cos(2*a1 - a2) + 36*cos(2*a1 + a2) + 9*cos(2*a1 + 2*a2) + 846)],

[4*L*(-32*(-cos(2*a1) + 1)**2 - 56*(-cos(2*a2) + 1)**2*cos(a1) + 12*(-cos(2*a2) + 1)**2*cos(2*a1) - 52*(-cos(2*a2) + 1)**2 + 3596*cos(a1) + 102*cos(2*a1) - 236*cos(3*a1) + 1312*cos(a2) + 144*cos(2*a2) - 88*cos(3*a2) + 6*cos(4*a2) - 4*cos(a1 - 4*a2) - 108*cos(a1 - 3*a2) - 14*cos(a1 - 2*a2) + 1512*cos(a1 - a2) + 1512*cos(a1 + a2) - 150*cos(a1 + 2*a2) - 108*cos(a1 + 3*a2) + 4*cos(a1 + 4*a2) - 3*cos(2*a1 - 4*a2) - 24*cos(2*a1 - 2*a2) - 96*cos(2*a1 - a2) + 40*cos(2*a1 + a2) - 24*cos(2*a1 + 2*a2) - 8*cos(2*a1 + 3*a2) - 3*cos(2*a1 + 4*a2) - 18*cos(3*a1 - 2*a2) - 108*cos(3*a1 - a2) - 108*cos(3*a1 + a2) - 10*cos(3*a1 + 2*a2) - 8*cos(4*a1 + a2) + 666)/(3*(-8*(-cos(2*a1) + 1)**2*(-cos(2*a2) + 1)**2 + 64*(-cos(2*a1) + 1)**2*cos(a2) + 16*(-cos(2*a1) + 1)**2*cos(2*a2) + 112*(-cos(2*a1) + 1)**2 + 64*(-cos(2*a2) + 1)**2*cos(a1) + 16*(-cos(2*a2) + 1)**2*cos(2*a1) + 112*(-cos(2*a2) + 1)**2 - 8224*cos(a1) + 1544*cos(2*a1) + 544*cos(3*a1) + 6*cos(4*a1) - 8224*cos(a2) + 1544*cos(2*a2) + 544*cos(3*a2) + 6*cos(4*a2) - 32*cos(a1 - 4*a2) - 32*cos(a1 - 3*a2) + 912*cos(a1 - 2*a2) + 960*cos(a1 - a2) - 3648*cos(a1 + a2) - 176*cos(a1 + 2*a2) + 224*cos(a1 + 3*a2) + 32*cos(a1 + 4*a2) - 12*cos(2*a1 - 4*a2) - 16*cos(2*a1 - 3*a2) + 224*cos(2*a1 - 2*a2) + 912*cos(2*a1 - a2) - 176*cos(2*a1 + a2) - 32*cos(2*a1 + 2*a2) + 48*cos(2*a1 + 3*a2) + 4*cos(2*a1 + 4*a2) - 16*cos(3*a1 - 2*a2) - 32*cos(3*a1 - a2) + 224*cos(3*a1 + a2) + 48*cos(3*a1 + 2*a2) + cos(4*a1 - 4*a2) - 12*cos(4*a1 - 2*a2) - 32*cos(4*a1 - a2) + 32*cos(4*a1 + a2) + 4*cos(4*a1 + 2*a2) + cos(4*a1 + 4*a2) - 18254)),

 4*L*(-56*(-cos(2*a1) + 1)**2*cos(a2) + 12*(-cos(2*a1) + 1)**2*cos(2*a2) - 52*(-cos(2*a1) + 1)**2 - 32*(-cos(2*a2) + 1)**2 + 1312*cos(a1) + 144*cos(2*a1) - 88*cos(3*a1) + 6*cos(4*a1) + 3596*cos(a2) + 102*cos(2*a2) - 236*cos(3*a2) - 108*cos(a1 - 3*a2) - 96*cos(a1 - 2*a2) + 1512*cos(a1 - a2) + 1512*cos(a1 + a2) + 40*cos(a1 + 2*a2) - 108*cos(a1 + 3*a2) - 8*cos(a1 + 4*a2) - 18*cos(2*a1 - 3*a2) - 24*cos(2*a1 - 2*a2) - 14*cos(2*a1 - a2) - 150*cos(2*a1 + a2) - 24*cos(2*a1 + 2*a2) - 10*cos(2*a1 + 3*a2) - 108*cos(3*a1 - a2) - 108*cos(3*a1 + a2) - 8*cos(3*a1 + 2*a2) - 3*cos(4*a1 - 2*a2) - 4*cos(4*a1 - a2) + 4*cos(4*a1 + a2) - 3*cos(4*a1 + 2*a2) + 666)/(3*(-8*(-cos(2*a1) + 1)**2*(-cos(2*a2) + 1)**2 + 64*(-cos(2*a1) + 1)**2*cos(a2) + 16*(-cos(2*a1) + 1)**2*cos(2*a2) + 112*(-cos(2*a1) + 1)**2 + 64*(-cos(2*a2) + 1)**2*cos(a1) + 16*(-cos(2*a2) + 1)**2*cos(2*a1) + 112*(-cos(2*a2) + 1)**2 - 8224*cos(a1) + 1544*cos(2*a1) + 544*cos(3*a1) + 6*cos(4*a1) - 8224*cos(a2) + 1544*cos(2*a2) + 544*cos(3*a2) + 6*cos(4*a2) - 32*cos(a1 - 4*a2) - 32*cos(a1 - 3*a2) + 912*cos(a1 - 2*a2) + 960*cos(a1 - a2) - 3648*cos(a1 + a2) - 176*cos(a1 + 2*a2) + 224*cos(a1 + 3*a2) + 32*cos(a1 + 4*a2) - 12*cos(2*a1 - 4*a2) - 16*cos(2*a1 - 3*a2) + 224*cos(2*a1 - 2*a2) + 912*cos(2*a1 - a2) - 176*cos(2*a1 + a2) - 32*cos(2*a1 + 2*a2) + 48*cos(2*a1 + 3*a2) + 4*cos(2*a1 + 4*a2) - 16*cos(3*a1 - 2*a2) - 32*cos(3*a1 - a2) + 224*cos(3*a1 + a2) + 48*cos(3*a1 + 2*a2) + cos(4*a1 - 4*a2) - 12*cos(4*a1 - 2*a2) - 32*cos(4*a1 - a2) + 32*cos(4*a1 + a2) + 4*cos(4*a1 + 2*a2) + cos(4*a1 + 4*a2) - 18254))],

[2*(-3*(-2*(sin(2*a1) - sin(2*a2))*(-7*cos(a1) - 2*cos(2*a1) + 7*cos(a2) + 2*cos(2*a2) + cos(a1 + 2*a2) - cos(2*a1 + a2)) + (4*sin(a1) + sin(2*a1) + 4*sin(a2) + sin(2*a2))*(cos(2*a1) + cos(2*a2) + cos(2*a1 + 2*a2) - 39))*sin(a1) - (3*cos(a1) + 4)*(cos(2*a1) + cos(2*a2) - 8)*(cos(2*a1) + cos(2*a2) + cos(2*a1 + 2*a2) - 39) + 6*(cos(2*a1) + cos(2*a2) - 8)*(-7*cos(a1) - 2*cos(2*a1) + 7*cos(a2) + 2*cos(2*a2) + cos(a1 + 2*a2) - cos(2*a1 + a2))*cos(a1))/(3*(-(cos(2*a1) + cos(2*a2) + cos(2*a1 + 2*a2) - 39)*(-28*cos(a1) + cos(2*a1) - 28*cos(a2) + cos(2*a2) + 4*cos(a1 - 2*a2) + 8*cos(a1 - a2) - 8*cos(a1 + a2) + cos(2*a1 - 2*a2) + 4*cos(2*a1 - a2) - 63) + 4*(-7*cos(a1) - 2*cos(2*a1) + 7*cos(a2) + 2*cos(2*a2) + cos(a1 + 2*a2) - cos(2*a1 + a2))**2)),                                                                                                                                                                                                                                           2*(3*(-2*(sin(2*a1) - sin(2*a2))*(-7*cos(a1) - 2*cos(2*a1) + 7*cos(a2) + 2*cos(2*a2) + cos(a1 + 2*a2) - cos(2*a1 + a2)) + (4*sin(a1) + sin(2*a1) + 4*sin(a2) + sin(2*a2))*(cos(2*a1) + cos(2*a2) + cos(2*a1 + 2*a2) - 39))*sin(a2) + (3*cos(a2) + 4)*(cos(2*a1) + cos(2*a2) - 8)*(cos(2*a1) + cos(2*a2) + cos(2*a1 + 2*a2) - 39) + 6*(cos(2*a1) + cos(2*a2) - 8)*(-7*cos(a1) - 2*cos(2*a1) + 7*cos(a2) + 2*cos(2*a2) + cos(a1 + 2*a2) - cos(2*a1 + a2))*cos(a2))/(3*(-(cos(2*a1) + cos(2*a2) + cos(2*a1 + 2*a2) - 39)*(-28*cos(a1) + cos(2*a1) - 28*cos(a2) + cos(2*a2) + 4*cos(a1 - 2*a2) + 8*cos(a1 - a2) - 8*cos(a1 + a2) + cos(2*a1 - 2*a2) + 4*cos(2*a1 - a2) - 63) + 4*(-7*cos(a1) - 2*cos(2*a1) + 7*cos(a2) + 2*cos(2*a2) + cos(a1 + 2*a2) - cos(2*a1 + a2))**2))]])

    def M(self, theta, a1, a2, da1, da2):
        """
        :return: the 5 * 1 dv/dt matrix: xdot, ydot, thetadot, a1dot, a2dot
        """
        da = np.array([[da1],
                       [da2]])
        f = self.TeLg(theta) @ (self.J(a1, a2) @ da)
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

        if self._no_joint_limit:
            enforce_angle_limits = False

        d_theta = 0

        if enforce_angle_limits:

            # print('a1: {x}, a2: {y}'.format(x=self.a1 + da1, y=self.a2+da2))

            # update integration time for each angle if necessary
            a1_t, a2_t = t, t
            if a1 < self.a_lower:
                a1_t = abs((self.a_lower - self.a1)/a1dot)
            elif a1 > self.a_upper:
                a1_t = abs((self.a_upper - self.a1)/a1dot)
            if a2 < self.a_lower:
                a2_t = abs((self.a_lower - self.a2)/a2dot)
            elif a2 > self.a_upper:
                a2_t = abs((self.a_upper - self.a2)/a2dot)

            # print('a1t: {x}, a2t: {y}'.format(x=a1_t, y= a2_t))

            # need to make 2 moves
            if abs(a1_t-a2_t) > 0.0000001:
                # print(a1_t, a2_t)
                t1 = min(a1_t, a2_t)
                old_theta = self.theta
                x, y, theta, a1, a2 = self.perform_integration(action, t1)
                d_theta += (theta - old_theta)
                self.update_params(x, y, theta, a1, a2)
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

            # only one move is needed
            else:
                # print('b')
                if t != a1_t:
                    t = a1_t
                old_theta = self.theta
                x, y, theta, a1, a2 = self.perform_integration(action, t)
                d_theta += (theta - old_theta)
                self.update_params(x, y, theta, a1, a2)
                self.update_alpha_dots(a1dot, a2dot, t)
        else:
            # print('a')
            old_theta = self.theta
            x, y, theta, a1, a2 = self.perform_integration(action, t)
            d_theta += (theta - old_theta)
            self.update_params(x, y, theta, a1, a2, enforce_angle_limits=False)
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
        if self._no_joint_limit:
            return
        if abs(self.a1 - self.a_upper) < tolerance:
            self.a1 = self.a_upper
        elif abs(self.a1 - self.a_lower) < tolerance:
            self.a1 = self.a_lower
        if abs(self.a2 - self.a_lower) < tolerance:
            self.a2 = self.a_lower
        elif abs(self.a2 - self.a_upper) < tolerance:
            self.a2 = self.a_upper

    def check_angles(self):
        if self._no_joint_limit:
            return
        if self.a1 < self.a_lower or self.a1 > self.a_upper:
            raise Exception('a1 out of limit: {x}'.format(x=self.a1))
        if self.a2 < self.a_lower or self.a2 > self.a_upper:
            raise Exception('a2 out of limit: {x}'.format(x=self.a2))

    def print_state(self):
        """
        print the current state
        :return: None
        """
        print('\nthe current state is: ' + str(self.state) + '\n')


if __name__ == "__main__":

    # create a robot simulation
    robot = SwimmingRobot(a1=0, a2=0, t_interval=1)
    # print('x, y:', robot.get_position())
    # robot.print_state()
    # robot.move(action=(0.39269908169872414, 0.39269908169872414))
    # print('x, y:', robot.get_position())
    # robot.print_state()
    # robot.move(action=(-0.39269908169872414, 0.39269908169872414))
    # print('x, y:', robot.get_position())
    # robot.print_state()
    # robot.move(action=(0.39269908169872414, -0.39269908169872414))
    # print('x, y:', robot.get_position())
    # robot.print_state()
    # robot.move(action=(-0.39269908169872414, 0.39269908169872414))
    # print('x, y:', robot.get_position())
    # robot.print_state()
    # robot.move(action=(0.39269908169872414, -0.39269908169872414))
    # print('x, y:', robot.get_position())
    # robot.print_state()
    # robot.print_state()
    # robot.move(action=(-pi/5,pi/5))
    # robot.print_state()
    # robot.move(action=(pi / 5, -pi / 5))
    # robot.print_state()

    # fx, fy, ftheta = robot.get_v(0.5,0.5)
    # print(robot.perform_integration((0.5, 0.5),fx, fy, ftheta, 0.1))


    x_pos = [robot.x]
    y_pos = [robot.y]
    thetas = [robot.theta]
    theta_displacements = [robot.theta_displacement]
    time = [0]
    a1 = [robot.a1]
    a2 = [robot.a2]
    print('initial x y theta a1 a2: ', robot.x, robot.y, robot.theta, robot.a1, robot.a2)
    for t in range(1000):
        print(t+1, 'th iteration')
        a1dot = 1/3*cos(t/5)
        a2dot = -1/3*sin(t/5)
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
        theta_displacements.append(robot.theta_displacement)


    # view results
    # print('x positions are: ' + str(x_pos))
    # print('y positions are: ' + str(y_pos))
    # print('thetas are: ' + str(thetas))

    plt.plot(x_pos, y_pos)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('y vs x')
    plt.show()

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

    plt.plot(time, theta_displacements)
    plt.ylabel('theta displacements')
    plt.show()
    plt.close()













