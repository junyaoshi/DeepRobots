"""
Robot Model File

Type: swimmer in ideal fluid
State space：continuous
Action space: continuous
Frame of Reference: inertial
State space singularity constraints: False

Creator: @junyaoshi
"""



from math import cos, sin, pi
import numpy as np
import random
from scipy.integrate import quad, odeint

# SET BACKEND
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from utils.csv_generator import generate_csv


class IdealFluidSwimmer(object):

    def __init__(self,
                 x=0.0,
                 y=0.0,
                 theta=0.0,
                 a1=0.0,
                 a2=0.0,
                 a_upper=pi / 2,
                 a_lower=-pi / 2,
                 no_joint_limit=False,
                 link_length=2,
                 k=1,
                 t_interval=0.25,
                 timestep=1):
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

        self.init_x = x
        self.init_y = y
        self.init_theta = theta
        self.init_a1 = a1
        self.init_a2 = a2

        self.x = x
        self.y = y
        self.theta = theta
        self.theta_displacement = 0  # for theta reward functions
        self.a1 = a1
        self.a2 = a2
        self.a_upper = a_upper
        self.a_lower = a_lower
        self._no_joint_limit = no_joint_limit

        self.a1dot = 0
        self.a2dot = 0
        self.time = 0

        # constants
        self.t_interval = t_interval
        self.timestep = timestep
        self.L = link_length
        self.k = k

        self.state = (self.theta, self.a1, self.a2)

    def reset(self):
        self.x = self.init_x
        self.y = self.init_y
        self.theta = self.init_theta
        self.a1 = self.init_a1
        self.a2 = self.init_a2
        self.state = (self.theta, self.a1, self.a2)

        self.theta_displacement = 0
        self.a1dot = 0
        self.a2dot = 0
        self.time = 0
        return self.state

    # mutator methods
    def set_state(self, theta, a1, a2):
        self.theta, self.a1, self.a2 = theta, a1, a2
        self.state = (theta, a1, a2)
        return self.state

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
            [
                (-17 * (1902938906458884 * sin(a1) + 67008778932370 * sin(2 * a1) + 430635669371306 * sin(
                    a1 - 2 * a2) + 1493437037304534 * sin(a1 - a2) - 1027399540055203 * sin(a2) - 430803566835808 * sin(
                    2 * a2) + 497812345768178 * sin(a1 + a2) + 13072332995 * (
                                2077 * sin(2 * (a1 + a2)) + 2401 * sin(2 * a1 + a2)) + 81648799884050 * sin(
                    a1 + 2 * a2))) /
                (6950344987924878 + 4433052168153896 * cos(a2) + 272004488 * cos(a1) * (
                        16297717 + 4667544 * cos(a2) - 2295085 * cos(2 * a2)) + 58516346103350 * cos(
                    2 * a2) - 20770 * cos(2 * a1) * (
                         -2817349355 + 30056495924 * cos(a2) + 17187045685 * cos(2 * a2)) - 829341683912 * sin(a1) * (
                         4802 * sin(a2) + 2077 * sin(2 * a2)) - 4154 * sin(2 * a1) * (
                         414670841956 * sin(a2) + 158422965851 * sin(2 * a2))),

                (17 * (2401 * (-427904848003 - 414670841956 * cos(a2) + 13072332995 * cos(2 * a2)) * sin(a1) + (
                        -430803566835808 - 348986869487256 * cos(a2) + 27151235630615 * cos(2 * a2)) * sin(
                    2 * a1) + 9604 * (198140244321 + 207335420978 * cos(a1) + 53340740239 * cos(2 * a1)) * sin(
                    a2) + 13072332995 * (5126 + 2401 * cos(a1) + 2077 * cos(2 * a1)) * sin(2 * a2))) / (
                        6950344987924878 + 4433052168153896 * cos(a2) + 272004488 * cos(a1) * (
                        16297717 + 4667544 * cos(a2) - 2295085 * cos(2 * a2)) + 58516346103350 * cos(
                    2 * a2) - 20770 * cos(2 * a1) * (-2817349355 + 30056495924 * cos(a2) + 17187045685 * cos(
                    2 * a2)) - 829341683912 * sin(a1) * (4802 * sin(a2) + 2077 * sin(2 * a2)) - 4154 * sin(
                    2 * a2) * (414670841956 * sin(a2) + 158422965851 * sin(2 * a2)))
            ],

            [
                (17 * (306147010883749 * cos(a2) + 13072332995 * cos(2 * a1) * (
                        3049 + 2401 * cos(a2)) + 118841571190429 * cos(2 * a2) + 4802 * cos(a1) * (
                               166502931985 + 198291271752 * cos(a2) + 40174617743 * cos(2 * a2)) - 2401 * (
                               -66097090584 + 13072332995 * sin(2 * a1) * sin(a2) - 34006164050 * sin(a1) * sin(
                           2 * a2)))) / (6950344987924878 + 4433052168153896 * cos(a2) + 272004488 * cos(a1) * (
                        16297717 + 4667544 * cos(a2) - 2295085 * cos(2 * a2)) + 58516346103350 * cos(
                    2 * a2) - 20770 * cos(2 * a1) * (-2817349355 + 30056495924 * cos(a2) + 17187045685 * cos(
                    2 * a2)) - 829341683912 * sin(a1) * (4802 * sin(a2) + 2077 * sin(2 * a2)) - 4154 * sin(2 * a1) * (
                                                 414670841956 * sin(a2) + 158422965851 * sin(2 * a2))),

                (17 * (158699114492184 + 306147010883749 * cos(a1) + 118841571190429 * cos(2 * a1) + 4802 * (
                        166502931985 + 198291271752 * cos(a1) + 40174617743 * cos(2 * a1)) * cos(
                    a2) + 13072332995 * (3049 + 2401 * cos(a1)) * cos(2 * a2) + 81648799884050 * sin(2 * a1) * sin(
                    a2) - 31386671520995 * sin(a1) * sin(2 * a2))) / (
                        6950344987924878 + 4433052168153896 * cos(a2) + 272004488 * cos(a1) * (
                        16297717 + 4667544 * cos(a2) - 2295085 * cos(2 * a2)) + 58516346103350 * cos(
                    2 * a2) - 20770 * cos(2 * a1) * (-2817349355 + 30056495924 * cos(a2) + 17187045685 * cos(
                    2 * a2)) - 829341683912 * sin(a1) * (4802 * sin(a2) + 2077 * sin(2 * a2)) - 4154 * sin(
                    2 * a1) * (414670841956 * sin(a2) + 158422965851 * sin(2 * a2)))
            ],

            [
                (952862342614013 + 1108263042038474 * cos(a1) + 27151235630615 * cos(2 * a1) - 293352012228338 * cos(
                    a1 - 2 * a2) - 339113231275994 * cos(a1 - a2) - 311961995645379 * cos(
                    2 * a2) + 656511460260362 * cos(a1 + a2) + 27151235630615 * cos(
                    2 * (a1 + a2)) + 137283657142968 * cos(a1 + 2 * a2)) / (
                        3475172493962439 + 2216526084076948 * cos(a2) + 136002244 * cos(a1) * (
                        16297717 + 4667544 * cos(a2) - 2295085 * cos(2 * a2)) + 29258173051675 * cos(
                    2 * a2) - 10385 * cos(2 * a1) * (-2817349355 + 30056495924 * cos(a2) + 17187045685 * cos(
                    2 * a2)) - 414670841956 * sin(a1) * (4802 * sin(a2) + 2077 * sin(2 * a2)) - 2077 * sin(
                    2 * a1) * (414670841956 * sin(a2) + 158422965851 * sin(2 * a2))),

                (952862342614013 - 311961995645379 * cos(2 * a1) + 68001122 * (
                        16297717 + 4667544 * cos(a1) - 2295085 * cos(2 * a1)) * cos(a2) + 54302471261230 * (
                     cos(a1)) ** 2 * cos(2 * a2) - 207335420978 * (4802 * sin(a1) + 2077 * sin(2 * a1)) * sin(
                    a2) - 27151235630615 * sin(2 * a1) * sin(2 * a2)) / (
                        -3475172493962439 - 29258173051675 * cos(2 * a1) + 136002244 * (
                        -16297717 + 2295085 * cos(2 * a1)) * cos(a2) + 51925 * (
                                -563469871 + 3437409137 * cos(2 * a1)) * cos(2 * a2) + 136002244 * cos(a1) * (
                                -16297717 - 4667544 * cos(a2) + 2295085 * cos(2 * a2)) + 414670841956 * (
                                4802 * sin(a1) + 2077 * sin(2 * a1)) * sin(a2) + 2077 * (
                                414670841956 * sin(a1) + 158422965851 * sin(2 * a1)) * sin(2 * a2))
            ]
        ])


    def M(self, theta, a1, a2, da1, da2):
        """
        :return: the 5 * 1 dv/dt matrix: xdot, ydot, thetadot, a1dot, a2dot
        """
        da = np.array([[da1],
                       [da2]])
        f = self.TeLg(theta) @ (self.J(a1, a2) @ da)
        xdot = f[0, 0]
        ydot = f[1, 0]
        thetadot = f[2, 0]
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
                a1_t = abs((self.a_lower - self.a1) / a1dot)
            elif a1 > self.a_upper:
                a1_t = abs((self.a_upper - self.a1) / a1dot)
            if a2 < self.a_lower:
                a2_t = abs((self.a_lower - self.a2) / a2dot)
            elif a2 > self.a_upper:
                a2_t = abs((self.a_upper - self.a2) / a2dot)

            # print('a1t: {x}, a2t: {y}'.format(x=a1_t, y= a2_t))

            # need to make 2 moves
            if abs(a1_t - a2_t) > 0.0000001:
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
            c3 = (t1 + t2) / (self.t_interval * self.timestep)

        if t1 + t2 == 0:
            c1 = 0
            c2 = 0
        else:
            c1 = t1 / (t1 + t2)
            c2 = t2 / (t1 + t2)
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
    robot = IdealFluidSwimmer(a1=0, a2=0, t_interval=1)
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
    time = [0]
    a1 = [robot.a1]
    a2 = [robot.a2]
    robot_params = []
    print('initial x y theta a1 a2: ', robot.x, robot.y, robot.theta, robot.a1, robot.a2)
    for t in range(25):
        print(t + 1, 'th iteration')
        a1dot = 1 / 3 * cos(t / 5)
        a2dot = -1 / 3 * sin(t / 5)
        action = (a1dot, a2dot)
        robot.move(action)
        print('action taken(a1dot, a2dot): ', action)
        print('robot x y theta a1 a2: ', robot.x, robot.y, robot.theta, robot.a1, robot.a2)
        x_pos.append(robot.x)
        y_pos.append(robot.y)
        thetas.append(robot.theta)
        time.append(t + 1)
        a1.append(robot.a1)
        a2.append(robot.a2)
        robot_param = [robot.x,
                       robot.y,
                       robot.theta,
                       float(robot.a1),
                       float(robot.a2),
                       robot.a1dot,
                       robot.a2dot]
        robot_params.append(robot_param)

    generate_csv(robot_params, "results/RobotTestResults/SwimmerIdealFluid/" + "result.csv")

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
