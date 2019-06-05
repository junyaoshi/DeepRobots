from math import cos, sin, pi
import numpy as np
from scipy import integrate
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
        self.theta = theta
        self.a1 = a1
        self.a2 = a2
        self.a1dot = 0
        self.a2dot = 0
        self.time = 0

        self.state = [self.theta, self.a1, self.a2]
        self.body_v = [0, 0, 0]
        self.inertial_v = [0, 0, 0]

        # constants
        self.t_interval = t_interval
        self.R = link_length
        self.a_interval = a_interval

    # mutator methods
    def set_state(self, theta, a1, a2):
        self.state = [theta, a1, a2]

    def set_body_v(self, e_x, e_y, e_theta):
        self.body_v = [e_x, e_y, e_theta]

    def set_inertial_v(self, x_dot, y_dot, theta_dot):
        self.body_v = [x_dot, y_dot, theta_dot]

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
                        [sin(theta), cos(theta),  0],
                        [0,               0,      1]])
        return arr

    def get_v(self, a1dot, a2dot):
        """
        Find the body and inertial velocity matrix of robot
        given controlled joint angle velocities
        :param a1dot: proximal joint angle velocity
        :param a2dot: distal joint angle velocity
        :return: body and inertial velocity matrix of robot
        """
        a1 = self.a1
        a2 = self.a2
        R = self.R
        theta = self.theta
        D = (2/R) * (-sin(a1)-sin(a1-a2)+sin(a2))
        A = np.array([[cos(a1)+cos(a1-a2),             1+cos(a1)],
                      [0,                                      0],
                      [(2/R)*(sin(a1)+sin(a1-a2)), (2/R)*sin(a1)]])
        adot_c = np.array([[a1dot],
                           [a2dot]])
        body_v = (1/D) * np.matmul(A, adot_c)
        inertial_v = np.matmul(self.TeLg(theta), body_v)

        # testing
        print('body v: \n' + str(body_v))
        print('inertial v: \n' + str(inertial_v))

        return body_v, inertial_v

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

        body_v, inertial_v = self.get_v(a1dot, a2dot)

        # lambdafication
        x_dot = lambda t: inertial_v[0][0]
        y_dot = lambda t: inertial_v[1][0]
        theta_dot = lambda t: inertial_v[2][0]
        _a1dot = lambda t: a1dot
        _a2dot = lambda t: a2dot

        # find the increments
        dx,_ = integrate.quad(x_dot, 0, timestep)
        dy,_ = integrate.quad(y_dot, 0, timestep)
        dtheta,_ = integrate.quad(theta_dot, 0, timestep)
        da1,_ = integrate.quad(_a1dot, 0, timestep)
        da2,_ = integrate.quad(_a2dot, 0, timestep)

        # testing
        print('the increments: ')
        print(dx, dy, dtheta, da1, da2)

        # update robot variables
        self.x += dx
        self.y += dy
        self.theta += dtheta
        self.a1 += da1
        self.a2 += da2
        self.time += timestep
        self.a1dot = a1dot
        self.a2dot = a2dot
        self.body_v = [body_v[0][0], body_v[1][0], body_v[2][0]]
        self.inertial_v = [inertial_v[0][0], inertial_v[1][0], inertial_v[2][0]]
        self.state = [self.theta, self.a1, self.a2]

        return self.state

    def print_state(self):
        """
        print the current state
        :return: None
        """
        print('\nthe current state is: ' + str(self.state) + '\n')


if __name__ == "__main__":

    # create a robot simulation
    robot = ThreeLinkRobot(x=0,y=0,theta=0,a1=pi/4,a2=-pi/4,link_length=2,t_interval=0.01,a_interval=pi/64)
    robot.print_state()
    x_pos = []
    y_pos = []
    thetas = []
    time = []
    a1 = []
    a2 = []
    for i in range(100):
        # robot.move(pi / 30, -pi / 30, 1)
        robot.move((pi/8) * cos(i), (pi/8) * cos(i+1), 1)
        robot.print_state()
        x_pos.append(robot.x)
        y_pos.append(robot.y)
        thetas.append(robot.theta)
        time.append(robot.time)
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












