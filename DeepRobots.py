from math import cos, sin, pi
import numpy as np
from scipy import integrate


class ThreeLinkRobot(object):

    def __init__(self, x, y, theta, a1, a2, link_length, t_interval, a_interval):
        """
        :param x: robot's initial x- displacement
        :param y: robot's initial y- displacement
        :param theta: robot's initial angle
        :param a1: joint angle of proximal link
        :param a2: joint angle of distal link
        :param link_length: length of every robot link
        :param time_interval: discretization of time
        :param angle_interval: discretization of joint angle
        """

        self.x = x
        self.y = y
        self.theta = theta
        self.a1 = a1
        self.a2 = a2
        self.adot1 = 0
        self.adot2 = 0

        self.state = [self.x, self.y, self.theta, self.a1, self.a2]
        self.body_v = [0, 0, 0]
        self.inertial_v = [0, 0, 0]

        # constants
        self.t_interval = t_interval
        self.R = link_length
        self.a_interval = a_interval # need implementation

    # mutator methods
    ### are these functions useless?
    '''
    def set_state(self, x, y, theta, a1, a2):
        self.state = [x, y, theta, a1, a2]

    def set_body_v(self, e_x, e_y, e_theta):
        self.body_v = [e_x, e_y, e_theta]

    def set_inertial_v(self, x_dot, y_dot, theta_dot):
        self.body_v = [x_dot, y_dot, theta_dot]
    '''

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

    '''
    def body_to_inertial_v(self, theta):
        """
        convert body velocity to inertial velocity
        :param body_v: a 1*3 array that denotes epsilon_x, epsilon_y, epsilon_theta
        :param theta: the inertial angle in radians
        :return: a 1*3 array that denotes x_dot, y_dot, theta_dot converted from body_v
        """
        inertial_v = np.matmul(self.TeLg(theta), self.body_v)
        return inertial_v
    '''

    def get_v(self, adot1, adot2):
        a1 = self.a1
        a2 = self.a2
        R = self.R
        theta = self.theta
        D = (2/R) * (-sin(a1)-sin(a1-a2)+sin(a2))
        A = np.array([[cos(a1)+cos(a1-a2),             1+cos(a1)],
                      [0,                                      0],
                      [(2/R)*(sin(a1)+sin(a1-a2)), (2/R)*sin(a1)]])
        adot_c = np.array([[adot1],
                           [adot2]])
        body_v = (1/D) * np.matmul(A, adot_c)
        inertial_v = np.matmul(self.TeLg(theta), body_v)

        # testing
        print('body v: \n' + str(body_v))
        print('inertial v: \n' + str(inertial_v))

        return body_v, inertial_v

    '''
    ### use passed in values instead of self. values
    def inertial_move(self, timestep=1):
        """
        move the robot for timestep number of discretized time intervals
        according to its inertial velocities
        :param timestep: number time intervals
        :return: the new state of robot
        """
        def integrand(a, x):
            return x
        x_dot = self.inertial_v[0]
        y_dot = self.inertial_v[1]
        theta_dot = self.inertial_v[2]

        # testing
        print(x_dot, y_dot, theta_dot)

        self.x += integrate.quad(integrand, 0, timestep, args=(x_dot))[0]
        self.x += integrate.quad(integrand, 0, timestep, args=(y_dot))[0]
        self.theta += integrate.quad(integrand, 0, timestep, args=(theta_dot))[0]
        self.state = (self.x, self.y, self.theta)

        return self.state
    '''

    def move(self, adot1, adot2, timestep):

        ### make alpha1 postive, alpha2 negative to avoid "singularity" (alpha1 = alpha2)
        """
        Implementation of Equation 9
        given the joint velocities of the 2 controlled joints
        and the number of discretized time intervals
        move the robot accordingly
        :param adot1: joint velocity of the proximal joint
        :param adot2: joint velocity of the distal joint
        :param timestep: number of time intevvals
        :return: new state of the robot
        """

        body_v, inertial_v = self.get_v(adot1, adot2)

        # lambdafication
        x_dot = lambda t: inertial_v[0][0]
        y_dot = lambda t: inertial_v[1][0]
        theta_dot = lambda t: inertial_v[2][0]
        _adot1 = lambda t: adot1
        _adot2 = lambda t: adot2

        dx,_ = integrate.quad(x_dot, 0, timestep)
        dy,_ = integrate.quad(y_dot, 0, timestep)
        dtheta,_ = integrate.quad(theta_dot, 0, timestep)
        da1,_ = integrate.quad(_adot1, 0, timestep)
        da2,_ = integrate.quad(_adot2, 0, timestep)

        # testing
        print('the increments: ')
        print(dx, dy, dtheta, da1, da2)

        # update robot variables
        self.x += dx
        self.y += dy
        self.theta += dtheta
        self.a1 += da1
        self.a2 += da2
        self.adot1 = adot1
        self.adot2 = adot2
        self.body_v = [body_v[0][0], body_v[1][0], body_v[2][0]]
        self.inertial_v = [inertial_v[0][0], inertial_v[1][0], inertial_v[2][0]]
        self.state = [self.x, self.y, self.theta, self.a1, self.a2]

        return self.state

    def print_state(self):
        print('the current state is: ' + str(self.state) + '\n')

if __name__ == "__main__":

    # create a robot simulation
    robot = ThreeLinkRobot(0,0,0,pi/4,-pi/4,3,0.1,0)
    robot.print_state()
    robot.move(pi/30, -pi/30, 3)
    robot.print_state()

# questions:
# openAI (don't worry)
# how to keep the continuity Scott mentioned (don't worry)
# this implementation is only for proximal link, do we need other links? (implement the function)
# do I need to integrate over theta_dot? Is it already given by x and y dot? (No)
# when the input is pi/3, pi/3, D becomes 0 (singularity)
# self.state is broken (another version of implementation)
# plot graph with matplotlib (noted)








