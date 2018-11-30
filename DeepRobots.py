from math import cos, sin, atan, pi
import numpy as np
from scipy import integrate


class ThreeLinkRobot(object):

    def __init__(self, x, y, theta, alpha1, alpha2, link_length, time_interval, angle_interval):
        """
        :param x: robot's initial x- displacement
        :param y: robot's initial x- displacement
        :param link_length: length of every robot link
        :param time_interval: discretization of time
        """
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.x = x
        self.y = y
        self.theta = theta

        ### find a way to implement these functions without depending on other variables in initialization
        ### /updating them in member functions
        self.state = (self.x, self.y, self.theta)
        self.epsilon_x = 0
        self.epsilon_y = 0
        self.epsilon_theta = 0
        self.body_v = np.array([[self.epsilon_x],
                                [self.epsilon_y],
                                [self.epsilon_theta]])
        self.inertial_v = self.body_to_inertial_v()

        # constants
        self.time_interval = time_interval
        self.R = link_length
        self.angle_interval = angle_interval # need implementation

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

    def body_to_inertial_v(self, theta):
        """
        convert body velocity to inertial velocity
        :param body_v: a 1*3 array that denotes epsilon_x, epsilon_y, epsilon_theta
        :param theta: the inertial angle in radians
        :return: a 1*3 array that denotes x_dot, y_dot, theta_dot converted from body_v
        """
        inertial_v = np.matmul(self.TeLg(theta), self.body_v)
        return inertial_v


    ### use passed in values instead of self. values
    def inertial_move(self, inertial_v, timestep):
        """
        move the robot for timestep number of discretized time intervals
        according to its inertial velocities
        :param timestep: number time intervals
        :return: the new state of robot
        """
        initial_state = self.state
        def integrand(a, x):
            return x
        x_dot = body_v[0][0]
        y_dot = self.inertial_v[1][0]
        theta_dot = self.inertial_v[2][0]
        print(x_dot, y_dot, theta_dot)
        self.x += integrate.quad(integrand, 0, timestep, args=(x_dot))[0]
        self.x += integrate.quad(integrand, 0, timestep, args=(y_dot))[0]
        self.theta += integrate.quad(integrand, 0, timestep, args=(theta_dot))[0]
        self.state = (self.x, self.y, self.theta)

        return self.state

    ### change incorrectly written a1dot and a2dot to a1 and a2
    ### don't forget to update a1 and a2
    def move(self, adot1, adot2, timestep):

        ### a1dot, a2dot
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

        ### change adot1, adot2, to alpha1, alpha2

        D = (2/self.R) * (sin(adot2)-sin(adot1)-sin(adot1-adot2))
        A = np.array([[cos(adot1)+cos(adot1-adot2),              1+cos(adot1)],
                      [0,                                        0],
                      [(2/self.R)*(sin(adot1)+sin(adot1-adot2)), (2/self.R)*sin(adot1)]])

        # matrix that represents controlled joints' velocities
        adot_c = np.array([[adot1],
                           [adot2]])

        # matrix that represents body velocities
        body_v = (1/D) * np.matmul(A, adot_c)
        self.epsilon_x = body_v[0]
        self.epsilon_y = body_v[1]
        self.epsilon_theta = body_v[2]
        self.body_v = np.array([[self.epsilon_x],
                                [self.epsilon_y],
                                [self.epsilon_theta]])
        self.inertial_move(timestep)

        # update variables in initialization  here
        return self.state


    def print_state(self):
        ### change this to formatted strings
        print('the current state is: ', str(self.x), str(self.y), str(self.theta))
        print()

if __name__ == "__main__":

    # create a robot simulation
    robot = ThreeLinkRobot(0,0,3,0.1) ### need to pass in theta; when initializing, set alpha1 and alpha2 to different numbers
    robot.print_state()
    robot.move(pi/6, pi/5, 3)
    robot.print_state()

# questions:
# openAI (don't worry)
# how to keep the continuity Scott mentioned (don't worry)
# this implementation is only for proximal link, do we need other links? (implement the function)
# do I need to integrate over theta_dot? Is it already given by x and y dot? (No)
# when the input is pi/3, pi/3, D becomes 0 (singularity)
# self.state is broken (another version of implementation)
# plot graph with matplotlib (noted)








