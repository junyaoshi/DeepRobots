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
#from utils.csv_generator import generate_csv
import csv
import time


class IdealFluidSwimmer(object):

    def __init__(self,
                 x=0.0,
                 y=0.0,
                 theta=0.0,
                 ahead=0.0,
                 atail=0.0,
                 a_upper=pi / 2,
                 a_lower=-pi / 2,
                 no_joint_limit=False,
                 a=6.0,
                 b=2.0,
                 t_interval=0.25,
                 timestep=1):
        """
        :param x: robot's initial x- displacement
        :param y: robot's initial y- displacement
        :param theta: robot's initial angle
        :param ahead: joint angle of head link (right of body when th=0)
        :param atail: joint angle of tail link (left of body when th=0)
        :param a_upper: upper joint limit, None if no limit, a_upper and a_lower must be both None in that case
        :param a_lower: lower joint limit, None if no limit, a_upper and a_lower must be both None in that case
        :param a: radius of link semi-major axis
        :param b: radius of link semi-minor axis
        :param t_interval: discretization of time
        """

        self.init_x = x
        self.init_y = y
        self.init_theta = theta
        self.init_ahead = ahead
        self.init_atail = atail

        self.x = x
        self.y = y
        self.theta = theta
        self.theta_displacement = 0  # for theta reward functions
        self.ahead = ahead
        self.atail = atail
        self.a_upper = a_upper
        self.a_lower = a_lower
        self._no_joint_limit = no_joint_limit

        self.aheaddot = 0
        self.ataildot = 0
        self.time = 0

        # constants
        self.t_interval = t_interval
        self.timestep = timestep
        self.a = a
        self.b = b

        self.state = (self.theta, self.ahead, self.atail)

    def reset(self):
        self.x = self.init_x
        self.y = self.init_y
        self.theta = self.init_theta
        self.ahead = self.init_ahead
        self.atail = self.init_atail
        self.state = (self.theta, self.ahead, self.atail)

        self.theta_displacement = 0
        self.aheaddot = 0
        self.ataildot = 0
        self.time = 0
        return self.state

    # mutator methods
    def set_state(self, theta, ahead, atail):
        self.theta, self.ahead, self.atail = theta, ahead, atail
        self.state = (theta, ahead, atail)
        return self.state

    # accessor methods
    def get_position(self):
        return self.x, self.y

    def randomize_state(self):
        self.theta = random.uniform(-pi, pi)
        if self._no_joint_limit:
            self.ahead = random.uniform(-pi, pi)
            self.atail = random.uniform(-pi, pi)
        else:
            self.ahead = random.uniform(self.a_lower, self.a_upper)
            self.atail = random.uniform(self.a_lower, self.a_upper)
        self.state = (self.theta, self.ahead, self.atail)
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

    def J(self, ahead, atail):
        """
        :return: the Jacobian matrix A_swim given joint angles
        """
        a = self.a
        b = self.b
        return np.array([[(a*((30*a**8 + 74*a**6*b**2 + 26*a**4*b**4 + 6*a**2*b**6)*
        sin(ahead) + (-2*a**8 + 21*a**6*b**2 + 5*a**4*b**4 -
          a**2*b**6 + b**8)*sin(2*ahead) +
       8*a**8*sin(ahead - 2*atail) +
       8*a**6*b**2*sin(ahead - 2*atail) -
             16*a**4*b**4*sin(ahead - 2*atail) +
       24*a**8*sin(ahead - atail) +
       48*a**6*b**2*sin(ahead - atail) -
       19*a**8*sin(atail) -
       21*a**6*b**2*sin(atail) +
       3*a**4*b**4*sin(atail) -
       3*a**2*b**6*sin(atail) -
       10*a**8*sin(2*atail) +
             5*a**6*b**2*sin(2*atail) +
       5*a**4*b**4*sin(2*atail) -
       a**2*b**6*sin(2*atail) + b**8*sin(2*atail) +
       8*a**8*sin(ahead + atail) +
       16*a**6*b**2*sin(ahead + atail) -
       a**8*sin(2*(ahead + atail)) +
             12*a**6*b**2*sin(2*(ahead + atail)) -
       14*a**4*b**4*sin(2*(ahead + atail)) +
       4*a**2*b**6*sin(2*(ahead + atail)) -
       b**8*sin(2*(ahead + atail)) -
       a**8*sin(2*ahead + atail) +
       11*a**6*b**2*sin(2*ahead + atail) -
             3*a**4*b**4*sin(2*ahead + atail) +
       a**2*b**6*sin(2*ahead + atail) +
       2*a**2*(a**6 + a**4*b**2 - 5*a**2*b**4 - b**6)*
        sin(ahead + 2*atail)))/((-(3*a**2 +
          b**2))*(11*a**6 + 93*a**4*b**2 + 7*a**2*b**4 + 9*b**6) +
          (11*a**8 - 92*a**6*b**2 + 74*a**4*b**4 + 4*a**2*b**6 + 3*b**8)*
      cos(2*ahead) -
     288*a**6*b**2*cos(ahead)**2*
      cos(atail) + (a - b)*(a + b)*(11*a**6 - 81*a**4*b**2 -
        7*a**2*b**4 -
        3*b**6 + (11*a**6 - 49*a**4*b**2 + 25*a**2*b**4 - 3*b**6)*
         cos(2*ahead))*
            cos(2*atail) +
     16*a**4*cos(ahead)*(-a**4 - 13*a**2*b**2 - 4*b**4 -
        6*a**2*b**2*cos(atail) + (a**4 - 5*a**2*b**2 + 4*b**4)*
         cos(2*atail)) -
     32*a**4*(a**2 + 2*b**2)**2*cos(atail)*
      sin(ahead)**2 +
     32*a**4*(a**2 +
        2*b**2)*(a**2 + (a - b)*(a + b)*cos(ahead))*
      sin(ahead)*sin(atail) +
     2*(a - b)*(a +
        b)*(8*(a**6 + 2*a**4*b**2) + (5*a**6 + 17*a**4*b**2 - 9*a**2*b**4 +
           3*b**6)*cos(ahead))*sin(ahead)*
      sin(2*atail)),

     (a*((-a**2)*(19*a**6 + 21*a**4*b**2 - 3*a**2*b**4 + 3*b**6 +
          16*a**4*(a**2 + 2*b**2)*
           cos(atail) + (a**6 - 11*a**4*b**2 + 3*a**2*b**4 -
             b**6)*cos(2*atail))*sin(ahead) -
             (2*a**2*(3*a**6 + 3*a**4*b**2 - 3*a**2*b**4 + b**6)*
           cos(atail) + (a - b)*(a + b)*(10*a**6 +
             5*a**4*b**2 +
             b**6 + (a**6 - 11*a**4*b**2 + 3*a**2*b**4 - b**6)*
              cos(2*atail)))*sin(2*ahead) +
       2*a**2*(15*a**6 + 37*a**4*b**2 + 13*a**2*b**4 + 3*b**6 +
          16*a**4*(a**2 + 2*b**2)*
           cos(ahead) + (5*a**6 + 5*a**4*b**2 - 13*a**2*b**4 -
             b**6)*cos(2*ahead))*sin(atail) -
             (a**6 - 11*a**4*b**2 + 3*a**2*b**4 - b**6)*(2*a**2 + b**2 +
          a**2*cos(ahead) + (a - b)*(a + b)*
           cos(2*ahead))*sin(2*atail)))/((3*a**2 +
        b**2)*(11*a**6 + 93*a**4*b**2 + 7*a**2*b**4 + 9*b**6) -
          (11*a**8 - 92*a**6*b**2 + 74*a**4*b**4 + 4*a**2*b**6 + 3*b**8)*
      cos(2*ahead) +
     288*a**6*b**2*cos(ahead)**2*
      cos(atail) - (a - b)*(a + b)*(11*a**6 - 81*a**4*b**2 -
        7*a**2*b**4 -
        3*b**6 + (11*a**6 - 49*a**4*b**2 + 25*a**2*b**4 - 3*b**6)*
         cos(2*ahead))*
            cos(2*atail) +
     16*a**4*cos(ahead)*(a**4 + 13*a**2*b**2 + 4*b**4 +
        6*a**2*b**2*cos(atail) - (a**4 - 5*a**2*b**2 + 4*b**4)*
         cos(2*atail)) +
     32*a**4*(a**2 + 2*b**2)**2*cos(atail)*
      sin(ahead)**2 -
     32*a**4*(a**2 +
        2*b**2)*(a**2 + (a - b)*(a + b)*cos(ahead))*
      sin(ahead)*sin(atail) -
     2*(a - b)*(a +
        b)*(8*(a**6 + 2*a**4*b**2) + (5*a**6 + 17*a**4*b**2 - 9*a**2*b**4 +
           3*b**6)*cos(ahead))*sin(ahead)*
      sin(2*atail))],

   [(a*(24*a**6*
        b**2 + (-a**8 + 9*a**6*b**2 + 19*a**4*b**4 - 5*a**2*b**6 + 2*b**8)*
        cos(2*ahead) +
       a**2*(a**6 + 43*a**4*b**2 - 9*a**2*b**4 +
          5*b**6 + (-a**6 + 11*a**4*b**2 - 3*a**2*b**4 + b**6)*
           cos(2*ahead))*cos(atail) +
             (a - b)*(a + b)*(a**6 + 16*a**4*b**2 - 3*a**2*b**4 + 2*b**6)*
        cos(2*atail) +
       2*a**2*cos(ahead)*(a**6 + 55*a**4*b**2 + 7*a**2*b**4 +
          5*b**6 + 72*a**4*b**2*
           cos(atail) + (-a**6 + 23*a**4*b**2 - 19*a**2*b**4 +
             b**6)*cos(2*atail)) +
       a**2*(a**6 - 11*a**4*b**2 + 3*a**2*b**4 - b**6)*
        sin(2*ahead)*sin(atail) +
       2*a**2*(a**6 + a**4*b**2 - 5*a**2*b**4 - b**6)*sin(ahead)*
        sin(2*atail)))/((3*a**2 + b**2)*(11*a**6 +
        93*a**4*b**2 + 7*a**2*b**4 + 9*b**6) -
          (11*a**8 - 92*a**6*b**2 + 74*a**4*b**4 + 4*a**2*b**6 + 3*b**8)*
      cos(2*ahead) +
     288*a**6*b**2*cos(ahead)**2*
      cos(atail) - (a - b)*(a + b)*(11*a**6 - 81*a**4*b**2 -
        7*a**2*b**4 -
        3*b**6 + (11*a**6 - 49*a**4*b**2 + 25*a**2*b**4 - 3*b**6)*
         cos(2*ahead))*
            cos(2*atail) +
     16*a**4*cos(ahead)*(a**4 + 13*a**2*b**2 + 4*b**4 +
        6*a**2*b**2*cos(atail) - (a**4 - 5*a**2*b**2 + 4*b**4)*
         cos(2*atail)) +
     32*a**4*(a**2 + 2*b**2)**2*cos(atail)*
      sin(ahead)**2 -
     32*a**4*(a**2 +
        2*b**2)*(a**2 + (a - b)*(a + b)*cos(ahead))*
      sin(ahead)*sin(atail) -
     2*(a - b)*(a +
        b)*(8*(a**6 + 2*a**4*b**2) + (5*a**6 + 17*a**4*b**2 - 9*a**2*b**4 +
           3*b**6)*cos(ahead))*sin(ahead)*
      sin(2*atail)),

     (-24*a**7*b**2 -
     a**3*(a**6 + 43*a**4*b**2 - 9*a**2*b**4 + 5*b**6)*
      cos(ahead) -
     a*(a - b)*(a + b)*(a**6 + 16*a**4*b**2 - 3*a**2*b**4 + 2*b**6)*
      cos(2*ahead) +
     2*a**3*(-a**6 - 55*a**4*b**2 - 7*a**2*b**4 - 5*b**6 -
        72*a**4*b**2*
         cos(ahead) + (a**6 - 23*a**4*b**2 + 19*a**2*b**4 - b**6)*
         cos(2*ahead))*cos(atail) +
     a*(a**6 - 11*a**4*b**2 + 3*a**2*b**4 - b**6)*(a**2 + 2*b**2 +
        a**2*cos(ahead))*cos(2*atail) -
     2*a**3*(a**6 + a**4*b**2 - 5*a**2*b**4 - b**6)*sin(2*ahead)*
      sin(atail) +
     a**3*(-a**6 + 11*a**4*b**2 - 3*a**2*b**4 + b**6)*sin(ahead)*
      sin(2*atail))/((-(3*a**2 + b**2))*(11*a**6 +
        93*a**4*b**2 + 7*a**2*b**4 + 9*b**6) +
          (11*a**8 - 92*a**6*b**2 + 74*a**4*b**4 + 4*a**2*b**6 + 3*b**8)*
      cos(2*ahead) -
     288*a**6*b**2*cos(ahead)**2*
      cos(atail) + (a - b)*(a + b)*(11*a**6 - 81*a**4*b**2 -
        7*a**2*b**4 -
        3*b**6 + (11*a**6 - 49*a**4*b**2 + 25*a**2*b**4 - 3*b**6)*
         cos(2*ahead))*
            cos(2*atail) +
     16*a**4*cos(ahead)*(-a**4 - 13*a**2*b**2 - 4*b**4 -
        6*a**2*b**2*cos(atail) + (a**4 - 5*a**2*b**2 + 4*b**4)*
         cos(2*atail)) -
     32*a**4*(a**2 + 2*b**2)**2*cos(atail)*
      sin(ahead)**2 +
     32*a**4*(a**2 +
        2*b**2)*(a**2 + (a - b)*(a + b)*cos(ahead))*
      sin(ahead)*sin(atail) +
     2*(a - b)*(a +
        b)*(8*(a**6 + 2*a**4*b**2) + (5*a**6 + 17*a**4*b**2 - 9*a**2*b**4 +
           3*b**6)*cos(ahead))*sin(ahead)*
      sin(2*atail))],

   [(11*a**8 + 70*a**6*b**2 + 6*a**4*b**4 + 6*a**2*b**6 + 3*b**8 -
     2*(a**8 - 12*a**6*b**2 + 14*a**4*b**4 - 4*a**2*b**6 + b**8)*
      cos(2*ahead)*
      cos(atail)**2 - (a**2 - b**2)**2*(9*a**4 - 2*a**2*b**2 +
        b**4)*cos(2*atail) +
     8*a**4*cos(ahead)*(a**4 + 13*a**2*b**2 + 4*b**4 +
        6*a**2*b**2*cos(atail) - (a**4 - 5*a**2*b**2 + 4*b**4)*
         cos(2*atail)) -
     16*a**4*(a**2 +
        2*b**2)*(a**2 + (a - b)*(a + b)*cos(atail))*
      sin(ahead)*sin(atail) +
          (a**8 - 12*a**6*b**2 + 14*a**4*b**4 - 4*a**2*b**6 + b**8)*
      sin(2*ahead)*
      sin(2*atail))/((3*a**2 + b**2)*(11*a**6 + 93*a**4*b**2 +
        7*a**2*b**4 + 9*b**6) - (11*a**8 - 92*a**6*b**2 + 74*a**4*b**4 +
        4*a**2*b**6 + 3*b**8)*cos(2*ahead) +
     288*a**6*b**2*cos(ahead)**2*
      cos(atail) - (a - b)*(a + b)*(11*a**6 - 81*a**4*b**2 -
        7*a**2*b**4 -
        3*b**6 + (11*a**6 - 49*a**4*b**2 + 25*a**2*b**4 - 3*b**6)*
         cos(2*ahead))*cos(2*atail) +
     16*a**4*cos(ahead)*(a**4 + 13*a**2*b**2 + 4*b**4 +
        6*a**2*b**2*cos(atail) - (a**4 - 5*a**2*b**2 + 4*b**4)*
         cos(2*atail)) +
     32*a**4*(a**2 + 2*b**2)**2*cos(atail)*
      sin(ahead)**2 -
     32*a**4*(a**2 +
        2*b**2)*(a**2 + (a - b)*(a + b)*cos(ahead))*
      sin(ahead)*sin(atail) -
     2*(a - b)*(a +
        b)*(8*(a**6 + 2*a**4*b**2) + (5*a**6 + 17*a**4*b**2 - 9*a**2*b**4 +
           3*b**6)*cos(ahead))*sin(ahead)*
      sin(2*atail)),

     (11*a**8 + 70*a**6*b**2 + 6*a**4*b**4 + 6*a**2*b**6 + 3*b**8 +
     8*a**4*(a**4 + 13*a**2*b**2 + 4*b**4 +
        6*a**2*b**2*cos(ahead))*
      cos(atail) - (a - b)*(a + b)*
      cos(2*ahead)*((a - b)*(a + b)*(9*a**4 - 2*a**2*b**2 + b**4) +
        8*a**4*(a**2 - 4*b**2)*cos(atail)) -
     2*(a**8 - 12*a**6*b**2 + 14*a**4*b**4 - 4*a**2*b**6 + b**8)*
      cos(ahead)**2*cos(2*atail) -
     16*a**4*(a**2 +
        2*b**2)*(a**2 + (a - b)*(a + b)*cos(ahead))*
      sin(ahead)*sin(atail) +
          (a**8 - 12*a**6*b**2 + 14*a**4*b**4 - 4*a**2*b**6 + b**8)*
      sin(2*ahead)*
      sin(2*atail))/((-(3*a**2 + b**2))*(11*a**6 +
        93*a**4*b**2 + 7*a**2*b**4 + 9*b**6) + (11*a**8 - 92*a**6*b**2 +
        74*a**4*b**4 + 4*a**2*b**6 + 3*b**8)*cos(2*ahead) -
     288*a**6*b**2*cos(ahead)**2*
      cos(atail) + (a - b)*(a + b)*(11*a**6 - 81*a**4*b**2 -
        7*a**2*b**4 -
        3*b**6 + (11*a**6 - 49*a**4*b**2 + 25*a**2*b**4 - 3*b**6)*
         cos(2*ahead))*cos(2*atail) +
     16*a**4*cos(ahead)*(-a**4 - 13*a**2*b**2 - 4*b**4 -
        6*a**2*b**2*cos(atail) + (a**4 - 5*a**2*b**2 + 4*b**4)*
         cos(2*atail)) -
     32*a**4*(a**2 + 2*b**2)**2*cos(atail)*
      sin(ahead)**2 +
     32*a**4*(a**2 +
        2*b**2)*(a**2 + (a - b)*(a + b)*cos(ahead))*
      sin(ahead)*sin(atail) +
     2*(a - b)*(a +
        b)*(8*(a**6 + 2*a**4*b**2) + (5*a**6 + 17*a**4*b**2 - 9*a**2*b**4 +
           3*b**6)*cos(ahead))*sin(ahead)*
      sin(2*atail))]])


    def M(self, theta, ahead, atail, dahead, datail):
        """
        :return: the 5 * 1 dv/dt matrix: xdot, ydot, thetadot, aheaddot, ataildot
        """
        da = np.array([[dahead],
                       [datail]])
        f = -self.TeLg(theta) @ (self.J(ahead, atail) @ da)
        xdot = f[0, 0]
        ydot = f[1, 0]
        thetadot = f[2, 0]
        M = [xdot, ydot, thetadot, dahead, datail]
        return M

    def robot(self, v, t, dahead, datail):
        """
        :return: function used for odeint integration
        """
        _, _, theta, ahead, atail = v
        # print('ahead atail:', ahead, atail)
        dvdt = self.M(theta, ahead, atail, dahead, datail)
        return dvdt

    def perform_integration(self, action, t_interval):
        """
        :return: perform integration of ode, return the final displacements and x-velocity
        """

        if t_interval == 0:
            return self.x, self.y, self.theta, self.ahead, self.atail
        aheaddot, ataildot = action
        v0 = [self.x, self.y, self.theta, self.ahead, self.atail]
        t = np.linspace(0, t_interval, 11)
        sol = odeint(self.robot, v0, t, args=(aheaddot, ataildot))
        x, y, theta, ahead, atail = sol[-1]
        return x, y, theta, ahead, atail

    def move(self, action, timestep=1, enforce_angle_limits=True):
        """
        Implementation of Equation 9
        given the joint velocities of the 2 controlled joints
        and the number of discretized time intervals
        move the robot accordingly
        :param aheaddot: joint velocity of head joint
        :param ataildot: joint velocity of tail joint
        :param timestep: number of time intervals
        :return: new state of the robot
        """
        self.timestep = timestep

        if enforce_angle_limits:
            self.check_angles()

        aheaddot, ataildot = action
        # print('action: ', action)
        t = self.timestep * self.t_interval
        # print('t: ', t)
        ahead = self.ahead + aheaddot * t
        atail = self.atail + ataildot * t
        # print('ds: ', x, y, theta, ahead, atail)

        if self._no_joint_limit:
            enforce_angle_limits = False

        d_theta = 0

        if enforce_angle_limits:

            # print('ahead: {x}, atail: {y}'.format(x=self.ahead + dahead, y=self.atail+datail))

            # update integration time for each angle if necessary
            ahead_t, atail_t = t, t
            if ahead < self.a_lower:
                ahead_t = abs((self.a_lower - self.ahead) / aheaddot)
            elif ahead > self.a_upper:
                ahead_t = abs((self.a_upper - self.ahead) / aheaddot)
            if atail < self.a_lower:
                atail_t = abs((self.a_lower - self.atail) / ataildot)
            elif atail > self.a_upper:
                atail_t = abs((self.a_upper - self.atail) / ataildot)

            # print('aheadt: {x}, atailt: {y}'.format(x=ahead_t, y= atail_t))

            # need to make 2 moves
            if abs(ahead_t - atail_t) > 0.0000001:
                # print(ahead_t, atail_t)
                t1 = min(ahead_t, atail_t)
                old_theta = self.theta
                x, y, theta, ahead, atail = self.perform_integration(action, t1)
                d_theta += (theta - old_theta)
                self.update_params(x, y, theta, ahead, atail)
                if atail_t > ahead_t:
                    t2 = atail_t - ahead_t
                    action = (0, ataildot)
                    old_theta = self.theta
                    x, y, theta, ahead, atail = self.perform_integration(action, t2)
                    d_theta += (theta - old_theta)
                    self.update_params(x, y, theta, ahead, atail)
                    self.update_alpha_dots(aheaddot, ataildot, t1, 0, ataildot, t2)
                else:
                    t2 = ahead_t - atail_t
                    action = (aheaddot, 0)
                    old_theta = self.theta
                    x, y, theta, ahead, atail = self.perform_integration(action, t2)
                    d_theta += (theta - old_theta)
                    self.update_params(x, y, theta, ahead, atail)
                    self.update_alpha_dots(aheaddot, ataildot, t1, aheaddot, 0, t2)

            # only one move is needed
            else:
                # print('b')
                if t != ahead_t:
                    t = ahead_t
                old_theta = self.theta
                x, y, theta, ahead, atail = self.perform_integration(action, t)
                d_theta += (theta - old_theta)
                self.update_params(x, y, theta, ahead, atail)
                self.update_alpha_dots(aheaddot, ataildot, t)
        else:
            # print('a')
            old_theta = self.theta
            x, y, theta, ahead, atail = self.perform_integration(action, t)
            d_theta += (theta - old_theta)
            self.update_params(x, y, theta, ahead, atail, enforce_angle_limits=False)
            self.update_alpha_dots(aheaddot, ataildot, t)

        self.theta_displacement = d_theta
        self.state = (self.theta, self.ahead, self.atail)

        return self.state

    def enforce_angle_range(self, angle_name):
        if angle_name == 'theta':
            angle = self.theta
        elif angle_name == 'ahead':
            angle = self.ahead
        else:
            angle = self.atail
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
        elif angle_name == 'ahead':
            self.ahead = angle
        elif angle_name == 'atail':
            self.atail = angle

    def update_alpha_dots(self, aheaddot1, ataildot1, t1=None, aheaddot2=0, ataildot2=0, t2=None):

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
        self.aheaddot = (c1 * aheaddot1 + c2 * aheaddot2) * c3
        self.ataildot = (c1 * ataildot1 + c2 * ataildot2) * c3

    def update_params(self, x, y, theta, ahead, atail, enforce_angle_limits=True):
        # update robot variables
        self.x = x
        self.y = y
        self.theta = theta
        self.enforce_angle_range('theta')

        self.ahead = ahead
        if not enforce_angle_limits:
            self.enforce_angle_range('ahead')

        self.atail = atail
        if not enforce_angle_limits:
            self.enforce_angle_range('atail')

        self.round_angles_to_limits()

    def round_angles_to_limits(self, tolerance=0.000000001):
        if self._no_joint_limit:
            return
        if abs(self.ahead - self.a_upper) < tolerance:
            self.ahead = self.a_upper
        elif abs(self.ahead - self.a_lower) < tolerance:
            self.ahead = self.a_lower
        if abs(self.atail - self.a_lower) < tolerance:
            self.atail = self.a_lower
        elif abs(self.atail - self.a_upper) < tolerance:
            self.atail = self.a_upper

    def check_angles(self):
        if self._no_joint_limit:
            return
        if self.ahead < self.a_lower or self.ahead > self.a_upper:
            raise Exception('ahead out of limit: {x}'.format(x=self.ahead))
        if self.atail < self.a_lower or self.atail > self.a_upper:
            raise Exception('atail out of limit: {x}'.format(x=self.atail))

    def print_state(self):
        """
        print the current state
        :return: None
        """
        print('\nthe current state is: ' + str(self.state) + '\n')


if __name__ == "__main__":

    # create a robot simulation
    robot = IdealFluidSwimmer(ahead=0, atail=-5/3, no_joint_limit=True, t_interval=1)
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
    times = [0]
    ahead = [robot.ahead]
    atail = [robot.atail]
    robot_params = []
    print('initial x y theta ahead atail: ', robot.x, robot.y, robot.theta, robot.ahead, robot.atail)
    for t in range(100):
        print(t + 1, 'th iteration')
        aheaddot = 1 / 3 * cos(t / 5)
        ataildot = 1 / 3 * sin(t / 5)
        action = (aheaddot, ataildot)
        start = time.time()
        robot.move(action)
        end = time.time()
        print("Move time: {}".format(end - start))
        print('action taken(aheaddot, ataildot): ', action)
        print('robot x y theta ahead atail: ', robot.x, robot.y, robot.theta, robot.ahead, robot.atail)
        x_pos.append(robot.x)
        y_pos.append(robot.y)
        thetas.append(robot.theta)
        times.append(t + 1)
        ahead.append(robot.ahead)
        atail.append(robot.atail)
        robot_param = [robot.x,
                       robot.y,
                       robot.theta,
                       float(robot.ahead),
                       float(robot.atail),
                       robot.aheaddot,
                       robot.ataildot]
        robot_params.append(robot_param)

    #generate_csv(robot_params, "results/RobotTestResults/SwimmerIdealFluid/" + "result.csv")
    with open("out.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(robot_params)

    # view results
    # print('x positions are: ' + str(x_pos))
    # print('y positions are: ' + str(y_pos))
    # print('thetas are: ' + str(thetas))

    plt.plot(x_pos, y_pos)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('y vs x')
    plt.show()

    plt.plot(times, ahead)
    plt.ylabel('ahead displacements')
    plt.show()

    plt.plot(times, atail)
    plt.ylabel('atail displacements')
    plt.show()

    plt.plot(times, x_pos)
    plt.ylabel('x positions')
    plt.show()

    plt.plot(times, y_pos)
    plt.ylabel('y positions')
    plt.show()

    plt.plot(times, thetas)
    plt.ylabel('thetas')
    plt.show()
    plt.close()
