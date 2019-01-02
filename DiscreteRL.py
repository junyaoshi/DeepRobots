# implementation of discrete RL for locomotion in x- direction
from DiscreteDeepRobots import ThreeLinkRobot
from math import pi
import numpy as np

def Qlearner(robot, alpha, epsilon, s_lower, s_upper, s_interval, a_lower, a_upper, a_interval):
    """
    :param robot: a ThreeLinkRobot object
    :param alpha: learning rate
    :param epsilon: probability of choosing random action while learning
    :param s_lower: lower limit of state space
    :param s_upoer: upper limit of state space
    :param s_interval: interval of discretized state space
    :param a_lower: lower limit of action space
    :param a_upoer: upper limit of action space
    :param a_interval: interval of discretized action space
    :return: a dictionary of (state, action) and Q-values pairs
    """

    # initialize state space
    states = set()
    for i in np.arange(-2 * pi, 2 * pi + 0.0001, pi / 64):
        for j in np.arange(-2 * pi, 2 * pi + 0.0001, pi / 64):
            for k in np.arange(-2 * pi, 2 * pi + 0.0001, pi / 64):
                states.add(0)

    Qvalues = {}
    return Qvalues

def get_state_space(lower_limit, upper_limit, interval):
    """
    :param lower_limit:
    :param upper_limit:
    :param interval:
    :return: a set of state space values in tuple format (a1, a2, theta)
    """
    upper_limit += (interval/10) # to ensure the range covers the rightmost value in the loop

    states = set()
    for i in np.arange(lower_limit, upper_limit, interval):
        for j in np.arange(lower_limit, upper_limit, interval):
            for k in np.arange(lower_limit, upper_limit, interval):
                states.add((i,j,k))
    return states

if __name__ == "__main__":
    states = get_state_space(-2*pi, 2*pi, pi)
    print(len(states))
    print(states)

