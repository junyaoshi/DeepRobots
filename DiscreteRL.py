# implementation of discrete RL for locomotion in x- direction
from DiscreteDeepRobots import ThreeLinkRobot
from math import pi, log
import random
import numpy as np
import copy


def Qlearner(robot, alpha, gamma, epsilon, theta_lower, theta_upper, theta_interval,
             a1_lower, a1_upper, a1_interval, a2_lower, a2_upper, a2_interval, a_lower, a_upper, a_interval):
    """
    :param robot: a ThreeLinkRobot object
    :param alpha: learning rate
    :param gamma: discount rate
    :param epsilon: probability of choosing random action while learning
    :param theta_lower: lower limit of theta in state space
    :param theta_upper: upper limit of theta in state space
    :param theta_interval: interval of theta values in state space
    :param a1_lower: lower limit of a1 in state space
    :param a1_upper: upper limit of a1 in state space
    :param a1_interval: interval of a1 values in state space
    :param a2_lower: lower limit of a2 in state space
    :param a2_upper: upper limit of a2 in state space
    :param a2_interval: interval of a2 values in state space
    :param a_lower: lower limit of action space
    :param a_upper: upper limit of action space
    :param a_interval: interval of discretized action space
    :return: a dictionary of (state, action) and Q-values pairs
    """

    # initialize state space, action space, and Qvalues
    print('loading state space')
    states = get_state_space(theta_lower, theta_upper, theta_interval,a1_lower,
                             a1_upper, a1_interval, a2_lower, a2_upper, a2_interval)  # state = (theta, a1, a2)
    print(len(states), 'states loaded')
    print('loading action space')
    actions = get_action_space(a_lower, a_upper, a_interval)  # action = (a1dot, a2dot)
    print(len(actions), 'actions loaded')
    Qvalues = {}
    print('initializing Qvalues')
    for state in states:
        for action in actions:
            Qvalues[(state, action)] = 0
    print(len(Qvalues.keys()), 'q-values loaded')

    # learn q-values
    print('\n\n')
    state = robot.state
    i = 0
    while True:
        i += 1
        print('In', i, 'th iteration the initial state is: ', state)
        # employ an epsilon-greedy strategy for exploration vs exploitation
        best_actions = []
        if random.random() < epsilon:

            # choose a random action
            best_actions = actions
        else:

            # find the best actions (the ones with largest q-value)
            maxQ = -float("inf")
            for action in actions:
                Q = Qvalues[(state, action)]
                if Q > maxQ:
                    best_actions = [action]
                    maxQ = Q
                elif Q == maxQ:
                    best_actions.append(action)

        print('the best actions are', best_actions)
        # randomly select a tie-breaking, valid action
        while True:
            best_action = random.choice(best_actions)
            print('The action randomly chosen is', best_action)
            temp_robot = copy.deepcopy(robot)
            temp_robot.move(best_action[0], best_action[1], 1)
            if (temp_robot.theta, temp_robot.a1, temp_robot.a2) in states:
                break
        robot = temp_robot
        print('In', i, 'th iteration the best action is: ', best_action)

        # transition to new state
        new_state = robot.state
        print('In', i, 'th iteration the new state is: ', new_state)

        # find the maximum Q value for new state
        Q = -float("inf")
        for action in actions:
            Q = max(Q, Qvalues[(new_state, action)])

        # find the reward of this transition
        reward = 0
        a1, a2, R, v, a1dot, a2dot = robot.a1, robot.a2, robot.R, robot.body_v[0], robot.a1dot, robot.a2dot
        if a1 == a2:
            reward += -10*R
        else:
            print('In ', i, 'th iteration the penalty for joint angle proximity is: ', log(a1-a2), 'for joint angles: ', a1, a2)
            reward += log(a1-a2)
        print('In ', i, 'th iteration the reward for x- velocity is: ', v/(a1dot**2 + a2dot**2), 'for velocity, a1dot, a2dot: ', v, a1dot, a2dot)
        reward += v/(a1dot**2 + a2dot**2)

        # TD update
        sample = gamma * Q
        old_Q = Qvalues[(state, best_action)]
        print('In ', i, 'th iteration the Q value before update is: ', old_Q)
        new_Q = (1 - alpha) * old_Q + alpha * (reward + sample)
        Qvalues[(state, best_action)] = new_Q
        print('In ', i, 'th iteration the Q value after update is: ', new_Q)
        state = new_state

        # check for convergence
        if old_Q == 0:
            pass
        elif abs((new_Q-old_Q)/old_Q) <= 0.05:
            print('algorithm converged')
            break
        print('\n')

    # extract policy
    policy = extract_policy(Qvalues, states, actions)

    return policy


def get_action_space(lower_limit, upper_limit, interval):
    """
    auxiliary function used by Qlearner() to get action space
    :return: a list of action space values in tuple format (a1dot, a2dot)
    """
    upper_limit += (interval/10)  # to ensure the range covers the rightmost value in the loop
    r = np.arange(lower_limit, upper_limit, interval)
    space = [(rnd(i), rnd(j)) for i in r for j in r]

    # remove a1dot = 0, a2dot = 0 from action space
    space.remove((0,0))

    return space


def get_state_space(theta_lower, theta_upper, theta_interval,
                    a1_lower, a1_upper, a1_interval, a2_lower, a2_upper, a2_interval):
    """
    auxiliary function used by Qlearner() to get action space
    :return: a list of state space values in tuple format (a1, a2, theta)
    """
    # to ensure the range covers the rightmost value in the loop
    theta_upper += (theta_interval/10)
    a1_upper += (a1_interval/10)
    a2_upper += (a2_interval/10)

    theta_range = np.arange(theta_lower, theta_upper, theta_interval)
    a1_range = np.arange(a1_lower, a1_upper, a1_interval)
    a2_range = np.arange(a2_lower, a2_upper, a2_interval)
    space = [(rnd(theta), rnd(a1), rnd(a2)) for theta in theta_range for a1 in a1_range for a2 in a2_range]

    return space


def extract_policy(Qvalues, states, actions):
    policy = {}
    for state in states:
        maxQ = -float("inf")
        best_action = None
        for action in actions:
            Q = Qvalues[(state, action)]
            maxQ = max(Q, maxQ)
            if Q == maxQ:
                best_action = action
        policy[state] = best_action
    return policy


def test_policy(robot, policy, timestep=20):
    x_displacement = 0
    for i in range(timestep):
        initial_state = robot.state
        print('In', i, 'th iteration the initial state is: ', initial_state)
        old_x = robot.x
        action = policy[initial_state]
        robot.move(action[0], action[1], 1)
        new_x = robot.x
        print('In ', i, 'th iteration, the robot moved ', new_x - old_x, ' in x direction')
        x_displacement += (new_x-old_x)
    return x_displacement


def rnd(number):
    return round(number, 8)


if __name__ == "__main__":
    robot = ThreeLinkRobot(x=0, 
                           y=0, 
                           theta=0, 
                           a1=pi/16,
                           a2=-pi/16,
                           link_length=2, 
                           t_interval=0.01, 
                           a_interval=pi/32)
    policy = Qlearner(robot=robot,
                      alpha=0.5,
                      gamma=0.9,
                      epsilon=0.5,
                      theta_lower=-pi,
                      theta_upper=pi,
                      theta_interval=pi/32,
                      a1_lower=pi/32,
                      a1_upper=pi/8,
                      a1_interval=pi/32,
                      a2_lower=-pi/8,
                      a2_upper=-pi/32,
                      a2_interval=pi/32,
                      a_lower=-pi/8,
                      a_upper=pi/8,
                      a_interval=pi/32)
    robot2 = ThreeLinkRobot(x=0,
                           y=0,
                           theta=0,
                           a1=pi/16,
                           a2=-pi/16,
                           link_length=2,
                           t_interval=0.01,
                           a_interval=pi/32)
    x_displacement = test_policy(robot=robot2, policy=policy)
