from math import pi
import numpy as np
import csv


def save_learning_data(path, num_episodes, avg_rewards, std_rewards, avg_losses, std_losses, avg_Qs, std_Qs):
    """
    saving learning results to csv
    """
    rows = zip(num_episodes, avg_rewards, std_rewards, avg_losses, std_losses, avg_Qs, std_Qs)
    with open(path + '/learning_data.csv', 'w') as f:
        w = csv.writer(f)
        w.writerows(rows)

# Perform DQN
def get_random_edge_states(robot):
    num = np.random.rand()
    theta = robot.theta
    if num < 0.2:
        pass
    elif num < 0.4:
        print('edge case 1!')
        robot.set_state(theta=theta, a1=-pi/2, a2=pi/2)
    elif num < 0.6:
        print('edge case 2!')
        robot.set_state(theta=theta, a1=pi/2, a2=pi/2)
    elif num < 0.8:
        print('edge case 3!')
        robot.set_state(theta=theta, a1=pi/2, a2=-pi/2)
    else:
        print('edge case 4')
        robot.set_state(theta=theta, a1=-pi/2, a2=-pi/2)
    return robot


def forward_reward_function(robot, action,
                            c_x=50, c_joint=0, c_zero_x=50, c_theta=5,
                            penalize_joint_limit=False, reward_theta=True):
    old_x, old_y, old_theta, old_a1, old_a2 = robot.x, \
                                              robot.y, \
                                              robot.theta, \
                                              robot.a1, \
                                              robot.a2
    # print("robot state before action: {}".format(robot.state))
    # print("chosen action: {}".format(action))
    robot.move(action=action)
    # print("robot state after action: {}".format(robot.state))
    # print('act state after: {s}'.format(s=next_state))

    # calculate reward
    new_x, new_y, new_theta, new_a1, new_a2 = robot.x, \
                                              robot.y, \
                                              robot.theta, \
                                              robot.a1, \
                                              robot.a2

    x_displacement_reward = new_x - old_x
    old_as = [old_a1, old_a2]
    new_as = [new_a1, new_a2]

    # incur joint limit penalty
    joint_penalty = 0
    if penalize_joint_limit and c_joint != 0:
        for i in range(len(old_as)):
            if abs(old_as[i] - pi/2) <= 0.00001 or abs(old_as[i] + pi/2) <= 0.00001:
                if old_as[i] == new_as[i]:
                    joint_penalty = -1
                    print('incur joint limit penalty')

    # 0 x-displacement penalty
    zero_x_penalty = 0
    if x_displacement_reward == 0:
        print('incur 0 x displacement penalty')
        zero_x_penalty = -1

    # theta displacement penalty/reward
    theta_reward = 0
    if reward_theta:
        if -pi / 4 <= new_theta <= pi / 4:
            theta_reward = 1  # constant when theta is in desired range
        else:
            theta_reward = pi / 4 - abs(new_theta)  # linearly decreasing as theta increases

    reward = c_x * x_displacement_reward + c_joint * joint_penalty + \
             c_zero_x * zero_x_penalty + c_theta * theta_reward

    return reward, robot


def backward_reward_function(robot, action,
                             c_x=50, c_joint=0, c_zero_x=20, c_theta=2,
                             penalize_joint_limit=False, reward_theta=True):

    old_x, old_y, old_theta, old_a1, old_a2 = robot.x, \
                                              robot.y, \
                                              robot.theta, \
                                              robot.a1, \
                                              robot.a2
    robot.move(action=action)
    # print('act state after: {s}'.format(s=next_state))

    # calculate reward
    new_x, new_y, new_theta, new_a1, new_a2 = robot.x, \
                                              robot.y, \
                                              robot.theta, \
                                              robot.a1, \
                                              robot.a2

    x_displacement_reward = old_x - new_x
    old_as = [old_a1, old_a2]
    new_as = [new_a1, new_a2]

    # incur joint limit penalty
    joint_penalty = 0
    if penalize_joint_limit and c_joint != 0:
        for i in range(len(old_as)):
            if abs(old_as[i] - pi/2) <= 0.00001 or abs(old_as[i] + pi/2) <= 0.00001:
                if old_as[i] == new_as[i]:
                    joint_penalty = -1
                    print('incur joint limit penalty')

    # 0 x-displacement penalty
    zero_x_penalty = 0
    if x_displacement_reward == 0:
        print('incur 0 x displacement penalty')
        zero_x_penalty = -1

    # theta displacement penalty/reward
    theta_reward = 0
    if reward_theta:
        if -pi/4 <= new_theta <= pi/4:
            theta_reward = 1  # constant when theta is in desired range
        else:
            theta_reward = pi/4 - abs(new_theta)  # linearly decreasing as theta increases

    reward = c_x * x_displacement_reward + c_joint * joint_penalty + \
             c_zero_x * zero_x_penalty + c_theta * theta_reward

    return reward, robot


def upward_reward_function(robot, action,
                           c_y=50, c_joint=0, c_zero_y=20, c_theta=2,
                           penalize_joint_limit=False, reward_theta=True):
    old_x, old_y, old_theta, old_a1, old_a2 = robot.x, \
                                              robot.y, \
                                              robot.theta, \
                                              robot.a1, \
                                              robot.a2
    robot.move(action=action)
    # print('act state after: {s}'.format(s=next_state))

    # calculate reward
    new_x, new_y, new_theta, new_a1, new_a2 = robot.x, \
                                              robot.y, \
                                              robot.theta, \
                                              robot.a1, \
                                              robot.a2
    y_displacement_reward = new_y - old_y
    old_as = [old_a1, old_a2]
    new_as = [new_a1, new_a2]

    # incur joint limit penalty
    joint_penalty = 0
    if penalize_joint_limit and c_joint != 0:
        for i in range(len(old_as)):
            if abs(old_as[i] - pi/2) <= 0.00001 or abs(old_as[i] + pi/2) <= 0.00001:
                if old_as[i] == new_as[i]:
                    joint_penalty = -1
                    print('incur joint limit penalty')

    # 0 x-displacement penalty
    zero_y_penalty = 0
    if y_displacement_reward == 0:
        print('incur 0 y displacement penalty')
        zero_y_penalty = -1

    # theta displacement penalty/reward
    theta_reward = 0
    if reward_theta:
        if pi/4 <= new_theta <= 3*pi/4:
            theta_reward = 1  # constant when theta is in desired range
        else:
            if new_theta > 3*pi/4:
                theta_reward = 3*pi/4 - new_theta
            else:
                theta_reward = new_theta - pi/4

            # linearly decreasing as theta increases

    reward = c_y * y_displacement_reward + c_joint * joint_penalty + \
             c_zero_y * zero_y_penalty + c_theta * theta_reward

    return reward, robot


def left_reward_function(robot, action,
                         c_t=50, c_joint=0, c_zero_t=20,
                         penalize_joint_limit=False):
    old_a1, old_a2 = robot.a1, robot.a2
    robot.move(action=action)
    # print('act state after: {s}'.format(s=next_state))

    # calculate reward
    new_a1, new_a2 = robot.a1, robot.a2
    t_displacement_reward = robot.theta_displacement
    old_as = [old_a1, old_a2]
    new_as = [new_a1, new_a2]

    # incur joint limit penalty
    joint_penalty = 0
    if penalize_joint_limit and c_joint != 0:
        for i in range(len(old_as)):
            if abs(old_as[i] - pi/2) <= 0.00001 or abs(old_as[i] + pi/2) <= 0.00001:
                if old_as[i] == new_as[i]:
                    joint_penalty = -1
                    print('incur joint limit penalty')

    # 0 x-displacement penalty
    zero_t_penalty = 0
    if t_displacement_reward == 0:
        print('incur 0 theta displacement penalty')
        zero_t_penalty = -1

    # linearly decreasing as theta increases

    reward = c_t * t_displacement_reward + c_joint * joint_penalty + \
             c_zero_t * zero_t_penalty

    return reward, robot


def physical_forward_reward_function(robot, action,
                                     c_x=1, c_zero_x=20):
    robot.move(action=action)
    # print('act state after: {s}'.format(s=next_state))

    x_displacement_reward = robot.encoder_displacement

    # 0 x-displacement penalty
    zero_x_penalty = 0
    if x_displacement_reward == 0:
        print('incur 0 x displacement penalty')
        zero_x_penalty = -1

    reward = c_x * x_displacement_reward + c_zero_x * zero_x_penalty

    return reward, robot
