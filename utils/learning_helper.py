from math import pi


def reward_fucntion(old_x, old_a1, old_a2,
                    new_x, new_a1, new_a2, theta,
                    c_x=50, c_joint=0, c_zero_x=100, c_theta=5, reward_theta=True):

    x_displacement_reward = new_x - old_x
    old_as = [old_a1, old_a2]
    new_as = [new_a1, new_a2]

    # incur joint limit penalty
    joint_penalty = 0
    for i in range(len(old_as)):
        if abs(old_as[i] - pi / 2) <= 0.00001 or abs(old_as[i] + pi / 2) <= 0.00001:
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
        if -pi / 4 <= theta <= pi / 4:
            theta_reward = 1  # constant when theta is in desired range
        else:
            theta_reward = pi / 4 - abs(theta)  # linearly decreasing as theta increases

    reward = c_x * x_displacement_reward + c_joint * joint_penalty + \
             c_zero_x * zero_x_penalty + c_theta * theta_reward

    return reward
