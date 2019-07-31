import sys

# Edit the system path as needed
sys.path.append('/home/jackshi/DeepRobots')


from DQN.DQN_agent import DQN_Agent
from Robots.ContinuousSwimmingBot import SwimmingRobot
from math import pi


def reward_function(old_x, old_a1, old_a2,
                    new_x, new_a1, new_a2, theta,
                    c_x=50, c_joint=0, c_zero_x=20, c_theta=2,
                    penalize_joint_limit=False, reward_theta=True):

    x_displacement_reward = new_x - old_x
    old_as = [old_a1, old_a2]
    new_as = [new_a1, new_a2]

    # incur joint limit penalty
    joint_penalty = 0
    if penalize_joint_limit:
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
        if -pi / 4 <= theta <= pi / 4:
            theta_reward = 1  # constant when theta is in desired range
        else:
            theta_reward = pi / 4 - abs(theta)  # linearly decreasing as theta increases

    reward = c_x * x_displacement_reward + c_joint * joint_penalty + \
             c_zero_x * zero_x_penalty + c_theta * theta_reward

    return reward


def main():

    # 0.99996 for 30000 iterations
    # 0.999 for 1000 iterations
    # 0.9998 for 10000 iterations
    # 0.99995 for 20000
    # 0.999965 for 40000
    # 0.99997 for 50000
    # 0.999975 for 60000
    # 0.999985 for 100000
    # 0.999993 for 200000
    # 0.999997 for 500000
    # 0.9999987 for 1000000
    # 0.999999 for 2000000
    # 0.9999994 for 3000000
    # 0.9999997 for 6000000

    robot = SwimmingRobot(t_interval=8)
    trial_name = 'DQN_swimming_w_theta_largest_action_10000_iters'
    trial_num = 31
    episodes = 20
    iterations = 500
    total_iterations = episodes * iterations
    network_update_freq = 30
    batch_size = 8
    epsilon_decay = 0.9998
    learning_rate = 2e-4

    dqn_agent = DQN_Agent(robot=robot,
                          reward_function=reward_function,
                          trial_name=trial_name,
                          trial_num=trial_num,
                          episodes=episodes,
                          iterations=iterations,
                          network_update_freq=network_update_freq,
                          check_singularity=False,
                          input_dim=5,
                          output_dim=1,
                          actions_params=(-pi/8, pi/8, pi/8),
                          model_architecture=(50, 10),
                          memory_size=total_iterations//50,
                          memory_buffer_coef=20,
                          randomize_theta=False,
                          batch_size=batch_size,
                          gamma=0.99,
                          epsilon=1.0,
                          epsilon_min=0.1,
                          epsilon_decay=epsilon_decay,
                          learning_rate=learning_rate)

    dqn_agent.run()


if __name__ == '__main__':
    main()
