import sys

# Edit the system path as needed
sys.path.append('/home/jackshi/DeepRobots')


from DQN.DQN_agent import DQN_Agent
from Robots.ContinuousSwimmingBot import SwimmingRobot
from math import pi
from utils.learning_helper import forward_reward_function, backward_reward_function, upward_reward_function


def main():

    # 0.99996 for 30000 iterations
    # 0.999 for 1000 iterations
    # 0.9998 for 10000 iterations
    # 0.99995 for 20000
    # 0.999965 for 40000
    # 0.999955 for 50000
    # 0.999975 for 60000
    # 0.999977 for 100000
    # 0.999993 for 200000
    # 0.999997 for 500000
    # 0.999997for 1000000
    # 0.999999 for 2000000
    # 0.9999994 for 3000000
    # 0.9999997 for 6000000

    robot = SwimmingRobot(t_interval=8)
    trial_name = 'DQN_swimming_w_theta_forward_20000_iters'
    trial_num = 0
    reward_function = forward_reward_function
    episodes = 20
    iterations = 1000
    total_iterations = episodes * iterations
    network_update_freq = 20
    batch_size = 8
    epsilon_decay = 0.99995
    learning_rate = 2e-4
    model_architecture = (50, 10)

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
                          model_architecture=model_architecture,
                          memory_size=total_iterations//50,
                          memory_buffer_coef=20,
                          randomize_theta=False,
                          batch_size=batch_size,
                          gamma=0.99,
                          epsilon=1.0,
                          epsilon_min=0.1,
                          epsilon_decay=epsilon_decay,
                          learning_rate=learning_rate,
                          params=None)

    dqn_agent.run()


if __name__ == '__main__':
    main()
