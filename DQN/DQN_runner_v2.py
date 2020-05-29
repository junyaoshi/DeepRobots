import sys

# Edit the system path as needed
sys.path.append('/home/jackshi/DeepRobots')

from DQN.DQN_agent_v2 import DQN_Agent
from Robots.ContinuousSwimmingBot import SwimmingRobot
from Robots.WheeledRobot_v1 import ThreeLinkRobot
from math import pi
from utils.learning_helper import forward_reward_function, left_reward_function

import numpy as np
np.warnings.filterwarnings('ignore', category=FutureWarning)

# ------------------------------------------- env ------------------------------------------- #
ROBOT_TYPE = "swimming"             # robot type: ["swimming", "wheeled"]
T_INTERVAL = 8                      # the number of timesteps used to execute each discrete action
A_UPPER = pi/2                      # upper joint limit
A_LOWER = -pi/2                     # lower joint limit
NO_JOINT_LIMIT = False              # whether or not there are joint limits

# ---------------------------------------- file-saving --------------------------------------- #
TRIAL_NUM = 71                      # the trial number
TRIAL_NOTE = "updated_nn"           # comment for this trial

# ----------------------------------------- step num ----------------------------------------- #
EPISODES = 20                       # number of total episodes per trial
ITERATIONS = 250                    # number of total iterations per episode

# ------------------------------------------- DQN -------------------------------------------- #
REWARD_FUNC = "forward"             # reward function: ["forward", "left"]
NETWORK_UPDATE_FREQ = 20            # frequency of updating the original network with copy network
BATCH_SIZE = 8                      # the size of minibatch sampled from replay buffer for SGD update
EPSILON_MIN = 0.1                   # minimum value of epsilon in epsilon-greedy exploration
LEARNING_RATE = 2e-4                # learning rate of neural network
MODEL_ARCHITECTURE = "30_5"        # number of neurons in each layer, separated by underscore


def main():
    robot_type = args.robot_type
    if robot_type == "swimming":
        robot = SwimmingRobot(t_interval=args.t_interval,
                              a_upper=args.a_upper,
                              a_lower=args.a_lower,
                              no_joint_limit=args.no_joint_limit)
        check_singularity = False
    elif robot_type == "wheeled":
        robot = ThreeLinkRobot(t_interval=args.t_interval)
        check_singularity = True
    else:
        raise ValueError("Unknown robot type: {}".format(robot_type))

    episodes = args.episodes
    iterations = args.iterations
    total_iterations = episodes * iterations
    if args.reward_func == "forward":
        reward_function = forward_reward_function
    elif args.reward_func == "left":
        reward_function = left_reward_function
    else:
        raise ValueError("Unknown reward function: {}".format(args.reward_func))

    network_update_freq = args.network_update_freq
    batch_size = args.batch_size
    epsilon_min = args.epsilon_min
    epsilon_decay = epsilon_min ** (1/total_iterations)
    learning_rate = args.learning_rate
    model_architecture = [int(num) for num in args.model_architecture.split('_')]

    trial_num = args.trial_num
    trial_name = 'DQN_{}_{}_{}_iters'.format(robot_type, args.reward_func, total_iterations)
    if args.trial_note:
        trial_name += "_{}".format(args.trial_note)

    params = {
        "robot_type": args.robot_type,
        "t_interval": args.t_interval,
        "a_upper": args.a_upper,
        "a_lower": args.a_lower,
        "no_joint_limit:": args.no_joint_limit,
        "trial_num": args.trial_num,
        "trial_note": args.trial_note,
        "episodes": args.episodes,
        "iterations": args.iterations,
        "reward_func": args.reward_func,
        "network_update_freq": args.network_update_freq,
        "epsilon_min": args.epsilon_min,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "model_architecture": args.model_architecture,
    }

    dqn_agent = DQN_Agent(robot=robot,
                          reward_function=reward_function,
                          trial_name=trial_name,
                          trial_num=trial_num,
                          episodes=episodes,
                          iterations=iterations,
                          network_update_freq=network_update_freq,
                          check_singularity=check_singularity,
                          actions_params=(-pi/8, pi/8, pi/8),
                          model_architecture=model_architecture,
                          memory_size=total_iterations//50,
                          memory_buffer_coef=5, #5 don't forget to change back to 20!
                          randomize_theta=False,
                          batch_size=batch_size,
                          gamma=0.99,
                          epsilon=1.0,
                          epsilon_min=epsilon_min,
                          epsilon_decay=epsilon_decay,
                          learning_rate=learning_rate,
                          params=params)

    dqn_agent.run()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # env
    parser.add_argument('--robot_type', type=str, choices=["swimming", "wheeled"], default=ROBOT_TYPE)
    parser.add_argument('--t_interval', type=int, default=T_INTERVAL)
    parser.add_argument('--a_upper', type=float, default=A_UPPER)
    parser.add_argument('--a_lower', type=float, default=A_LOWER)
    parser.add_argument('--no_joint_limit', action='store_true', default=NO_JOINT_LIMIT)

    # file-saving
    parser.add_argument('--trial_num', type=int, default=TRIAL_NUM)
    parser.add_argument('--trial_note', type=str, default=TRIAL_NOTE)

    # step num
    parser.add_argument('--episodes', type=int, default=EPISODES)
    parser.add_argument('--iterations', type=int, default=ITERATIONS)

    # DQN
    parser.add_argument('--reward_func', type=str, choices=["forward", "left"], default=REWARD_FUNC)
    parser.add_argument('--network_update_freq', type=int, default=NETWORK_UPDATE_FREQ)
    parser.add_argument('--epsilon_min', type=float, default=EPSILON_MIN)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--model_architecture', type=str, default=MODEL_ARCHITECTURE)

    args = parser.parse_args()

    main()
