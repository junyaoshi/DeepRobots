import sys

# Edit the system path as needed
sys.path.append('/home/pi/DeepRobots')

from DQN.DQN_agent import DQN_Agent
from Robots.PhysicalRobot import PhysicalRobot
from math import pi
from utils.learning_helper import physical_forward_reward_function

# ------------------------------------------- env ------------------------------------------- #
ROBOT_TYPE = "physical"             # robot type: ["swimming", "wheeled"]
DELAY = 0.025 #0.015                       # the number of seconds the servo sleeps between each differential actions
A_LOWER = -60                       # lower bound of joint angles
A_UPPER = 60                        # upper bound of joint angles
A_INTERVAL = 60                     # interval used to discretize joint angle action space

# ---------------------------------------- file-saving --------------------------------------- #
TRIAL_NUM = 0                       # the trial number
TRIAL_NOTE = "test"                 # comment for this trial

# ----------------------------------------- step num ----------------------------------------- #
EPISODES = 1 #6                       # number of total episodes per trial
ITERATIONS = 10 #500                    # number of total iterations per episode

# ------------------------------------------- DQN -------------------------------------------- #
REWARD_FUNC = "forward"             # reward function: ["forward"]
NETWORK_UPDATE_FREQ = 50            # frequency of updating the original network with copy network
BATCH_SIZE = 8                       # the size of minibatch sampled from replay buffer for SGD update
EPSILON_MIN = 0.1                   # minimum value of epsilon in epsilon-greedy exploration
LEARNING_RATE = 2e-4                # learning rate of neural network
MODEL_ARCHITECTURE = "100_20"       # number of neurons in each layer, separated by underscore


def main():
    robot_type = args.robot_type
    if robot_type == "physical":
        robot = PhysicalRobot(delay=args.delay)
        check_singularity = False
    else:
        raise ValueError("Unknown robot type: {}".format(robot_type))

    episodes = args.episodes
    iterations = args.iterations
    total_iterations = episodes * iterations
    if args.reward_func == "forward":
        reward_function = physical_forward_reward_function
    else:
        raise ValueError("Unknown reward function: {}".format(args.reward_func))

    a_lower = args.a_lower
    a_upper = args.a_upper
    a_interval = args.a_interval
    action_params = (a_lower, a_upper, a_interval)
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
        "delay": args.delay,
        "a_lower": args.a_lower,
        "a_upper": args.a_upper,
        "a_interval": args.a_interval, 
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
                          input_dim=len(robot.state) + 2,
                          output_dim=1,
                          actions_params=action_params,
                          model_architecture=model_architecture,
                          memory_size=total_iterations//50,
                          memory_buffer_coef=20,
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
    parser.add_argument('--delay', type=float, default=DELAY)
    parser.add_argument('--a_lower', type=int, default=A_LOWER)
    parser.add_argument('--a_upper', type=int, default=A_UPPER)
    parser.add_argument('--a_interval', type=int, default=A_INTERVAL)

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
