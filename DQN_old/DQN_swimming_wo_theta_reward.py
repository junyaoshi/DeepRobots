# change current working directory to parent directory
import os, sys

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# os.chdir(parentdir)

# Edit the system path as needed
sys.path.append('/home/jackshi/DeepRobots')

# import libraries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Robots.ContinuousSwimmingBot import SwimmingRobot
import datetime
import random
import numpy as np
from collections import deque
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.optimizers import RMSprop
from utils.csv_generator import generate_csv
from math import pi
import csv

# Define DQNAgent Class
class DQNAgent:

    INPUT_DIM = 5
    OUTPUT_DIM = 1

    def __init__(self, actions_params=(-pi/16, pi/16, pi/128), memory_size=500, gamma=0.98, epsilon=1.0,
                 epsilon_min=0.1, epsilon_decay=0.9995, learning_rate=0.001):
        self.memory = deque(maxlen=memory_size)
        self.memory_size = memory_size
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.actions = self._get_actions(actions_params)
        self.model = self._build_model()
        self.model_clone = self._build_model()
        self.model_clone.set_weights(self.model.get_weights())

    def _get_actions(self, actions_params):
        """
        :return: a list of action space values in tuple format (a1dot, a2dot)
        """
        lower_limit, upper_limit, interval = actions_params
        upper_limit += (interval/10)  # to ensure the range covers the rightmost value in the loop
        r = np.arange(lower_limit, upper_limit, interval)
        # r = np.delete(r, len(r) // 2) # delete joint actions with 0 joint movement in either joint
        actions = [(i, j) for i in r for j in r]

        # remove a1dot = 0, a2dot = 0 from action space
        actions.remove((0.0,0.0))
        print(actions)
        return actions

    def _build_model(self):

        # Neural Net for Deep-Q learning Model
        model = Sequential()

        # input layer
        model.add(Dense(100, input_dim=self.INPUT_DIM, activation='relu'))

        # hidden layers
        model.add(Dense(20, activation='relu'))
        model.add(Dense(10, activation='relu'))

        # output layer
        model.add(Dense(self.OUTPUT_DIM, activation = 'linear'))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def choose_action(self, state, epsilon_greedy=False):
        """
        epsilon-greedy approach for choosing an action and transition into next state
        returns the next state, reward resulting from the chosen action
        """
        chosen_action = None
        if epsilon_greedy:
            if np.random.rand() <= self.epsilon:
                print('random actions')

                # choose random action
                chosen_action = random.choice(self.actions)

            else:
                print('argmax')

                # find the action with greatest Q value
                maxQ = -float("inf")
                for action in self.actions:
                    input_data = np.asarray(state + action).reshape(self.OUTPUT_DIM, self.INPUT_DIM)
                    Q = self.model.predict(input_data)
                    if Q > maxQ:
                        maxQ = Q
                        chosen_action = action

        else:

            # policy rollout
            maxQ = -float("inf")
            for action in self.actions:
                input_data = np.asarray(state + action).reshape(self.OUTPUT_DIM, self.INPUT_DIM)
                Q = self.model.predict(input_data)
                if Q > maxQ:
                    maxQ = Q
                    chosen_action = action

        return chosen_action

    def act(self, robot, action, c_x=100, c_joint=50, c_zero_x=50, c_theta=5, reward_theta=True):

        # transition into next state
        # print('act state: {s}'.format(s=robot.state))
        # print('act action: {s}'.format(s=action))
        old_x, old_a1, old_a2 = robot.x, robot.a1, robot.a2
        next_state = robot.move(action=action)
        # print('act state after: {s}'.format(s=next_state))

        # calculate reward
        # a1, a2, a1dot, a2dot = robot.a1, robot.a2, robot.a1dot, robot.a2dot
        new_x, new_a1, new_a2, theta = robot.x, robot.a1, robot.a2, robot.theta
        x_displacement_reward = new_x-old_x
        old_as = [old_a1, old_a2]
        new_as = [new_a1, new_a2]

        # incur joint limit penalty
        joint_penalty = 0
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
        # if reward_theta:
        #     if -pi / 4 <= theta <= pi / 4:
        #         theta_reward = 1  # constant when theta is in desired range
        #     else:
        #         theta_reward = pi / 4 - abs(theta)  # linearly decreasing as theta increases

        reward = c_x * x_displacement_reward + c_joint * joint_penalty + \
                 c_zero_x * zero_x_penalty + c_theta * theta_reward

        return robot, reward, next_state

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        losses = []
        Q_targets = []
        for state, action, reward, next_state in minibatch:

            # find max Q for next state
            Q_prime = float('-inf')
            for next_action in self.actions:
                next_input = np.asarray(next_state + next_action).reshape(self.OUTPUT_DIM, self.INPUT_DIM)
                # print('reward: ', reward, 'prediction: ', self.model.predict(input_data))
                current_Q = self.model_clone.predict(next_input)
                # print('Qprime: {x}, current_Q: {y}'.format(x=Q_prime, y=current_Q))
                Q_prime = max(Q_prime, current_Q)
                # print('afterwards, Qprime: {x}'.format(x=Q_prime))

            # Q_prime = Q_prime[0, 0]
            # print('Q prime: ', Q_prime)
            # calculate network update target
            # print('Qprime: {}, gamma: {}, reward: {}'.format(Q_prime, self.gamma, reward))
            Q_target = reward + self.gamma * Q_prime
            # print('Qtarget: {}'.format(Q_target))

            # print('Qtarget: {x}'.format(x=Q_target[0, 0]))

            # perform a gradient descent step
            input_data = np.asarray(state + action).reshape(self.OUTPUT_DIM, self.INPUT_DIM)
            loss = self.model.train_on_batch(input_data, Q_target)
            # print('loss: {x}'.format(x=loss))
            # print('loss: ', loss, 'input: ', input_data, 'Q_target: ', Q_target)
            losses.append(loss)
            Q_targets.append(Q_target[0, 0])
            # self.model.fit(state, target_f, epochs=1, verbose=0)

        # update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # return the average loss of this experience replay
        return sum(losses)/len(losses), sum(Q_targets)/len(Q_targets)

    def load_model(self, json_name, h5_name):

        # load json and create model
        json_file = open(json_name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)

        # load weights into new model
        self.model.load_weights(h5_name)

        # compile model
        self.model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate))
        print("Loaded model from disk")

    def save_model(self, path, e):

        # serialize model to JSON
        model_json = self.model.to_json()
        with open(path + "/" + str(e) + "th_episode_model.json", "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save_weights(path + "/" + str(e) + "th_episode_weights.h5")
        print("Saved model to disk")

    def update_model(self):
        self.model_clone.set_weights(self.model.get_weights())


# Policy Rollout Function
def policy_rollout(agent, path, timesteps=200):
    for j in range(1):
        robot = SwimmingRobot(a1=0, a2=0, t_interval=1)
        xs = [robot.x]
        ys = [robot.y]
        thetas = [robot.theta]
        a1s = [robot.a1]
        a2s = [robot.a2]
        steps = [0]
        # robot.randomize_state(enforce_opposite_angle_signs=True)
        robot_params = []
        robot_param = [robot.x, robot.y, robot.theta, float(robot.a1), float(robot.a2), robot.a1dot, robot.a2dot]
        robot_params.append(robot_param)
        print('Beginning', j + 1, 'th Policy Rollout')
        try:
            for i in range(timesteps):
                # rollout
                state = robot.state
                print('In', i + 1, 'th iteration the initial state is: ', state)
                old_x = robot.x
                action = agent.choose_action(state)
                print('In', i + 1, 'th iteration the chosen action is: ', action)
                robot.move(action=action)
                new_x = robot.x
                print('In', i + 1, 'th iteration, the robot moved ', new_x - old_x, ' in x direction')

                # add values to lists
                xs.append(robot.x)
                ys.append(robot.y)
                thetas.append(robot.theta)
                a1s.append(robot.a1)
                a2s.append(robot.a2)
                steps.append(i + 1)
                robot_param = [robot.x, robot.y, robot.theta, float(robot.a1), float(robot.a2), robot.a1dot,
                               robot.a2dot]
                robot_params.append(robot_param)

        except ZeroDivisionError as e:
            print(str(e), 'occured at ', j + 1, 'th policy rollout')

        # plotting
        make_rollout_graphs(xs, ys, thetas, a1s, a2s, steps, path=path)
        generate_csv(robot_params, path + "/policy_rollout.csv")


# Graphing Functions
def make_rollout_graphs(xs, ys, thetas, a1s, a2s, steps, path):

    # plotting
    fig1 = plt.figure(1)
    fig1.suptitle('Policy Rollout X, Y, Theta vs Time')
    ax1 = fig1.add_subplot(311)
    ax2 = fig1.add_subplot(312)
    ax3 = fig1.add_subplot(313)

    fig2 = plt.figure(2)
    fig2.suptitle('Policy Rollout a1 vs a2')
    ax4 = fig2.add_subplot(111)

    fig3 = plt.figure(3)
    fig3.suptitle('Policy Rollout a1 and a2 vs Time')
    ax5 = fig3.add_subplot(211)
    ax6 = fig3.add_subplot(212)

    fig4 = plt.figure(4)
    fig4.suptitle('Policy Rollout X vs Y')
    ax7 = fig4.add_subplot(111)

    ax1.plot(steps, xs, '.-')
    ax1.set_ylabel('x')
    ax1.set_xlabel('steps')
    ax2.plot(steps, ys, '.-')
    ax2.set_ylabel('y')
    ax2.set_xlabel('steps')
    ax3.plot(steps, thetas, '.-')
    ax3.set_ylabel('theta')
    ax3.set_xlabel('steps')

    ax4.plot(a1s,a2s,'.-')
    ax4.set_xlabel('a1')
    ax4.set_ylabel('a2')

    ax5.plot(steps, a1s, '.-')
    ax5.set_xlabel('a1')
    ax5.set_ylabel('steps')
    ax6.plot(steps, a2s, '.-')
    ax6.set_xlabel('a2')
    ax6.set_ylabel('steps')

    ax7.plot(xs, ys,'.-')
    ax7.set_xlabel('x')
    ax7.set_ylabel('y')

    ax1.grid(True, which='both', alpha=.2)
    ax2.grid(True, which='both', alpha=.2)
    ax3.grid(True, which='both', alpha=.2)
    ax4.grid(True, which='both', alpha=.2)
    ax5.grid(True, which='both', alpha=.2)
    ax6.grid(True, which='both', alpha=.2)
    ax7.grid(True, which='both', alpha=.2)

    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig1.savefig(path + '/Policy_Rollout_X_Y_Theta_vs_Time.png')
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2.savefig(path + '/Policy_Rollout_a1_vs_a2.png')
    fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig3.savefig(path + '/Policy_Rollout_a1_and_a2_vs_Time.png')
    fig4.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig4.savefig(path + '/Policy_Rollout_X_and_Y.png')

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)


def make_loss_plot(num_episodes, avg_losses, std_losses, path):
    avg_losses = np.array(avg_losses)
    std_losses = np.array(std_losses)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Average Loss vs Number of Iterations')
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Average Loss')
    ax.grid(True, which='both', alpha=.2)
    ax.plot(num_episodes, avg_losses)
    ax.fill_between(num_episodes, avg_losses-std_losses, avg_losses+std_losses, alpha=.2)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(path + '/Average_Loss_vs_Number_of_Iterations.png')
    plt.close()


def make_learning_plot(num_episodes, avg_rewards, std_rewards, path):
    avg_rewards = np.array(avg_rewards)
    std_rewards = np.array(std_rewards)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Learning Curve Plot')
    ax.set_xlabel('Number of Episodes')
    ax.set_ylabel('Average Reward')
    ax.grid(True, which='both', alpha=.2)
    ax.plot(num_episodes, avg_rewards)
    ax.fill_between(num_episodes, avg_rewards-std_rewards, avg_rewards+std_rewards, alpha=.2)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(path + '/Learning_Curve_Plot.png')
    plt.close()


def make_Q_plot(num_episodes, avg_Qs, std_Qs, path):
    avg_Qs = np.array(avg_Qs)
    std_Qs = np.array(std_Qs)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Q Plot')
    ax.set_xlabel('Number of Episodes')
    ax.set_ylabel('Q')
    ax.grid(True, which='both', alpha=.2)
    ax.plot(num_episodes, avg_Qs)
    ax.fill_between(num_episodes, avg_Qs-std_Qs, avg_Qs+std_Qs, alpha=.2)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(path + '/Q_Plot.png')
    plt.close()


# Perform DQN
def get_random_edge_states():
    num = np.random.rand()
    if num < 0.2:
        print('Normal robot!')
        robot = SwimmingRobot(t_interval=1)
    elif num < 0.4:
        print('edge case 1!')
        robot = SwimmingRobot(a1=-pi / 2, a2=pi / 2, t_interval=0.5)
    elif num < 0.6:
        print('edge case 2!')
        robot = SwimmingRobot(a1=-pi / 2, a2=-pi / 2, t_interval=0.5)
    elif num < 0.8:
        print('edge case 3!')
        robot = SwimmingRobot(a1=pi / 2, a2=-pi / 2, t_interval=0.5)
    else:
        print('edge case 4')
        robot = SwimmingRobot(a1=pi / 2, a2=pi / 2, t_interval=0.5)

    return robot


def save_learning_data(path, num_episodes, avg_rewards, std_rewards, avg_losses, std_losses, avg_Qs, std_Qs):
    """
    saving learning results to csv
    """
    rows = zip(num_episodes, avg_rewards, std_rewards, avg_losses, std_losses, avg_Qs, std_Qs)
    with open(path + '/learning_data.csv', 'w') as f:
        w = csv.writer(f)
        w.writerows(rows)


def perform_DQN(agent, episodes, iterations, path, batch_size=4, C=30, randomize_theta=False):
    """
    :param agent: the RL agent
    :param batch_size: size of minibatch sampled from replay buffer
    :param C: network update frequency
    :return: agent, and other information about DQN
    """
    avg_losses = []
    std_losses = []
    avg_rewards = []
    std_rewards = []
    avg_Qs = []
    std_Qs = []
    # gd_iterations = [] # gradient descent iterations
    # gd_iteration = 0
    num_episodes = []

    try:
        # loop through each episodes
        for e in range(1, episodes + 1):

            # save model
            if e % (episodes / 10) == 0:
                agent.save_model(path, e)

            theta = random.uniform(-pi / 4, pi / 4) if randomize_theta else 0
            robot = SwimmingRobot(a1=0, a2=0, theta=theta, t_interval=1)
            # state = robot.randomize_state()
            state = robot.state
            rewards = []
            losses = []
            Qs = []

            # loop through each iteration
            for i in range(1, iterations + 1):
                # print('In ', e, ' th epsiode, ', i, ' th iteration, the initial state is: ', state)
                action = agent.choose_action(state, epsilon_greedy=True)
                print('In {}th epsiode {}th iteration, the chosen action is: {}'.format(e, i, action))
                robot_after_transition, reward, next_state = agent.act(robot=robot, action=action,
                                                                       c_x=50, c_joint=50, c_zero_x=50, c_theta=0)
                print('The reward is: {}'.format(reward))
                rewards.append(reward)
                # print('In ', e, ' th epsiode, ', i, ' th iteration, the state after transition is: ', next_state)
                agent.remember(state, action, reward, next_state)
                state = next_state
                robot = robot_after_transition
                if len(agent.memory) > agent.memory_size / 20:
                    loss, Q = agent.replay(batch_size)
                    # gd_iteration += 1
                    losses.append(loss)
                    Qs.append(Q)
                    # gd_iterations.append(gd_iteration)
                    print('The average loss is: {}'.format(loss))
                    print('The average Q is: {}'.format(Q))

                if i % C == 0:
                    agent.update_model()

            num_episodes.append(e)
            avg_rewards.append(np.mean(rewards))
            std_rewards.append(np.std(rewards))
            avg_losses.append(np.mean(losses))
            std_losses.append(np.std(losses))
            avg_Qs.append(np.mean(Qs))
            std_Qs.append(np.std(Qs))

    except TypeError as e:
        print(e)

    finally:

        # save learning data
        save_learning_data(path, num_episodes, avg_rewards, std_rewards, avg_losses, std_losses, avg_Qs, std_Qs)
        return agent, num_episodes, avg_rewards, std_rewards, avg_losses, std_losses, avg_Qs, std_Qs


if __name__ == '__main__':

    # specify program information
    TIMESTAMP = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')[:-7]
    TRIAL_NAME = 'DQN_Swimming_Reproduce_14'
    TRIAL_NUM = 23
    PATH = 'Trials/' + TRIAL_NAME + '_Trial_' + str(TRIAL_NUM) + "_" + TIMESTAMP

    # create directory
    os.mkdir(PATH)
    os.chmod(PATH, 0o0777)

    # 0.99996 for 30000 iterations
    # 0.999 for 1000 iterations
    # 0.,99987 for 10000 iterations
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
    agent = DQNAgent(gamma=0.98, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.9999987,
                     memory_size=20000, actions_params=(-pi/8, pi/8, pi/8), learning_rate=0.0001)

    # Perform DQN
    learning_results = perform_DQN(agent=agent, path=PATH, episodes=1000, iterations=1000, batch_size=8, C=200)
    agent, num_episodes, avg_rewards, std_rewards, avg_losses, std_losses, avg_Qs, std_Qs = learning_results

    # Loss Plot
    make_loss_plot(num_episodes, avg_losses, std_losses, path=PATH)

    # Learning Curve Plot
    make_learning_plot(num_episodes, avg_rewards, std_rewards, path=PATH)

    # Make Q Plot
    make_Q_plot(num_episodes, avg_Qs, std_Qs, path=PATH)

    # Policy Rollout
    policy_rollout(agent=agent, path=PATH)

