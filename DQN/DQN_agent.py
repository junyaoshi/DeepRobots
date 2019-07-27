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
import datetime
import random
import numpy as np
import csv
from pprint import pprint
from collections import deque
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.optimizers import RMSprop
from utils.csv_generator import generate_csv
from math import pi
from Robots.ContinuousSwimmingBot import SwimmingRobot
from utils.graphing_helper import make_learning_plot, make_loss_plot, make_Q_plot, make_rollout_graphs


# Define DQNAgent Class
class DQNAgent:

    def __init__(self,
                 robot,
                 reward_function,
                 input_dim=5,
                 output_dim=1,
                 actions_params=(-pi/8, pi/8, pi/8),
                 model_architecture=(50, 10),
                 memory_size=500,
                 batch_size=8,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_min=0.1,
                 epsilon_decay=0.9995,
                 learning_rate=0.001):

        """
        :param reward_function:
        takes the following parameters as an input:
            old_x, old_a1, old_a2,
            new_x, new_a1, new_a2, theta,
            c_x, c_joint, c_zero_x, c_theta, reward_theta
        it returns the reward for these current moves
        """
        self.robot = robot
        self.reward_function = reward_function
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.actions_params = actions_params
        self.model_architecture = model_architecture
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=self.memory_size)
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.actions = self._get_actions()
        self.model = self._build_model()
        self.model_clone = self._build_model()
        self.model_clone.set_weights(self.model.get_weights())

    def _get_actions(self):
        """
        :return: a list of action space values in tuple format (a1dot, a2dot)
        """
        lower_limit, upper_limit, interval = self.actions_params
        upper_limit += (interval/10)  # to ensure the range covers the rightmost value in the loop
        r = np.arange(lower_limit, upper_limit, interval)
        # r = np.delete(r, len(r) // 2) # delete joint actions with 0 joint movement in either joint
        actions = [(i, j) for i in r for j in r]

        # remove a1dot = 0, a2dot = 0 from action space
        actions.remove((0.0,0.0))
        pprint('The actions initialized are: {}'.format(actions))
        return actions

    def _build_model(self):

        assert len(self.model_architecture) > 0, 'model architecture cannot be an empty list'

        # Neural Net for Deep-Q learning Model
        model = Sequential()

        # layers
        for i in range(len(self.model_architecture)):

            num_neurons = self.model_architecture[i]
            assert num_neurons is int and num_neurons > 0, \
                'number of neurons specified in model architecture needs to be a positive integer'

            # input layer
            if i == 0:
                model.add(Dense(num_neurons, input_dim=self.input_dim, activation='relu'))

            # hidden layers
            else:
                model.add(Dense(num_neurons, activation='relu'))

        # output layer
        model.add(Dense(self.output_dim, activation='linear'))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate))
        return model

    def _get_argmax_action(self, state):
        """
        :param state: current state of the robot
        :return: the action that is associated with the largest Q value
        """

        argmax_action = None
        maxQ = -float("inf")
        for action in self.actions:
            input_data = np.asarray(state + action).reshape(self.output_dim, self.input_dim)
            Q = self.model.predict(input_data)
            if Q > maxQ:
                maxQ = Q
                argmax_action = action
        return argmax_action

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def choose_action(self, state, epsilon_greedy=False):
        """
        epsilon-greedy approach for choosing an action and transition into next state
        returns the next state, reward resulting from the chosen action
        """

        if epsilon_greedy:
            if np.random.rand() <= self.epsilon:
                print('random actions')

                # choose random action
                return random.choice(self.actions)

            else:
                print('argmax')
                return self._get_argmax_action(state)

        else:
            return self._get_argmax_action(state)

    def act(self, action):

        # transition into next state
        # print('act state: {s}'.format(s=robot.state))
        # print('act action: {s}'.format(s=action))
        old_x, old_a1, old_a2 = self.robot.x, self.robot.a1, self.robot.a2
        next_state = self.robot.move(action=action)
        # print('act state after: {s}'.format(s=next_state))

        # calculate reward
        new_x, new_a1, new_a2, theta = self.robot.x, self.robot.a1, self.robot.a2, self.robot.theta
        reward = self.reward_function(old_x=old_x,
                                      old_a1=old_a1,
                                      old_a2=old_a2,
                                      new_x=new_x,
                                      new_a1=new_a2,
                                      new_a2=new_a2,
                                      theta=theta)

        return reward, next_state

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        losses = []
        Q_targets = []
        for state, action, reward, next_state in minibatch:

            # find max Q for next state
            Q_prime = float('-inf')
            for next_action in self.actions:
                next_input = np.asarray(next_state + next_action).reshape(self.output_dim, self.input_dim)
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
            input_data = np.asarray(state + action).reshape(self.output_dim, self.input_dim)
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

    def policy_rollout(self, path, t_interval=1, timesteps=200):
        robot = SwimmingRobot(a1=0, a2=0, t_interval=t_interval)
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
        print('Beginning Policy Rollout')
        try:
            for i in range(timesteps):
                # rollout
                state = robot.state
                print('In', i + 1, 'th iteration the initial state is: ', state)
                old_x = robot.x
                action = self.choose_action(state)
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

'''
@junyaoshi todo: fix this function
'''
def perform_DQN(agent, episodes, iterations, path, batch_size=4, C=30, t_interval=1, randomize_theta=False):
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
            robot = SwimmingRobot(a1=0, a2=0, theta=theta, t_interval=t_interval)
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
                # robot_after_transition, \
                reward, next_state = agent.act(robot=robot, action=action,
                                                                       c_x=50, c_joint=0, c_zero_x=50, c_theta=5)
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
    TRIAL_NAME = 'DQN_Swimming_w_theta_largest_action'
    TRIAL_NUM = 24
    PATH = 'Trials/' + TRIAL_NAME + '_Trial_' + str(TRIAL_NUM) + "_" + TIMESTAMP

    # create directory
    os.mkdir(PATH)
    os.chmod(PATH, 0o0777)

    # set some variables

    episodes = 200
    iterations = 1000
    total_iterations = episodes * iterations
    memory_size = total_iterations//50
    C = total_iterations//10000

    # 0.99996 for 30000 iterations
    # 0.999 for 1000 iterations
    # 0.99987 for 10000 iterations
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
    agent = DQNAgent(gamma=0.99,
                     epsilon=1.0,
                     epsilon_min=0.1,
                     epsilon_decay=0.999999,
                     memory_size=memory_size,
                     actions_params=(-pi/8, pi/8, pi/8),
                     learning_rate=2e-4)

    # Perform DQN
    learning_results = perform_DQN(agent=agent,
                                   path=PATH,
                                   episodes=episodes,
                                   iterations=iterations,
                                   batch_size=8,
                                   C=200,
                                   t_interval=8)
    agent, num_episodes, avg_rewards, std_rewards, avg_losses, std_losses, avg_Qs, std_Qs = learning_results

    # Loss Plot
    make_loss_plot(num_episodes, avg_losses, std_losses, path=PATH)

    # Learning Curve Plot
    make_learning_plot(num_episodes, avg_rewards, std_rewards, path=PATH)

    # Make Q Plot
    make_Q_plot(num_episodes, avg_Qs, std_Qs, path=PATH)

    # Policy Rollout
    policy_rollout(agent=agent, path=PATH, t_interval=8)

