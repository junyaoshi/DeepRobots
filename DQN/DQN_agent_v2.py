# change current working directory to parent directory
import os, sys

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# os.chdir(parentdir)

# import libraries
import traceback
import sys

sys.path.append("/home/pi/DeepRobots")

import matplotlib

matplotlib.use('Agg')
import datetime
import random
import numpy as np
import csv, json
from copy import deepcopy
from pprint import pprint
from collections import deque
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.optimizers import RMSprop
from utils.csv_generator import generate_csv
from math import pi
from utils.graphing_helper import make_learning_plot, \
    make_loss_plot, make_Q_plot, make_rollout_graphs, make_rollout_graphs_physical
from utils.learning_helper import save_learning_data


# Define DQNAgent Class
class DQN_Agent:

    def __init__(self,
                 robot,
                 reward_function,
                 trial_name,
                 trial_num,
                 episodes,
                 iterations,
                 network_update_freq,
                 params,
                 check_singularity=False,
                 is_physical_robot=False,
                 actions_params=(-pi / 8, pi / 8, pi / 8),
                 model_architecture=(50, 10),
                 memory_size=500,
                 memory_buffer_coef=20,
                 randomize_theta=False,
                 batch_size=8,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_min=0.1,
                 epsilon_decay=0.9954,  # 0.9995
                 learning_rate=0.001):

        """
        :param reward_function:
        takes the following parameters as an input:
            old_x, old_a1, old_a2,
            new_x, new_a1, new_a2, theta,
            c_x, c_joint, c_zero_x, c_theta, reward_theta
        it returns the reward for these current moves
        :param file_path: the directory where all the learning results are saved
        :param check_singularity: for wheeled robot, we need to check singularity before choosing an action
        """

        self.trial_name = trial_name
        self.trial_num = trial_num
        timestamp = str(datetime.datetime.now()).replace(' ', '-').replace(':', '-')[:-10]
        if not os.path.exists('Trials'):
            os.mkdir('Trials')
            os.chmod('Trials', 0o0777)
        self.file_path = 'Trials/Trial_' + str(self.trial_num) + "_" + self.trial_name + "_" + timestamp
        self.params = params

        # initialize DQN parameters
        self.robot_in_action = robot
        self._robot_original_state = robot.state
        self.check_singularity = check_singularity
        self.is_physical_robot = is_physical_robot
        self.actions_params = actions_params
        self.actions = self._get_actions()
        self.reward_function = reward_function
        self.episodes = episodes
        self.iterations = iterations
        self.network_update_freq = network_update_freq
        self.randomize_theta = randomize_theta
        self.memory_size = memory_size
        self.memory_buffer_coef = memory_buffer_coef
        self.memory = deque(maxlen=self.memory_size)
        self.batch_size = batch_size
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # initialize neural network parameters
        self.input_dim = len(robot.state)
        self.output_dim = len(self.actions)
        self.model_architecture = model_architecture
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.model_clone = self._build_model()
        self.model_clone.set_weights(self.model.get_weights())

    def _get_actions(self):
        """
        :return: a list of action space values in tuple format (a1dot, a2dot)
        """

        lower_limit, upper_limit, interval = self.actions_params
        upper_limit += (interval / 10)  # to ensure the range covers the rightmost value in the loop
        r = np.arange(lower_limit, upper_limit, interval)
        if self.is_physical_robot:
            actions = [(int(i), int(j)) for i in r for j in r]
        else:
            actions = [(i, j) for i in r for j in r]

        # remove a1dot = 0, a2dot = 0 from action space
        if self.is_physical_robot:
            actions.remove((0, 0))
        else:
            actions.remove((0.0, 0.0))
        pprint('The actions initialized are: {}'.format(actions))
        return actions

    def _build_model(self):

        assert len(self.model_architecture) > 0, 'model architecture cannot be an empty list'

        # Neural Net for Deep-Q learning Model
        model = Sequential()

        # layers
        for i in range(len(self.model_architecture)):

            num_neurons = self.model_architecture[i]
            assert isinstance(num_neurons, int) and num_neurons > 0, \
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

    def _get_argmax_action(self, state, epsilon_greedy):
        """
        :param state: current state of the robot
        :return: the action that is associated with the largest Q value, the largest Q_value
        """

        argmax_action = None
        maxQ = -float("inf")
        input_state = np.asarray(state).reshape(1, self.input_dim)
        Qs = self.model.predict(input_state)
        argmax = np.argmax(Qs)
        max_Q = Qs[0, argmax]
        argmax_action = self.actions[argmax]
        if self.check_singularity and epsilon_greedy:
            temp_robot = deepcopy(self.robot_in_action)
            _, a1, a2 = temp_robot.move(argmax_action)
            while abs(a1 - a2) <= 0.00001:
                Qs[argmax] = -np.inf
                argmax = np.argmax(Qs)
                max_Q = Qs[argmax]
                argmax_action = self.actions[argmax]
                temp_robot = deepcopy(self.robot_in_action)
                _, a1, a2 = temp_robot.move(argmax_action)
        return argmax_action, max_Q

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

                if self.check_singularity:
                    while True:
                        chosen_action = random.choice(self.actions)
                        temp_robot = deepcopy(self.robot_in_action)
                        _, a1, a2 = temp_robot.move(chosen_action)
                        if abs(a1 - a2) > 0.00001:  # check for singularity
                            break

                    del temp_robot
                    return chosen_action
                else:
                    return random.choice(self.actions)

            else:
                print('argmax')
                return self._get_argmax_action(state, epsilon_greedy=epsilon_greedy)[0]

        else:
            return self._get_argmax_action(state, epsilon_greedy=epsilon_greedy)[0]

    def act(self, action):

        # transition into next state

        reward, robot = self.reward_function(self.robot_in_action, action)
        self.robot_in_action = robot
        next_state = self.robot_in_action.state

        return reward, next_state

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        losses = []
        Q_targets = []
        for state, action, reward, next_state in minibatch:

            input_state = np.asarray(state).reshape(1, self.input_dim)
            _, Q_prime = self._get_argmax_action(state=next_state, epsilon_greedy=False)
            Q_target = reward + self.gamma * Q_prime

            Qs_target = self.model.predict(input_state)
            Qs_target[0, self.actions.index(action)] = Q_target

            # perform a gradient descent step
            loss = self.model.train_on_batch(input_state, Qs_target)
            losses.append(loss)
            Q_targets.append(Q_target)

        # update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        # return the average loss of this experience replay
        return sum(losses) / len(losses), sum(Q_targets) / len(Q_targets)

    def perform_DQN(self):
        """
        :param agent: the RL agent
        :param batch_size: size of minibatch sampled from replay buffer
        :param C: network update frequency
        :return: agent, and other information about DQN
        """

        models_path = self.file_path + "/models"
        robot_data_path = self.file_path + "/robot_data"
        if not os.path.exists(models_path):
            os.mkdir(models_path)
            os.chmod(models_path, 0o0777)
        if not os.path.exists(robot_data_path):
            os.mkdir(robot_data_path)
            os.chmod(robot_data_path, 0o0777)

        avg_losses = []
        std_losses = []
        avg_rewards = []
        std_rewards = []
        avg_Qs = []
        std_Qs = []
        num_episodes = []

        try:
            # loop through each episodes
            for e in range(1, self.episodes + 1):

                # save model
                if e % (self.episodes / 10) == 0:
                    self.save_model(models_path, e)

                self.robot_in_action.set_state(*self._robot_original_state)
                self.robot_in_action.x = 0
                self.robot_in_action.y = 0
                assert self.robot_in_action.state == self._robot_original_state, 'there is a problem with reset'

                # theta = random.uniform(-pi/4, pi/4) if self.randomize_theta else 0
                # self.robot_in_action.theta = theta
                state = self.robot_in_action.state
                rewards = []
                losses = []
                Qs = []
                robot_params = []
                if self.is_physical_robot:
                    robot_param = [float(self.robot_in_action.encoder_val),
                                   float(self.robot_in_action.a1),
                                   float(self.robot_in_action.a2),
                                   self.robot_in_action.a1dot,
                                   self.robot_in_action.a2dot]
                else:
                    robot_param = [self.robot_in_action.x,
                                   self.robot_in_action.y,
                                   self.robot_in_action.theta,
                                   float(self.robot_in_action.a1),
                                   float(self.robot_in_action.a2),
                                   self.robot_in_action.a1dot,
                                   self.robot_in_action.a2dot]
                robot_params.append(robot_param)

                # loop through each iteration
                for i in range(1, self.iterations + 1):
                    # print('In ', e, ' th epsiode, ', i, ' th iteration, the initial state is: ', state)
                    action = self.choose_action(state, epsilon_greedy=True)
                    print('In {}th epsiode {}th iteration, the chosen action is: {}'.format(e, i, action))
                    reward, next_state = self.act(action=action)
                    if self.is_physical_robot:
                        robot_param = [float(self.robot_in_action.encoder_val),
                                       float(self.robot_in_action.a1),
                                       float(self.robot_in_action.a2),
                                       self.robot_in_action.a1dot,
                                       self.robot_in_action.a2dot]
                    else:
                        robot_param = [self.robot_in_action.x,
                                       self.robot_in_action.y,
                                       self.robot_in_action.theta,
                                       float(self.robot_in_action.a1),
                                       float(self.robot_in_action.a2),
                                       self.robot_in_action.a1dot,
                                       self.robot_in_action.a2dot]
                    robot_params.append(robot_param)
                    print('The reward is: {}'.format(reward))
                    rewards.append(reward)
                    # print('In ', e, ' th epsiode, ', i, ' th iteration, the state after transition is: ', next_state)
                    self.remember(state, action, reward, next_state)
                    state = next_state
                    if len(self.memory) > self.memory_size / self.memory_buffer_coef:
                        loss, Q = self.replay()
                        losses.append(loss)
                        Qs.append(Q)
                        print('The average loss is: {}'.format(loss))
                        print('The average Q is: {}'.format(Q))

                    if i % self.network_update_freq == 0:
                        self.update_model()

                num_episodes.append(e)
                avg_rewards.append(np.mean(rewards))
                std_rewards.append(np.std(rewards))
                avg_losses.append(np.mean(losses))
                std_losses.append(np.std(losses))
                avg_Qs.append(np.mean(Qs))
                std_Qs.append(np.std(Qs))

                # self.robot_in_action = None

                generate_csv(robot_params, robot_data_path + "/episode {}.csv".format(e))

        except Exception as e:
            traceback.print_exc()

        finally:

            # save learning data
            learning_path = self.file_path + "/learning_results"
            if not os.path.exists(learning_path):
                os.mkdir(learning_path)
                os.chmod(learning_path, 0o0777)

            save_learning_data(learning_path,
                               num_episodes,
                               avg_rewards,
                               std_rewards,
                               avg_losses,
                               std_losses,
                               avg_Qs,
                               std_Qs)
            return num_episodes, avg_rewards, std_rewards, avg_losses, std_losses, avg_Qs, std_Qs

    def policy_rollout(self, timesteps=200, random_start=False):
        rollout_path = self.file_path + "/policy_rollout_results"
        if not os.path.exists(rollout_path):
            os.mkdir(rollout_path)
            os.chmod(rollout_path, 0o0777)

        reps = 5 if random_start else 1
        for i in range(reps):
            rep_path = rollout_path + "/" + str(reps)
            if not os.path.exists(rep_path):
                os.mkdir(rep_path)
                os.chmod(rep_path, 0o0777)

            self.robot_in_action.set_state(*self._robot_original_state)
            self.robot_in_action.x = 0
            self.robot_in_action.y = 0
            assert self.robot_in_action.state == self._robot_original_state, 'there is a problem with reset'

            if self.is_physical_robot:
                encoders = [self.robot_in_action.encoder_val]
            else:
                xs = [self.robot_in_action.x]
                ys = [self.robot_in_action.y]
                thetas = [self.robot_in_action.theta]
            a1s = [self.robot_in_action.a1]
            a2s = [self.robot_in_action.a2]
            steps = [0]
            if random_start:
                self.robot_in_action.randomize_state(enforce_opposite_angle_signs=True)
            robot_params = []

            if self.is_physical_robot:
                robot_param = [float(self.robot_in_action.encoder_val),
                               float(self.robot_in_action.a1),
                               float(self.robot_in_action.a2),
                               self.robot_in_action.a1dot,
                               self.robot_in_action.a2dot]
            else:
                robot_param = [self.robot_in_action.x,
                               self.robot_in_action.y,
                               self.robot_in_action.theta,
                               float(self.robot_in_action.a1),
                               float(self.robot_in_action.a2),
                               self.robot_in_action.a1dot,
                               self.robot_in_action.a2dot]
            robot_params.append(robot_param)
            print('Beginning Policy Rollout')
            try:
                for i in range(timesteps):
                    # rollout
                    state = self.robot_in_action.state
                    print('In', i + 1, 'th iteration the initial state is: ', state)
                    if not self.is_physical_robot:
                        old_x = self.robot_in_action.x
                    action = self.choose_action(state)
                    print('In', i + 1, 'th iteration the chosen action is: ', action)
                    self.robot_in_action.move(action=action)

                    if self.is_physical_robot:
                        displacement = self.robot_in_action.encoder_val
                    else:
                        displacement = self.robot_in_action.x - old_x
                    print('In', i + 1, 'th iteration, the robot moved ', displacement, ' in x direction')

                    # add values to lists
                    if self.is_physical_robot:
                        encoders.append(self.robot_in_action.encoder_val)
                    else:
                        xs.append(self.robot_in_action.x)
                        ys.append(self.robot_in_action.y)
                        thetas.append(self.robot_in_action.theta)
                    a1s.append(self.robot_in_action.a1)
                    a2s.append(self.robot_in_action.a2)
                    steps.append(i + 1)
                    if self.is_physical_robot:
                        robot_param = [float(self.robot_in_action.encoder_val),
                                       float(self.robot_in_action.a1),
                                       float(self.robot_in_action.a2),
                                       self.robot_in_action.a1dot,
                                       self.robot_in_action.a2dot]
                    else:
                        robot_param = [self.robot_in_action.x,
                                       self.robot_in_action.y,
                                       self.robot_in_action.theta,
                                       float(self.robot_in_action.a1),
                                       float(self.robot_in_action.a2),
                                       self.robot_in_action.a1dot,
                                       self.robot_in_action.a2dot]
                    robot_params.append(robot_param)

            except ZeroDivisionError as e:
                print(str(e), 'occured during policy rollout')

            # self.robot_in_action = None

            # plotting
            if self.is_physical_robot:
                make_rollout_graphs_physical(encoders, a1s, a2s, steps, path=rep_path)
            else:
                make_rollout_graphs(xs, ys, thetas, a1s, a2s, steps, path=rep_path)
            generate_csv(robot_params, rep_path + "/policy_rollout.csv")

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

    def run(self):

        # create directory
        os.mkdir(self.file_path)
        os.chmod(self.file_path, 0o0777)

        # save params
        param_fname = os.path.join(self.file_path, "params.json")
        with open(param_fname, "w") as f:
            json.dump(self.params, f, indent=4, sort_keys=True)

        # Perform DQN
        learning_results = self.perform_DQN()

        num_episodes, avg_rewards, std_rewards, avg_losses, std_losses, avg_Qs, std_Qs = learning_results

        learning_path = self.file_path + "/learning_results"
        if not os.path.exists(learning_path):
            os.mkdir(learning_path)
            os.chmod(learning_path, 0o0777)

        # Loss Plot
        make_loss_plot(num_episodes, avg_losses, std_losses, path=learning_path)

        # Learning Curve Plot
        make_learning_plot(num_episodes, avg_rewards, std_rewards, path=learning_path)

        # Make Q Plot
        make_Q_plot(num_episodes, avg_Qs, std_Qs, path=learning_path)

        # Policy Rollout
        self.policy_rollout(timesteps=40)




