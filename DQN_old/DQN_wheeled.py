import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Robots.ContinuousDeepRobots import ThreeLinkRobot
import datetime
import random
import copy
import numpy as np
import os
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from utils.csv_generator import generate_csv
from math import pi


# # Define DQNAgent Class

class DQNAgent:
    INPUT_DIM = 5
    OUTPUT_DIM = 1
    def __init__(self, actions_params=(-pi/16, pi/16, pi/128), memory_size=1000, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.2, epsilon_decay=0.9995, learning_rate=0.0005):
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
        actions = [(i, j) for i in r for j in r]

        # remove a1dot = 0, a2dot = 0 from action space
        actions.remove((0.0,0.0))
        print(actions)
        return actions
    
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        # input layer
        model.add(Dense(15, input_dim=self.INPUT_DIM, activation='relu'))
        # hidden layers
        model.add(Dense(30, activation='relu'))
        model.add(Dense(10, activation='relu'))
        # output layer
        model.add(Dense(self.OUTPUT_DIM, activation = 'linear'))
        model.compile(loss='mse',
                      optimizer=RMSprop(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def choose_action(self, robot, state, epsilon_greedy=False):
        """
        epsilon-greedy approach for choosing an action and transition into next state
        returns the next state, reward resulting from the chosen action
        """
        chosen_action = None
        if epsilon_greedy:
            if np.random.rand() <= self.epsilon:
                print('random actions')
                # choose random action
                while True:
                    chosen_action = random.choice(self.actions)
                    temp_robot = copy.deepcopy(robot)
                    # print('current state: ', temp_robot.state)
                    # print('action in process: ', chosen_action)
                    _, a1, a2 = temp_robot.move(chosen_action)
                    if abs(a1 - a2) > 0.00001: # check for singularity
                        break
            else:
                print('argmax')
                # find the action with greatest Q value
                maxQ = -float("inf")
                for action in self.actions:
                    input_data = np.asarray(state + action).reshape(1, 5)
                    Q = self.model.predict(input_data)        
                    if Q > maxQ:
                        temp_robot = copy.deepcopy(robot)
                        # print('current state: ', temp_robot.state)
                        # print('action in process: ', action)
                        _, a1, a2 = temp_robot.move(action)
                        # print('a1 - a2 > 0.00001: ', a1 - a2 > 0.00001, '-pi/2 <= a1 <= pi/2: ', -pi/2 <= a1 <= pi/2, '-pi/2 <= a2 <= pi/2: ', -pi/2 <= a2 <= pi/2)
                        if abs(a1 - a2) > 0.00001: # check for singularity
                            maxQ = Q
                            chosen_action = action


        else:
            
            # policy rollout
            maxQ = -float("inf")
            for action in self.actions:
                input_data = np.asarray(state + action).reshape(1, 5)
                Q = self.model.predict(input_data)
                if Q > maxQ:
                    maxQ = Q
                    chosen_action = action
                     
        return chosen_action
    
    def act(self, robot, action):
        
        # transition into next state
        # print('act state: {s}'.format(s=robot.state))
        # print('act action: {s}'.format(s=action))
        old_x = robot.x
        old_theta = robot.theta
        next_state = robot.move(action=action)
        # print('act state after: {s}'.format(s=next_state))
        
        # calculate reward
        # a1, a2, a1dot, a2dot = robot.a1, robot.a2, robot.a1dot, robot.a2dot
        new_x = robot.x
        new_theta = robot.theta
        reward = 10 * (new_x-old_x)
        # find penalty
        # penalty = 10 * abs(new_theta-old_theta)
        print('reward: ', reward)
        # print('penalty: ', penalty)
        if reward == 0:
            print('incur penalty')
            reward = -10
        # else:
            # print('a1dot: ', a1dot, 'a2dot: ', a2dot )
            # reward = v/sqrt(a1dot**2 + a2dot**2) 
            # reward = v
            # print('this reward: ', reward)

        # reward += penalty
        
        return robot, reward, next_state

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        losses = []
        for state, action, reward, next_state in minibatch:
            
            # find max Q for next state
            Q_prime = float('-inf')
            for next_action in self.actions:
                next_input = np.asarray(next_state + next_action).reshape(1, 5)
                # print('reward: ', reward, 'prediction: ', self.model.predict(input_data))
                current_Q = self.model_clone.predict(next_input)
                # print('Qprime: {x}, current_Q: {y}'.format(x=Q_prime, y=current_Q))
                Q_prime = max(Q_prime, current_Q)
                # print('afterwards, Qprime: {x}'.format(x=Q_prime))

            # calculate network update target
            # print('In the end, Qprime: {x}'.format(x=Q_prime))
            Q_target = reward + self.gamma * Q_prime

            # print('Qtarget: {x}'.format(x=Q_target))
            
            # perform a gradient descent step
            input_data = np.asarray(state + action).reshape(1, 5)
            loss = self.model.train_on_batch(input_data, Q_target)
            # print('loss: {x}'.format(x=loss))
            # print('loss: ', loss, 'input: ', input_data, 'Q_target: ', Q_target)
            losses.append(loss)
            # self.model.fit(state, target_f, epochs=1, verbose=0)
            
        # update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # return the average lost of this experience replay
        return sum(losses)/len(losses)
        
    def load(self, name):
        self.model.load_weights(name)

    def update_model(self):
        self.model_clone.set_weights(self.model.get_weights())

    def save(self, name):
        self.model.save_weights(name)


# # DeepRobots

# In[4]:

TIMESTAMP = str(datetime.datetime.now())
EPISODES = 200
ITERATIONS = 100
TRIAL_NAME = 'DQN Wheeled '
TRIAL_NUM = 1
PATH = 'Trials/' + TRIAL_NAME + 'Trial ' + str(TRIAL_NUM) + " " + TIMESTAMP
os.mkdir(PATH)


# 0.99996 for 30000 iterations
# 0.999 for 1000 iterations
# 0.,99987 for 10000 iterations
# 0.99995 for 20000
agent = DQNAgent(epsilon_decay=0.99995,actions_params=(-pi/5, pi/5, pi/5))
batch_size = 4
C = 10 # network update frequency
avg_losses = []
gd_iterations = [] # gradient descent iterations
gd_iteration = 0

for e in range(1, EPISODES+1):

    # save model
    if e%100 == 0:
        # serialize model to JSON
        model_json = agent.model.to_json()
        with open(PATH + "/" + str(e) + " th episode model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        agent.save(PATH + "/" + str(e) + " th episode weights.h5")
        print("Saved model to disk")

    robot = ThreeLinkRobot(t_interval=0.25)
    # state = robot.randomize_state(enforce_opposite_angle_signs=True)
    state = robot.state
    # print(state)
    for i in range(1,ITERATIONS+1):
        # print('In ', e, ' th epsiode, ', i, ' th iteration, the initial state is: ', state)
        action = agent.choose_action(robot, state, epsilon_greedy=True)
        print('In ', e, ' th epsiode, ', i, ' th iteration, the chosen action is: ', action)
        robot_after_transition, reward, next_state = agent.act(robot, action)
        print('In ', e, ' th epsiode, ', i, ' th iteration, the reward is: ', reward)
        # print('In ', e, ' th epsiode, ', i, ' th iteration, the state after transition is: ', next_state)
        agent.remember(state, action, reward, next_state)
        state = next_state
        robot = robot_after_transition
        if len(agent.memory) > agent.memory_size/20:
            avg_loss = agent.replay(batch_size)
            gd_iteration += 1
            avg_losses.append(avg_loss)
            gd_iterations.append(gd_iteration)
            print('In ', e, ' th episode, ', i, ' th iteration, the average loss is: ', avg_loss)
        if i%C == 0:
            agent.update_model()
            
        # print('\n')


# # Loss Plot

# In[5]:


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Loss vs Number of Iterations')
ax.set_xlabel('Number of Iterations')
ax.set_ylabel('Loss')
ax.plot(gd_iterations, avg_losses)
fig.savefig(PATH + '/Loss vs Number of Iterations.png')
plt.close()


# # Policy Rollout

# In[6]:


def make_graphs(xs, ys, a1s, a2s, steps, number, trial_name):

    # plotting
    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
    fig1.suptitle('Policy Rollout ' + str(number))
    ax1 = fig1.add_subplot(311)
    ax2 = fig1.add_subplot(312)
    ax3 = fig1.add_subplot(313)
    # fig2 = plt.figure(2)
    # fig2.suptitle('a1 vs a2')
    # ax4 = fig2.add_subplot(111)
    fig2.suptitle('Policy Rollout ' + str(number) + " x vs y")
    ax4 = fig2.add_subplot(111)

    ax1.plot(steps, xs, '.-')
    ax1.set_ylabel('x')
    ax1.set_xlabel('steps')
    ax2.plot(steps, a1s, '.-')
    ax2.set_ylabel('a1')
    ax2.set_xlabel('steps')
    ax3.plot(steps, a2s, '.-')
    ax3.set_ylabel('a2')
    ax3.set_xlabel('steps')
    ax4.plot(xs, ys, '.-')
    ax4.set_ylabel('y')
    ax4.set_xlabel('x')

    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig(PATH + "/" + str(number) + ' th Policy Rollout.png')
    fig2.savefig(PATH + "/" + str(number) + ' th Policy Rollout x vs y.png')
    plt.close(fig1)
    plt.close(fig2)

# In[9]:

TIMESTEPS = 200
for j in range(1):
    robot = ThreeLinkRobot(t_interval=0.25)
    xs = [robot.x]
    ys = [robot.y]
    a1s = [robot.a1]
    a2s = [robot.a2]
    steps = [0]
    # robot.randomize_state(enforce_opposite_angle_signs=True)
    robot_params = []
    robot_param = [robot.x, robot.y, robot.theta, float(robot.a1), float(robot.a2), robot.a1dot, robot.a2dot]
    robot_params.append(robot_param)
    print('Beginning', j+1,  'th Policy Rollout')
    try:
        for i in range(TIMESTEPS):

            # rollout
            state = robot.state
            print('In', i+1, 'th iteration the initial state is: ', state)
            old_x = robot.x
            action = agent.choose_action(robot, state)
            print('In', i+1, 'th iteration the chosen action is: ', action)
            robot.move(action=action)
            new_x = robot.x
            print('In', i+1, 'th iteration, the robot moved ', new_x - old_x, ' in x direction')

            # add values to lists
            xs.append(robot.x)
            ys.append(robot.y)
            a1s.append(robot.a1)
            a2s.append(robot.a2)
            steps.append(i+1)
            robot_param = [robot.x, robot.y, robot.theta, float(robot.a1), float(robot.a2), robot.a1dot, robot.a2dot]
            robot_params.append(robot_param)

    except ZeroDivisionError as e:
        print(str(e), 'occured at ', j+1, 'th policy rollout')

    # plotting
    make_graphs(xs, ys, a1s, a2s, steps, j+1, trial_name=TRIAL_NAME)
    generate_csv(robot_params, PATH + "/" + str(j+1) + " th rollout.csv")

