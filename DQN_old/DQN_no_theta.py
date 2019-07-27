import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ContinuousDeepRobots import ThreeLinkRobot
import datetime
import random
import copy
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from math import pi, log, sqrt

# # Define DQNAgent Class

# In[3]:


class DQNAgent:
    INPUT_DIM = 5
    OUTPUT_DIM = 1
    def __init__(self, actions_params=(-pi/16, pi/16, pi/128), memory_size=500, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.9995, learning_rate=0.001):
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.actions = self._get_actions(actions_params)
        self.model = self._build_model()

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
        model.add(Dense(15, activation='relu'))
        # output layer
        model.add(Dense(self.OUTPUT_DIM, activation = 'linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
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
        next_state = robot.move(action=action)
        # print('act state after: {s}'.format(s=next_state))
        
        # calculate reward
        a1, a2, v, a1dot, a2dot = robot.a1, robot.a2, robot.inertial_v[0][0], robot.a1dot, robot.a2dot
        if v == 0 or abs(a1dot-a2dot) != 0:
            reward = -5
        else:
            # print('a1dot: ', a1dot, 'a2dot: ', a2dot )
            # reward = v/sqrt(a1dot**2 + a2dot**2) 
            reward = v
            # print('this reward: ', reward)
        
        return robot, reward, next_state

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        losses = []
        for state, action, reward, next_state in minibatch:
            
            # perform Bellman Update (use temporal difference?)
            input_data = np.asarray(state + action).reshape(1, 5)
            # print('reward: ', reward, 'prediction: ', self.model.predict(input_data))
            Q_target = reward + self.gamma * self.model.predict(input_data)
            
            # perform a gradient descent step
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

    def save(self, name):
        self.model.save_weights(name)


# # DeepRobots

# In[4]:


EPISODES = 300
ITERATIONS = 100

# 0.99996 for 30000 iterations
# 0.999 for 1000 iterations
agent = DQNAgent(epsilon_decay=0.99995,actions_params=(-pi/4, pi/4, pi/4))
batch_size = 100
avg_losses = []
gd_iterations = [] # gradient descent iterations
gd_iteration = 0

for e in range(1, EPISODES+1):

    # save model
    if e%100 == 0:
        # serialize model to JSON
        model_json = agent.model.to_json()
        with open("models/model " + str(e) + " th epsiode "
                  + str(datetime.datetime.now()).replace(' ', '').replace(':','') + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        agent.model.save_weights("weights/model " + str(e) + " th epsiode "
                                 + str(datetime.datetime.now()).replace(' ', '').replace(':','') + ".h5")
        print("Saved model to disk")

    robot = ThreeLinkRobot(t_interval=0.5)
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
        if len(agent.memory) > batch_size:
            avg_loss = agent.replay(batch_size)
            gd_iteration += 1
            avg_losses.append(avg_loss)
            gd_iterations.append(gd_iteration)
            print('In ', e, ' th episode, ', i, ' th iteration, the average loss is: ', avg_loss)
            
        # print('\n')


# # Loss Plot

# In[5]:


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Loss vs Number of Iterations')
ax.set_xlabel('Number of Iterations')
ax.set_ylabel('Loss')
ax.plot(gd_iterations, avg_losses)
fig.savefig('images/Loss vs Number of Iterations.png')
plt.close()


# # Policy Rollout

# In[6]:


def make_graphs(xs, a1s, a2s, steps, number):

    # plotting
    fig1 = plt.figure(1)
    fig1.suptitle('Policy Rollout ' + str(number))
    ax1 = fig1.add_subplot(311)
    ax2 = fig1.add_subplot(312)
    ax3 = fig1.add_subplot(313)
    # fig2 = plt.figure(2)
    # fig2.suptitle('a1 vs a2')
    # ax4 = fig2.add_subplot(111)

    ax1.plot(steps, xs, '.-')
    ax1.set_ylabel('x')
    ax1.set_xlabel('steps')
    ax2.plot(steps, a1s, '.-')
    ax2.set_ylabel('a1')
    ax2.set_xlabel('steps')
    ax3.plot(steps, a2s, '.-')
    ax3.set_ylabel('a2')
    ax3.set_xlabel('steps')
    # ax4.plot(a1s,a2s,'.-')
    # ax4.set_xlabel('a1')
    # ax4.set_ylabel('a2')

    fig1.tight_layout()
    # fig2.tight_layout()
    fig1.tight_layout()
    fig1.savefig('images/Policy Rollout ' + str(number) + '.png')
    plt.close(fig1)

# In[9]:

TIMESTEPS = 100
for j in range(1):
    xs = []
    a1s = []
    a2s = []
    steps = []
    dx = 0
    robot = ThreeLinkRobot(t_interval=0.5)
    # robot.randomize_state(enforce_opposite_angle_signs=True)
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
            dx += (new_x-old_x)

            # add values to lists
            xs.append(dx)
            a1s.append(robot.a1)
            a2s.append(robot.a2)
            steps.append(i)
    except ZeroDivisionError as e:
        print(str(e), 'occured at ', j+1, 'th policy rollout')

    # plotting
    make_graphs(xs,a1s,a2s,steps, j+1)

