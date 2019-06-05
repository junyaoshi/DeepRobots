import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Robots.ContinuousSwimmingBot_body import SwimmingRobot
import datetime
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
from csv_generator import generate_csv
from math import pi, log, sqrt

# # Define DQNAgent Class


class DQNAgent:
    INPUT_DIM = 4
    OUTPUT_DIM = 1
    def __init__(self, actions_params=(-pi/16, pi/16, pi/128), memory_size=500, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.2, epsilon_decay=0.9995, learning_rate=0.001):
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
                    input_data = np.asarray(state + action).reshape(1, 4)
                    Q = self.model.predict(input_data)        
                    if Q > maxQ:
                        maxQ = Q
                        chosen_action = action

        else:
            
            # policy rollout
            maxQ = -float("inf")
            for action in self.actions:
                input_data = np.asarray(state + action).reshape(1, 4)
                Q = self.model.predict(input_data)
                if Q > maxQ:
                    maxQ = Q
                    chosen_action = action
                     
        return chosen_action
    
    def act(self, robot, action):
        
        # transition into next state
        # print('act state: {s}'.format(s=robot.state))
        # print('act action: {s}'.format(s=action))
        old_body_x = robot.body_x
        next_state = robot.move(action=action)
        # print('act state after: {s}'.format(s=next_state))
        
        # calculate reward
        # a1, a2, a1dot, a2dot = robot.a1, robot.a2, robot.a1dot, robot.a2dot
        new_body_x = robot.body_x
        reward = 10 * (new_body_x-old_body_x)
        if reward == 0:
            print('incur penalty')
            reward = -10

        print('reward: ', reward)
        
        return robot, reward, next_state

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        losses = []
        for state, action, reward, next_state in minibatch:
            
            # find max Q for next state
            Q_prime = float('-inf')
            for next_action in self.actions:
                next_input = np.asarray(next_state + next_action).reshape(1, 4)
                # print('reward: ', reward, 'prediction: ', self.model.predict(input_data))
                current_Q = self.model_clone.predict(next_input)
                # print('Qprime: {x}, current_Q: {y}'.format(x=Q_prime, y=current_Q))
                Q_prime = max(Q_prime, current_Q)
                # print('afterwards, Qprime: {x}'.format(x=Q_prime))

            # print('Q prime: ', Q_prime)
            # calculate network update target
            # print('In the end, Qprime: {x}'.format(x=Q_prime))
            Q_target = reward + self.gamma * Q_prime

            # print('Qtarget: {x}'.format(x=Q_target))
            
            # perform a gradient descent step
            input_data = np.asarray(state + action).reshape(1, 4)
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

    def update_model(self):
        self.model_clone.set_weights(self.model.get_weights())


# # DeepRobots

# In[4]:


EPISODES = 300
ITERATIONS = 100
TRIAL_NAME = ' Swimming Body '

# 0.99996 for 30000 iterations
# 0.999 for 1000 iterations
# 0.,99987 for 10000 iterations
# 0.99995 for 20000
agent = DQNAgent(epsilon_decay=0.99996,actions_params=(-pi/5, pi/5, pi/5))
batch_size = 8
C = 15 # network update frequency
avg_losses = []
gd_iterations = [] # gradient descent iterations
gd_iteration = 0

for e in range(1, EPISODES+1):

    # save model
    if e%100 == 0:
        # serialize model to JSON
        model_json = agent.model.to_json()
        with open("NN model checkpoints/models/Trial" + TRIAL_NAME + "model " + str(e) + "th epsiode "
                  + str(datetime.datetime.now()).replace(' ', '').replace(':','') + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        agent.save("NN model checkpoints/weights/Trial" + TRIAL_NAME + "model " + str(e) + " th epsiode "
                   + str(datetime.datetime.now()).replace(' ', '').replace(':','') + ".h5")
        print("Saved model to disk")

    num = np.random.rand()
    if num < 0.2:
        print('Normal robot!')
        robot = SwimmingRobot(t_interval=0.5)
    elif num < 0.4:
        print('edge case 1!')
        robot = SwimmingRobot(a1=-pi/2, a2=pi/2, t_interval=0.5)
    elif num < 0.6:
        print('edge case 2!')
        robot = SwimmingRobot(a1=-pi/2, a2=-pi/2, t_interval=0.5)
    elif num < 0.8:
        print('edge case 3!')
        robot = SwimmingRobot(a1=pi/2, a2=-pi/2, t_interval=0.5)
    else:
        print('edge case 4')
        robot = SwimmingRobot(a1=pi/2, a2=pi/2, t_interval=0.5)

    # state = robot.randomize_state(enforce_opposite_angle_signs=True)
    state = robot.state
    # print(state)
    for i in range(1,ITERATIONS+1):
        # print('In ', e, ' th epsiode, ', i, ' th iteration, the initial state is: ', state)
        action = agent.choose_action(state, epsilon_greedy=True)
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
fig.savefig('images/Trial' + TRIAL_NAME + 'Loss vs Number of Iterations '
            + str(datetime.datetime.now()).replace(' ', '').replace(':','') + '.png')
plt.close()


# # Policy Rollout

# In[6]:


def make_graphs(xs, ys, a1s, a2s, steps, number, trial_name):

    # plotting
    fig1 = plt.figure(1)
    fig1.suptitle('Policy Rollout ' + str(number))
    ax1 = fig1.add_subplot(311)
    ax2 = fig1.add_subplot(312)
    ax3 = fig1.add_subplot(313)
    # fig2 = plt.figure(2)
    # fig2.suptitle('a1 vs a2')
    # ax4 = fig2.add_subplot(111)

    ax1.plot(xs, ys, '.-')
    ax1.set_ylabel('ys')
    ax1.set_xlabel('xs')
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
    fig1.savefig('images/Trial Swimming ' + trial_name + 'Policy Rollout ' + str(number) + ' '
                 + str(datetime.datetime.now()).replace(' ', '').replace(':','') + '.png')
    plt.close(fig1)

# In[9]:

TIMESTEPS = 200
for j in range(1):
    dx = 0
    robot = SwimmingRobot(t_interval=0.5)
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
            old_x, old_y = robot.x, robot.y
            action = agent.choose_action(state)
            print('In', i+1, 'th iteration the chosen action is: ', action)
            robot.move(action=action)
            new_x, new_y = robot.x, robot.y
            print('In', i+1, 'th iteration, the robot moved ', sqrt((new_x-old_x) ** 2 + (new_y-old_y) ** 2))

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
    generate_csv(robot_params, "Trial" + TRIAL_NAME + str(j+1) + " th rollout "
                  + str(datetime.datetime.now()).replace(' ', '').replace(':','') + ".csv")

