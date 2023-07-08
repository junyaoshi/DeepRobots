import random
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class QNetwork(nn.Module):
    def __init__(self, params):
        super(QNetwork, self).__init__()
        self.first_layer = params['first_layer_size']
        self.second_layer = params['second_layer_size']
        world_size = params['world_size']
        self.f1 = nn.Linear(world_size ** 2 + 4, self.first_layer)
        self.f2 = nn.Linear(self.first_layer, self.second_layer)
        self.f3 = nn.Linear(self.second_layer, 1)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        return self.f3(x)

class DQNAgent():
    def __init__(self, params):
        super().__init__()
        self.gamma = params['gamma']
        self.short_memory = np.array([])
        self.memory = collections.deque(maxlen=params['memory_size'])
        self.epsilon = params['epsilon']
        self.epsilon_decay = params['epsilon_decay']
        self.epsilon_minimum = params['epsilon_minimum']
        self.world_size = params['world_size']
        self.model = QNetwork(params)
        self.model.to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), weight_decay=params['weight_decay'], lr=params['learning_rate'])

    def remember(self, state_with_action, reward, next_state, is_done):
        """
        Store the <state, action, reward, next_state> tuple in a 
        memory buffer for replay memory.
        """
        self.memory.append((state_with_action, reward, next_state, is_done))

    def replay_mem(self, batch_size):
        """
        Replay memory.
        """
        if len(self.memory) > batch_size:
            minibatch = random.sample(self.memory, batch_size)
        else:
            minibatch = self.memory
        for state_with_action, reward, next_state, is_done in minibatch:
            self.model.train()
            torch.set_grad_enabled(True)
            state_tensor = torch.tensor(np.array(state_with_action)[np.newaxis, :], dtype=torch.float32, requires_grad=True).to(DEVICE)
            output = self.model(state_tensor[0])
            target = self.get_target(reward, next_state, is_done)
            loss = (output[0] - target) ** 2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_minimum)

    def get_possible_states_with_action(self, state):
        N_state = state + (1,0,0,0)
        E_state = state + (0,1,0,0)
        W_state = state + (0,0,1,0)
        S_state = state + (0,0,0,1)
        states = []
        index = state.index(1)
        position = ((int)(index % self.world_size), (int)(index / self.world_size))
        if position[0] > 0:
            states.append(W_state)
        if position[1] > 0:
            states.append(S_state)
        if position[0] < self.world_size - 1:
            states.append(E_state)
        if position[1] < self.world_size - 1:
            states.append(N_state)
        return states


    def include_action(self, state):
        states = self.get_possible_states_with_action(state)
        if random.uniform(0, 1) < self.epsilon:
            return states[np.random.randint(0,len(states))]

        with torch.no_grad():
            max_index = None
            max_value = None
            for index, state in enumerate(states):
                state_tensor = torch.tensor(np.array(state)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
                value = self.model(state_tensor[0])[0]
                if max_value == None or value > max_value:
                    max_value = value
                    max_index = index
            return states[max_index]

    def get_target(self, reward, next_state, is_done):
        if is_done is True:
            return reward
        states = self.get_possible_states_with_action(next_state)
        """
        Return the appropriate TD target depending on the type of the agent
        """
        with torch.no_grad():
            max_value = None
            for state in states:
                state_tensor = torch.tensor(np.array(state)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
                value = self.model(state_tensor[0])[0]
                if max_value == None or value > max_value:
                    max_value = value
            return reward + self.gamma * max_value