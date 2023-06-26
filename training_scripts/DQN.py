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
        self.third_layer = params['third_layer_size']
        self.action_bins = params['action_bins']
        self.f1 = nn.Linear(3, self.first_layer) # theta, phi, psi states
        self.f2 = nn.Linear(self.first_layer, self.second_layer)
        self.f3 = nn.Linear(self.second_layer, self.third_layer)
        self.f4 = nn.Linear(self.third_layer, self.action_bins ** 2) # phidot, psidot actions

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = F.softmax(self.f4(x), dim=-1)
        return x

class DQNAgent():
    def __init__(self, params):
        super().__init__()
        self.gamma = params['gamma']
        self.short_memory = np.array([])
        self.memory = collections.deque(maxlen=params['memory_size'])
        self.action_bins = params['action_bins']
        self.epsilon = params['epsilon']
        self.epsilon_decay = params['epsilon_decay']
        self.epsilon_minimum = params['epsilon_minimum']
        self.target_model_update_iterations = params['target_model_update_iterations']
        self.model = QNetwork(params)
        self.model.to(DEVICE)
        self.target_model = QNetwork(params)
        self.target_model.to(DEVICE)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), weight_decay=params['weight_decay'], lr=params['learning_rate'])

    def remember(self, state, action_index, reward, next_state):
        """
        Store the <state, action, reward, next_state> tuple in a 
        memory buffer for replay memory.
        """
        self.memory.append((state, action_index, reward, next_state))

    def replay_mem(self, batch_size, current_iteration):
        """
        Replay memory.
        """
        if len(self.memory) > batch_size:
            minibatch = random.sample(self.memory, batch_size)
        else:
            minibatch = self.memory
        for state, action_index, reward, next_state in minibatch:
            self.model.train()
            torch.set_grad_enabled(True)
            state_tensor = torch.tensor(np.array(state)[np.newaxis, :], dtype=torch.float32, requires_grad=True).to(DEVICE)
            target = self.get_target(reward, next_state)
            output = self.model.forward(state_tensor)
            self.optimizer.zero_grad()
            loss = (output[0][action_index] - target) ** 2
            loss.backward()
            self.optimizer.step()
        if current_iteration % self.target_model_update_iterations == 0 and current_iteration != 0:
            self.target_model.load_state_dict(self.model.state_dict())
        if current_iteration % batch_size == 0 and current_iteration != 0:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_minimum)

    def select_action_index(self, state, apply_epsilon_random):
        if apply_epsilon_random == True and random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_bins ** 2) # phidot, psidot actions

        with torch.no_grad():
            state_tensor = torch.tensor(np.array(state)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
            prediction = self.model(state_tensor)
            return np.argmax(prediction.detach().cpu().numpy()[0])

    def get_target(self, reward, next_state):
        """
        Return the appropriate TD target depending on the type of the agent
        """
        with torch.no_grad():
            next_state_tensor = torch.tensor(np.array(next_state)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
            q_values_next_state = self.target_model.forward(next_state_tensor[0])
            target = reward + self.gamma * torch.max(q_values_next_state) # Q-Learning is off-policy
        return target