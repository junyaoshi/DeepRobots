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
        self.action_bins = params['action_bins']
        self.f1 = nn.Linear(params['state_size'], self.first_layer) # theta
        self.f2 = nn.Linear(self.first_layer, self.second_layer)
        self.f3 = nn.Linear(self.second_layer, self.action_bins ** 2) # phidot, psidot actions

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = self.f3(x)
        return x

class DQNAgent():
    def __init__(self, params):
        super().__init__()
        self.gamma = params['gamma']
        self.memory = collections.deque(maxlen=params['memory_size'])
        self.action_bins = params['action_bins']
        self.epsilon = params['epsilon']
        self.epsilon_decay = params['epsilon_decay']
        self.epsilon_minimum = params['epsilon_minimum']
        self.model = QNetwork(params)
        self.model.to(DEVICE)
        self.target_model = QNetwork(params)
        self.target_model.to(DEVICE)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model_update_iterations = params['target_model_update_iterations']
        self.optimizer = optim.Adam(self.model.parameters(), weight_decay=params['weight_decay'], lr=params['learning_rate'])
        self.current_iteration = 0

    def on_new_sample(self, state, action, reward, next_state):
        """
        Store the <state, action, reward, next_state> tuple in a 
        memory buffer for replay memory.
        """
        self.memory.append((state, action, reward, next_state))

    def on_finished(self):
        pass

    def on_episode_start(self, episode_index):
        pass

    def replay_mem(self, batch_size):
        if len(self.memory) > batch_size:
            minibatch = random.sample(self.memory, batch_size)
        else:
            minibatch = self.memory

        self.model.train()
        torch.set_grad_enabled(True)
        self.optimizer.zero_grad()
        states, actions, rewards, next_states = zip(*minibatch)
        states_tensor = torch.tensor(states, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            targets = self.get_targets(rewards, next_states)
        outputs = self.model.forward(states_tensor)
        outputs_selected = outputs[torch.arange(len(minibatch)), actions]
        loss = F.mse_loss(outputs_selected, targets)
        loss.backward()
        self.optimizer.step()
        
        # epsilon decay
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_minimum)
        self.current_iteration = self.current_iteration + 1
        if self.current_iteration % self.target_model_update_iterations == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def select_action_index(self, state, apply_epsilon_random):
        if (apply_epsilon_random == True and random.uniform(0, 1) < self.epsilon):
            return np.random.choice(self.action_bins ** 2) # phidot, psidot actions

        with torch.no_grad():
            state_tensor = torch.tensor(np.array(state)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
            prediction = self.model(state_tensor)
            return np.argmax(prediction.detach().cpu().numpy()[0])

    def get_targets(self, rewards, next_states):
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(DEVICE)
            q_values_next_states = self.target_model.forward(next_states_tensor)
            max_values, _ = torch.max(q_values_next_states, dim=1)
            targets = rewards_tensor + self.gamma * max_values # Q-Learning is off-policy
        return targets

    def on_terminated(self):
        pass