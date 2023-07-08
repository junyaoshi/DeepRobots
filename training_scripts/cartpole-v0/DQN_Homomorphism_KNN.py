import random
import numpy as np
import math
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
        self.f1 = nn.Linear(4, self.first_layer)
        self.f2 = nn.Linear(self.first_layer, self.second_layer)
        self.f3 = nn.Linear(self.second_layer, 2)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = self.f3(x)
        return x

class RewardPredictor(nn.Module):
    def __init__(self, input_size, output_size=1, activation=F.relu):
        super().__init__()
        self.fc_in = nn.Linear(input_size, 64)
        self.fc_out = nn.Linear(64, output_size)
        self.act1 = activation

    def forward(self, z):
        h = self.act1(self.fc_in(z))
        r = self.fc_out(h)
        return r


class ActionEncoder(nn.Module):
    def __init__(self, n_dim, n_actions, hidden_dim=100, temp=1.):
        super().__init__()
        self.linear1 = nn.Linear(n_dim+n_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, n_dim)

    def forward(self, z, a):
        za = torch.cat([z, a], dim=1)
        za = F.relu(self.linear1(za))
        zt = self.linear2(za)
        return zt

class StateEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, mid=64, mid2=32,
                 activation=F.relu):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, mid)
        self.fc2 = nn.Linear(mid, mid2)
        self.fc3 = nn.Linear(mid2, out_dim)
        self.act = activation

    def forward(self, obs):
        h = self.act(self.fc1(obs))
        h = self.act(self.fc2(h))
        z = self.fc3(h)
        return z

class Model(nn.Module):
    def __init__(self, state_encoder, action_encoder, reward):
        super().__init__()
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.reward = reward

    def forward(self, x):
        raise NotImplementedError("This model is a placeholder")

    def train(self, boolean):
        self.state_encoder.train = boolean
        self.action_encoder.train = boolean
        self.reward.train = boolean

class DQNAgent():
    def __init__(self, params):
        super().__init__()
        self.gamma = params['gamma']
        self.short_memory = np.array([])
        self.memory = collections.deque(maxlen=params['memory_size'])
        self.epsilon = params['epsilon']
        self.epsilon_decay = params['epsilon_decay']
        self.epsilon_minimum = params['epsilon_minimum']
        self.model = QNetwork(params)
        self.model.to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), weight_decay=params['weight_decay'], lr=params['learning_rate'])
        self.params = params

        state_encoder = StateEncoder(4, params['abstract_state_space_dimmension'])
        action_encoder = ActionEncoder(params['abstract_state_space_dimmension'], params['number_of_actions'])
        rewards = RewardPredictor(params['abstract_state_space_dimmension'])
        self.abstraction_model = Model(state_encoder, action_encoder, rewards)


    def on_new_sample(self, state, action, reward, next_state, is_done):
        self.memory.append((state, action, reward, next_state, is_done))

    def replay_mem(self, batch_size, is_decay_epsilon):
        """
        Replay memory.
        """
        if len(self.memory) > batch_size:
            minibatch = random.sample(self.memory, batch_size)
        else:
            minibatch = self.memory
        symmetries_batch = self.find_symmetries(minibatch)
        for state, action, reward, next_state, is_done in minibatch:
            state = tuple(state)
            self.model.train()
            torch.set_grad_enabled(True)
            state_tensor = torch.tensor(np.array(state)[np.newaxis, :], dtype=torch.float32, requires_grad=True).to(DEVICE)
            target = self.get_target(reward, next_state, is_done)
            output = self.model.forward(state_tensor)
            self.optimizer.zero_grad()

            symmetry_loss_sum = None
            symmetry_key = state + (action,)
            if symmetry_key in symmetries_batch:
                symmetries = symmetries_batch[state + (action,)]
                for symmetry in symmetries:
                    symmetry_state, symmetry_action_index = symmetry
                    symmetry_state_tensor = torch.tensor(np.array(symmetry_state)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
                    symmetry_output = self.model.forward(symmetry_state_tensor)
                    symmetry_loss = (target - symmetry_output[0][symmetry_action_index]) ** 2
                    if symmetry_loss_sum is None:
                        symmetry_loss_sum = symmetry_loss
                    else:
                        symmetry_loss_sum += symmetry_loss
            loss = (output[0][action] - target) ** 2

            if symmetry_loss_sum != None:
                loss += self.reward_trail_symmetry_weight * symmetry_loss_sum

            loss.backward()
            self.optimizer.step()
        if is_decay_epsilon == True:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_minimum)

    def select_action_index(self, state, apply_epsilon_random, is_random_policy):
        if is_random_policy == True or (apply_epsilon_random == True and random.uniform(0, 1) < self.epsilon):
            return random.randint(0,1)

        with torch.no_grad():
            state_tensor = torch.tensor(np.array(state)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
            prediction = self.model(state_tensor)
            return np.argmax(prediction.detach().cpu().numpy()[0])

    def get_target(self, reward, next_state, is_done):
        if is_done:
            return reward
        """
        Return the appropriate TD target depending on the type of the agent
        """
        with torch.no_grad():
            next_state_tensor = torch.tensor(np.array(next_state)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
            q_values_next_state = self.model.forward(next_state_tensor[0])
            target = reward + self.gamma * torch.max(q_values_next_state) # Q-Learning is off-policy
        return target

    def find_symmetries(self, batch):
        pass

    def on_terminated(self):
        pass