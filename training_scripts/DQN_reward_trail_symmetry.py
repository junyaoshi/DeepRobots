import random
import numpy as np
from operator import add
import collections
from random import randint
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class RewardTreeNode:
    def __init__(self):
        self.children = {}
        self.occurrences = []

class DQNAgent(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.gamma = params['gamma']
        self.short_memory = np.array([])
        self.first_layer = params['first_layer_size']
        self.second_layer = params['second_layer_size']
        self.third_layer = params['third_layer_size']
        self.memory = collections.deque(maxlen=params['memory_size'])
        self.action_bins = params['action_bins']
        self.optimizer = None
        self.reward_trail_length = params['reward_trail_length']
        self.reward_trail_reward_decimals = params['reward_trail_reward_decimals']
        self.reward_trail_state_decimals = params['reward_trail_state_decimals']
        self.reward_trail = []
        self.reward_tree = RewardTreeNode()
        self.network()
          
    def network(self):
        # Layers
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

    def remember(self, state, action_index, reward, next_state):
        """
        Store the <state, action, reward, next_state> tuple in a 
        memory buffer for replay memory.
        """
        self.memory.append((state, action_index, reward, next_state))

    def replay_mem(self, batch_size):
        """
        Replay memory.
        """
        if len(self.memory) > batch_size:
            minibatch = random.sample(self.memory, batch_size)
        else:
            minibatch = self.memory
        for state, action_index, reward, next_state in minibatch:
            self.train_short_memory(state, action_index, reward, next_state)

    def get_target(self, reward, next_state):
        """
        Return the appropriate TD target depending on the type of the agent
        """
        next_state_tensor = torch.tensor(np.array(next_state)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
        q_values_next_state = self.forward(next_state_tensor[0])
        target = reward + self.gamma * torch.max(q_values_next_state) # Q-Learning is off-policy
        return target

    def update_reward_history_tree(self, state, action_index, reward):
        reward = round(reward, self.reward_trail_reward_decimals)
        state = tuple(round(x,self.reward_trail_state_decimals) for x in state)
        self.reward_trail.append((state, action_index, reward))
        if len(self.reward_trail) <= self.reward_trail_length:
            return
        updating_state, updating_action_index, updating_reward = self.reward_trail.pop(0)
        node = self.reward_tree
        for item in self.reward_trail:
            trail_reward = item[2]
            if trail_reward not in node.children:
                node.children[trail_reward] = RewardTreeNode()
            node = node.children[trail_reward]
            is_occurrences_exist = False
            for index, item in enumerate(node.occurrences):
                occurrence_state, occurrence_action_index, occurrence_count = item
                if occurrence_state != updating_state or occurrence_action_index != updating_action_index:
                    continue
                node.occurrences[index] = (occurrence_state, occurrence_action_index, occurrence_count + 1)
                is_occurrences_exist = True
                break
            if is_occurrences_exist == False:
                node.occurrences.append((updating_state,updating_action_index,1))

    def train_short_memory(self, state, action_index, reward, next_state, update_reward_trail = False):
        """
        Train the DQN agent on the <state, action, reward, next_state, is_done>
        tuple at the current timestep.
        """
        if update_reward_trail == True:
            self.update_reward_history_tree(state,action_index, reward)
        self.train()
        torch.set_grad_enabled(True)
        state_tensor = torch.tensor(np.array(state)[np.newaxis, :], dtype=torch.float32, requires_grad=True).to(DEVICE)
        target = self.get_target(reward, next_state)
        output = self.forward(state_tensor)
        target_f = output.clone()
        target_f[0][action_index] = target
        target_f.detach()
        self.optimizer.zero_grad()
        loss = F.mse_loss(output, target_f)
        loss.backward()
        self.optimizer.step()