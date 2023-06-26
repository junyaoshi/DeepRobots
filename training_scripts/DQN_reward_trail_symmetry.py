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

class RewardTreeNode:
    def __init__(self):
        self.children = {}
        self.occurrences = {}

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

        self.reward_trail_length = params['reward_trail_length']
        self.reward_trail_reward_decimals = params['reward_trail_reward_decimals']
        self.reward_trail_state_decimals = params['reward_trail_state_decimals']
        self.reward_trail_symmetry_threshold = params['reward_trail_symmetry_threshold']
        self.reward_trail_symmetry_weight = params['reward_trail_symmetry_weight']
        self.reward_trail = []
        self.reward_tree = RewardTreeNode()

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
            state_tensor = torch.tensor(np.array(state)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
            target = self.get_target(reward, next_state)
            output = self.model.forward(state_tensor)
            self.optimizer.zero_grad()

            symmetry_loss_sum = None
            symmetries = self.find_symmetries(state, action_index)
            for symmetry in symmetries:
                symmetry_state, symmetry_action_index = symmetry
                symmetry_state_tensor = torch.tensor(np.array(symmetry_state)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
                symmetry_output = self.model.forward(symmetry_state_tensor)
                symmetry_loss = (target - symmetry_output[0][symmetry_action_index]) ** 2
                if symmetry_loss_sum is None:
                    symmetry_loss_sum = symmetry_loss
                else:
                    symmetry_loss_sum += symmetry_loss
            loss = (output[0][action_index] - target) ** 2
            if symmetry_loss_sum != None:
                loss += self.reward_trail_symmetry_weight * symmetry_loss_sum
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

    def reset_reward_trail(self):
        self.reward_trail = []

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
            occurrence_key = updating_state + (updating_action_index, )
            if occurrence_key in node.occurrences:
                node.occurrences[occurrence_key] += 1
            else:
                node.occurrences[occurrence_key] = 1
                
    def find_symmetries(self, state, action_index):
        state = tuple(round(x,self.reward_trail_state_decimals) for x in state)
        occurences_counts = {}
        occurences_counts_intersection = {}
        nodes = [self.reward_tree]
        target_occurence_key = state + (action_index, )
        for i in range(self.reward_trail_length + 1): # + 1 because of the root
            next_nodes = []
            for node in nodes:
                target_occurrence_count = 0
                if target_occurence_key in node.occurrences:
                    target_occurrence_count = node.occurrences[target_occurence_key]

                for state_action_key, occurence_count in node.occurrences.items():
                    if state_action_key in occurences_counts:
                        occurences_counts[state_action_key] += occurence_count
                    else:
                        occurences_counts[state_action_key] = occurence_count

                    if target_occurrence_count != 0 and state_action_key != target_occurence_key:
                        intersection_count = min(occurence_count, target_occurrence_count)
                        if state_action_key in occurences_counts_intersection:
                            occurences_counts_intersection[state_action_key] += intersection_count
                        else:
                            occurences_counts_intersection[state_action_key] = intersection_count
                for reward, node in node.children.items():
                    next_nodes.append(node)
            nodes = next_nodes
        
        symmetries = []
        for state_action_key, intersection_count in occurences_counts_intersection.items():
            similarity = intersection_count / math.sqrt(occurences_counts[state_action_key] * occurences_counts[target_occurence_key])
            if similarity < self.reward_trail_symmetry_threshold:
                continue
            symmetries.append((state_action_key[:3], state_action_key[3]))
        return symmetries