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
        self.f1 = nn.Linear(4, self.first_layer)
        self.f2 = nn.Linear(self.first_layer, self.second_layer)
        self.f3 = nn.Linear(self.second_layer, 2)

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = self.f3(x)
        return x

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

        self.reward_trail_length = params['reward_trail_length']
        self.reward_trail_symmetry_threshold = params['reward_trail_symmetry_threshold']
        self.reward_trail_symmetry_weight = params['reward_trail_symmetry_weight']
        self.reward_trail = []
        self.reward_tree = RewardTreeNode()


    def remember(self, state, action, reward, next_state, is_done):
        """
        Store the <state, action, reward, next_state> tuple in a 
        memory buffer for replay memory.
        """
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

    def reset_reward_trail(self):
        self.reward_trail = []

    def update_reward_history_tree(self, state, action_index, reward):
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

    def find_symmetries(self, batch):
        occurences_counts = {}
        occurences_counts_intersections = {}
        nodes = [self.reward_tree]
        for i in range(self.reward_trail_length + 1): # + 1 because of the root
            next_nodes = []
            for node in nodes:
                target_occurrence_count = {}
                for state, action, reward, next_state, is_done in batch:
                    state = tuple(state)
                    target_occurence_key = state + (action, )
                    if target_occurence_key in node.occurrences:
                        target_occurrence_count[target_occurence_key] = node.occurrences[target_occurence_key]

                for state_action_key, occurence_count in node.occurrences.items():
                    if state_action_key in occurences_counts:
                        occurences_counts[state_action_key] += occurence_count
                    else:
                        occurences_counts[state_action_key] = occurence_count

                    for target_occurence_key, target_occurrence in target_occurrence_count.items():
                        if state_action_key == target_occurence_key:
                            continue
                        intersection_count = min(occurence_count, target_occurrence)
                        if target_occurence_key not in occurences_counts_intersections:
                            occurences_counts_intersections[target_occurence_key] = {}

                        if state_action_key in occurences_counts_intersections[target_occurence_key]:
                            occurences_counts_intersections[target_occurence_key][state_action_key] += intersection_count
                        else:
                            occurences_counts_intersections[target_occurence_key][state_action_key] = intersection_count
                for reward, node in node.children.items():
                    next_nodes.append(node)
            nodes = next_nodes

        symmetries = {}
        for target_occurence_key, occurences_counts_intersection in occurences_counts_intersections.items():
            for state_action_key, intersection_count in occurences_counts_intersection.items():
                similarity = intersection_count / math.sqrt(occurences_counts[state_action_key] * occurences_counts[target_occurence_key])
                if similarity < self.reward_trail_symmetry_threshold:
                    continue
                if target_occurence_key not in symmetries:
                    symmetries[target_occurence_key] = []
                symmetries[target_occurence_key].append((state_action_key[:4], state_action_key[4]))
        return symmetries