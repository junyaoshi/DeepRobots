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

        self.reward_trail_length = params['reward_trail_length']
        self.reward_trail_symmetry_threshold = params['reward_trail_symmetry_threshold']
        self.reward_trail_symmetry_weight = params['reward_trail_symmetry_weight']
        self.reward_trail = []
        self.reward_tree = RewardTreeNode()

    def remember(self, state_with_action, reward, next_state, is_done):
        """
        Store the <state, action, reward, next_state> tuple in a 
        memory buffer for replay memory.
        """
        self.memory.append((state_with_action, reward, next_state, is_done))

    def replay_mem(self, batch_size, goal, episodes, length):
        """
        Replay memory.
        """
        if len(self.memory) > batch_size:
            minibatch = random.sample(self.memory, batch_size)
        else:
            minibatch = self.memory
        symmetries_batch = self.find_symmetries(minibatch)
        for state_with_action, reward, next_state, is_done in minibatch:
            self.model.train()
            torch.set_grad_enabled(True)
            state_tensor = torch.tensor(np.array(state_with_action)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
            target = self.get_target(reward, next_state, is_done)
            output = self.model(state_tensor[0])
            self.optimizer.zero_grad()

            symmetry_loss_sum = None
            if state_with_action in symmetries_batch:
                symmetries = symmetries_batch[state_with_action]

                def print_position(string, state_with_action):
                    index = state_with_action.index(1)
                    position = ((int)(index % self.world_size), (int)(index / self.world_size))
                    dir = ''
                    if state_with_action[self.world_size**2] == 1: #north
                        dir = 'North'
                    elif state_with_action[self.world_size**2 + 1] == 1: #east
                        dir = 'Right'
                    elif state_with_action[self.world_size**2 + 2] == 1: #west
                        dir = 'Left'
                    else: # south
                        dir = 'Down'
                    print(string, position, dir)
                #print_position(f'goal: {goal}, {episodes}-{length} current:', state_with_action)

                for symmetry_state_with_action in symmetries:
                    #print_position('symmetries:', symmetry_state_with_action)

                    symmetry_state_tensor = torch.tensor(np.array(symmetry_state_with_action)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
                    symmetry_output = self.model(symmetry_state_tensor[0])
                    symmetry_loss = (target - symmetry_output[0]) ** 2
                    if symmetry_loss_sum is None:
                        symmetry_loss_sum = symmetry_loss
                    else:
                        symmetry_loss_sum += symmetry_loss
            loss = (output[0] - target) ** 2
            if symmetry_loss_sum != None:
                loss += self.reward_trail_symmetry_weight * symmetry_loss_sum
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

    def reset_reward_trail(self):
        self.reward_trail = []

    def update_reward_history_tree(self, state_with_action, reward):
        self.reward_trail.append((state_with_action, reward))
        if len(self.reward_trail) <= self.reward_trail_length:
            return
        updating_state_with_action, updating_reward = self.reward_trail.pop(0)
        node = self.reward_tree
        for item in self.reward_trail:
            trail_reward = item[1]
            if trail_reward not in node.children:
                node.children[trail_reward] = RewardTreeNode()
            node = node.children[trail_reward]
            if updating_state_with_action in node.occurrences:
                node.occurrences[updating_state_with_action] += 1
            else:
                node.occurrences[updating_state_with_action] = 1

    def find_symmetries(self, batch):
        occurences_counts = {}
        occurences_counts_intersections = {}
        nodes = [self.reward_tree]
        for i in range(self.reward_trail_length + 1): # + 1 because of the root
            next_nodes = []
            for node in nodes:
                target_occurrence_count = {}
                for state_with_action, reward, next_state, is_done in batch:
                    if state_with_action in node.occurrences:
                        target_occurrence_count[state_with_action] = node.occurrences[state_with_action]

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
                symmetries[target_occurence_key].append(state_action_key)
        return symmetries