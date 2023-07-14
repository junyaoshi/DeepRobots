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

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import gymnasium as gym
from PIL import Image, ImageDraw, ImageFont
import matplotlib
# pip install PyQt5

import numpy as np
#pip install faiss
import faiss

class FaissKNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k + 1)
        predictions = self.y[indices[0][1:]]
        return predictions

class QNetwork(nn.Module):
    def __init__(self, params):
        super(QNetwork, self).__init__()
        self.f1 = nn.Linear(4, params['first_layer_size'])
        self.f2 = nn.Linear(params['first_layer_size'], params['second_layer_size'])
        self.f3 = nn.Linear(params['second_layer_size'], params['number_of_actions'])

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

class RewardTrailEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, mid=64, mid2=64, activation=F.relu):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, mid)
        self.fc2 = nn.Linear(mid, mid2)
        self.fc3 = nn.Linear(mid2, out_dim)
        self.act = activation

    def forward(self, s):
        h = self.act(self.fc1(s))
        h = self.act(self.fc2(h))
        r = self.fc3(h)
        return r

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

def square_dist(x, y, dim=1):
    return (x-y).pow(2).sum(dim=dim)

def tile(embedding, example):
    n = example.shape[0]//embedding.shape[0]
    embedding = embedding.unsqueeze(1).repeat(1, n, 1)
    embedding = squeeze_embedding(embedding)
    return embedding

def squeeze_embedding(x):
    b, n, d = x.shape
    x = x.reshape(b*n, d)
    return x

class HingedSquaredEuclidean(nn.Module):
    def __init__(self, eps=1.0):
        super().__init__()
        self.eps = eps

    def forward(self, x, y, dim=1):
        return 0.5 * square_dist(x, y, dim)

    def negative_distance(self, x, y, dim=1):
        dist = self.forward(x, y, dim)
        neg_dist = torch.max(torch.zeros_like(dist), self.eps-dist)
        return neg_dist

class Loss(nn.Module):
    def __init__(self, hinge):
        super().__init__()
        self.distance = HingedSquaredEuclidean(eps=hinge)

    def forward(self, z_c, z_l, z_n, z_f, r, r_e):
        # Transition loss
        transition_loss = self.distance(z_n, z_l).mean()
        # Reward loss
        reward_loss = 0.5 * square_dist(r, r_e).mean()
        # Negative loss
        negative_loss = None    
        for abstract_negative_state in z_f:
            if negative_loss == None:
                negative_loss = self.distance.negative_distance(z_l, abstract_negative_state)
            else:
                negative_loss += self.distance.negative_distance(z_l, abstract_negative_state)
        negative_loss = negative_loss
        return transition_loss, reward_loss, negative_loss

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
        self.abstraction_memory = collections.deque(maxlen=params['memory_size_for_abstraction'])
        self.abstraction_batch_size = params['batch_size_for_abstraction']
        self.negative_samples_size = params['negative_samples_size']
        self.loss_function = Loss(params['hinge'])
        self.abstract_state_holders = {}
        self.symmetry_weight = params['symmetry_weight']
        
        self.reward_trail_length = params['reward_trail_length']
        self.reward_trail_encoder = RewardTrailEncoder(4, self.reward_trail_length)
        self.reward_trail = []
        self.reward_trail_memory = collections.deque(maxlen=params['reward_trail_memory_size'])
        self.reward_trail_batch_size = params['reward_trail_batch_size']

    def on_new_sample(self, state, action, reward, next_state, is_done):
        self.memory.append((state, action, reward, next_state, is_done))
        self.abstraction_memory.append((state, action, reward, next_state, is_done))
        self.update_reward_trail(tuple(state), reward)
        with torch.no_grad():
            next_state_tensor = torch.tensor(np.array(next_state)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
            abstract_next_state = self.abstraction_model.state_encoder(next_state_tensor)
            state_tensor = torch.tensor(np.array(state)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
            reward_trail = self.reward_trail_encoder(state_tensor)
            self.abstract_state_holders[(tuple(state), action)] = (abstract_next_state, reward_trail, next_state)

    def update_reward_trail(self, new_state, new_reward):
        self.reward_trail.append((new_state, new_reward))
        if len(self.reward_trail) <= self.reward_trail_length:
            return
        updating_state, updating_reward = self.reward_trail.pop(0)
        rewards = []
        for item in self.reward_trail:
            rewards.append(item[1])
        self.reward_trail_memory.append((updating_state, rewards))

    def on_finished(self):
        pass

    def on_episode_start(self, episode_index):
        self.reward_trail = []
        self.update_all_in_abstract_state_holders()
        if self.params['plot_t-sne'] == True and episode_index != 0 and (episode_index+1) % 10 == 0:
            self.draw_tsne(episode_index)

    def update_all_in_abstract_state_holders(self):
        for key, value in self.abstract_state_holders.items():
            state, action = key
            abstract_next_state, reward_trail, next_state = value
            with torch.no_grad():
                next_state_tensor = torch.tensor(np.array(next_state)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
                updated_abstract_next_state = self.abstraction_model.state_encoder(next_state_tensor)
                state_tensor = torch.tensor(np.array(state)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
                updated_reward_trail = self.reward_trail_encoder(state_tensor)
                self.abstract_state_holders[key] = (updated_abstract_next_state, updated_reward_trail ,next_state)

    def draw_tsne(self, episode_index):
        matplotlib.use("Qt5Agg")
        # Convert tensor coordinates to numpy arrays
        plot_index = 0
        if self.params['t-sne_plot_abstraction'] == False:
            plot_index = 1 # reward trail plotting
        X = np.array([tensor[plot_index].numpy().flatten() for tensor in self.abstract_state_holders.values()])

        env = gym.make("CartPole-v0", render_mode="rgb_array")
        env.reset()
        # Perform t-SNE
        tsne = TSNE(n_components=2)
        tsne_states = tsne.fit_transform(X)

        tx, ty = tsne_states[:,0], tsne_states[:,1]
        tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

        labels = list(self.abstract_state_holders.keys())
        values = list(self.abstract_state_holders.values())
        width = 16000
        height = 12000
        max_dim = 100

        full_image = Image.new('RGBA', (width, height))
        for i, coord in enumerate(tsne_states):
            x = tx[i]
            y = ty[i]
            abstract_state, reward_trail, next_state = values[i]
            state, action = labels[i]

            if self.params['t-sne_next_state'] == False:
                env.state = env.unwrapped.state = state
                text = 'v, av:'+str(round(state[1],2))+','+str(round(state[3],2)) + '\n' + 'a:' + ('<-' if action == 0 else '->')
            else:
                env.state = env.unwrapped.state = next_state
                text = 'v, av:'+str(round(next_state[1],2))+','+str(round(next_state[3],2))
            img = env.render()

            tile = Image.fromarray(img)
            draw = ImageDraw.Draw(tile)
            font = ImageFont.truetype("Arial.ttf",80)  # load font
            position = (10, 10)
            text_color = (0, 0, 0)  # Use RGB values for the desired color
            draw.text(position, text, font=font, fill=text_color)
            rs = max(1, tile.width/max_dim, tile.height/max_dim)
            tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
            full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))
        plt.figure(figsize = (16,12))
        plt.title(f'{episode_index + 1}\'th episode, new abstraction')
        plt.imshow(full_image)
        plt.show(block=True)
        return

    def create_knn_model(self, is_abstraction):
        if self.params['exploit_symmetry'] == False:
            return None
        index = 0
        if is_abstraction == False:
            index = 1 #reward trail
        kNNModel = FaissKNeighbors(self.params['K_for_KNN'])
        X = np.array([tensor[index].numpy().flatten() for tensor in self.abstract_state_holders.values()])
        kNNModel.fit(X, np.array(list(self.abstract_state_holders.keys()), dtype=object))
        return kNNModel

    def find_symmetries(self, state, action, knn_model):
        if self.params['exploit_symmetry'] == False:
            return []
        target_key = (tuple(state), action)
        target_tensor = self.abstract_state_holders[target_key][0]
        return knn_model.predict(target_tensor[0].numpy().flatten()[np.newaxis, :])

    def one_hot(self, action):
        zeros = np.zeros((1, self.params['number_of_actions']))
        zeros[np.arange(zeros.shape[0]), action] = 1
        return torch.FloatTensor(zeros)

    def replay_abstract_model(self):
        if len(self.abstraction_memory) > self.abstraction_batch_size:
            minibatch = random.sample(self.abstraction_memory, self.abstraction_batch_size)
        else:
            minibatch = self.abstraction_memory
        self.abstraction_model.train(True)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.abstraction_model.parameters()), lr=self.params['abstraction_learning_rate'])
        for state, action, reward, next_state, is_done in minibatch:
            optimizer.zero_grad()
            action_onehot = self.one_hot(action)

            # Abstract state
            state_tensor = torch.tensor(np.array(state)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
            z_c = self.abstraction_model.state_encoder(state_tensor)

            # Abstract action
            action_embedding = self.abstraction_model.action_encoder(z_c, action_onehot)
            # transition in latent space
            z_l = z_c + action_embedding

            if len(self.abstraction_memory) > self.negative_samples_size:
                negative_batch = random.sample(self.abstraction_memory, self.negative_samples_size)
            else:
                negative_batch = self.abstraction_memory
            z_f = []

            for state, action, reward, next_state, is_done in negative_batch:
                negative_state_tensor = torch.tensor(np.array(state)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
                z_f.append(self.abstraction_model.state_encoder(negative_state_tensor))

            next_state_tensor = torch.tensor(np.array(next_state)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
            z_n = self.abstraction_model.state_encoder(next_state_tensor)
            old_value = self.abstract_state_holders[(tuple(state), action)]
            self.abstract_state_holders[(tuple(state), action)] = (z_n.detach(), old_value[1], old_value[2])

            # Predicted reward
            r_e = self.abstraction_model.reward(z_l)

            # Loss components
            trans_loss, reward_loss, neg_loss = self.loss_function(z_c, z_l, z_n,
                                                                z_f, reward,
                                                                r_e)
            loss = trans_loss + reward_loss
            if neg_loss != None:
                loss = loss + neg_loss
            loss.backward()
            optimizer.step()

    def replay_reward_trail(self):
        if len(self.reward_trail_memory) > self.reward_trail_batch_size:
            minibatch = random.sample(self.reward_trail_memory, self.reward_trail_batch_size)
        else:
            minibatch = self.reward_trail_memory

        self.reward_trail_encoder.train()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.reward_trail_encoder.parameters()), lr=self.params['reward_trail_learning_rate'])
        for state, rewards in minibatch:
            optimizer.zero_grad()
            state_tensor = torch.tensor(np.array(state)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
            output = self.reward_trail_encoder(state_tensor)
            target = torch.tensor(np.array(rewards)[np.newaxis, :], dtype=torch.float32).detach().to(DEVICE)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()

    def replay_mem(self, batch_size, is_decay_epsilon):
        self.replay_reward_trail()
        #self.replay_abstract_model()
        if len(self.memory) > batch_size:
            minibatch = random.sample(self.memory, batch_size)
        else:
            minibatch = self.memory
        knn_model = self.create_knn_model(True)
        for state, action, reward, next_state, is_done in minibatch:
            state = tuple(state)
            self.model.train()
            torch.set_grad_enabled(True)
            state_tensor = torch.tensor(np.array(state)[np.newaxis, :], dtype=torch.float32, requires_grad=True).to(DEVICE)
            target = self.get_target(reward, next_state, is_done)
            output = self.model.forward(state_tensor)
            self.optimizer.zero_grad()

            symmetry_loss_sum = None
            symmetries = self.find_symmetries(state, action, knn_model)
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
                loss += self.symmetry_weight * symmetry_loss_sum

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

    def on_terminated(self):
        pass