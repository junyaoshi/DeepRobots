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
matplotlib.use("Qt5Agg")

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
        self.distance = square_dist
        self.eps = eps

    def forward(self, x, y, dim=1):
        return 0.5 * self.distance(x, y, dim)

    def negative_distance(self, x, y, dim=1):
        dist = self.forward(x, y, dim)
        neg_dist = torch.max(torch.zeros_like(dist), self.eps-dist)
        return neg_dist

class Loss(nn.Module):
    def __init__(self, hinge):
        super().__init__()
        self.reward_loss = square_dist
        self.distance = HingedSquaredEuclidean(eps=hinge)

    def forward(self, z_c, z_l, z_n, z_f, r, r_e):
        # Transition loss
        transition_loss = self.distance(z_n, z_l).mean()
        # Reward loss
        reward_loss = 0.5 * self.reward_loss(r, r_e).mean()
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

    def on_finished(self):
        pass

    def on_episode_start(self, episode_index):
        if episode_index != 0 and (episode_index+1) % 10 == 0:
            self.draw_tsne(episode_index)

    def draw_tsne(self, episode_index):
        # Convert tensor coordinates to numpy arrays
        X = np.array([tensor.numpy().flatten() for tensor in self.abstract_state_holders.values()])

        env = gym.make("CartPole-v1", render_mode="rgb_array")
        env.reset()
        # Perform t-SNE
        tsne = TSNE(n_components=2)
        tsne_states = tsne.fit_transform(X)

        tx, ty = tsne_states[:,0], tsne_states[:,1]
        tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

        labels = list(self.abstract_state_holders.keys())
        width = 16000
        height = 12000
        max_dim = 100

        full_image = Image.new('RGBA', (width, height))
        for i, coord in enumerate(tsne_states):
            x = tx[i]
            y = ty[i]
            state, action = labels[i]
            env.state = env.unwrapped.state = state
            img = env.render()  # Get the environment image
            tile = Image.fromarray(img)
            # Create a drawing object
            draw = ImageDraw.Draw(tile)
            # Specify the text content and font
            action_text = 'left' if action == 0 else 'right'
            text = f'v, av:{round(state[1],2)},{round(state[3],2)}\nmoving {action_text}'
            font = ImageFont.truetype("Arial.ttf",80)  # load font
            position = (10, 10)
            text_color = (0, 0, 0)  # Use RGB values for the desired color
            draw.text(position, text, font=font, fill=text_color)
            rs = max(1, tile.width/max_dim, tile.height/max_dim)
            tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
            full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))
        plt.figure(figsize = (16,12))
        plt.title(f'{episode_index + 1}\'th episode')
        plt.imshow(full_image)
        plt.show(block=True)
        return

    def on_new_sample(self, state, action, reward, next_state, is_done):
        self.memory.append((state, action, reward, next_state, is_done))
        self.abstraction_memory.append((state, action, reward, next_state, is_done))
        with torch.no_grad():
            next_state_tensor = torch.tensor(np.array(next_state)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
            self.abstract_state_holders[(tuple(state), action)] = self.abstraction_model.state_encoder(next_state_tensor)

    def find_symmetries(self, state, action):
        target_key = (tuple(state), action)
        target_tensor = self.abstract_state_holders[target_key]
        distances = {k: torch.norm(target_tensor - v) for k, v in self.abstract_state_holders.items() if k != target_key}
        sorted_distances = sorted(distances.items(), key=lambda x: x[1])
        return [item[0] for item in sorted_distances[:self.params['K_for_KNN']]]

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
            self.abstract_state_holders[(tuple(state), action)] = z_n.detach()

            # Predicted reward
            r_e = self.abstraction_model.reward(z_l)

            # Loss components
            trans_loss, reward_loss, neg_loss = self.loss_function(z_c, z_l, z_n,
                                                                z_f, reward,
                                                                r_e)
            loss = trans_loss + reward_loss + neg_loss
            loss.backward()
            optimizer.step()

    def replay_mem(self, batch_size, is_decay_epsilon):
        self.replay_abstract_model()
        if len(self.memory) > batch_size:
            minibatch = random.sample(self.memory, batch_size)
        else:
            minibatch = self.memory
        for state, action, reward, next_state, is_done in minibatch:
            state = tuple(state)
            self.model.train()
            torch.set_grad_enabled(True)
            state_tensor = torch.tensor(np.array(state)[np.newaxis, :], dtype=torch.float32, requires_grad=True).to(DEVICE)
            target = self.get_target(reward, next_state, is_done)
            output = self.model.forward(state_tensor)
            self.optimizer.zero_grad()

            symmetry_loss_sum = None
            symmetries = self.find_symmetries(state, action)
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