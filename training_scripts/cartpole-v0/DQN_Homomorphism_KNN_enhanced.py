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

from collections import OrderedDict

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

class ActionEncoder(nn.Module):
    def __init__(self, n_dim, n_actions, hidden_dim1=100, hidden_dim2=100, temp=1.):
        super().__init__()
        self.linear1 = nn.Linear(n_dim+n_actions, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, n_dim)

    def forward(self, z, a):
        za = torch.cat([z, a], dim=1)
        za = F.relu(self.linear1(za))
        za = F.relu(self.linear2(za))
        zt = self.linear3(za)
        return torch.sigmoid(zt)

class StateEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, mid=200, mid2=200, mid3=200,
                 activation=F.relu):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, mid)
        self.fc2 = nn.Linear(mid, mid2)
        self.fc3 = nn.Linear(mid2, mid3)
        self.fc4 = nn.Linear(mid3, out_dim)
        self.act = activation

    def forward(self, obs):
        h = self.act(self.fc1(obs))
        h = self.act(self.fc2(h))
        h = self.act(self.fc3(h))
        z = self.fc4(h)
        return torch.sigmoid(z)

class Model(nn.Module):
    def __init__(self, state_encoder, action_encoder):
        super().__init__()
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder

    def forward(self, x):
        raise NotImplementedError("This model is a placeholder")

    def train(self, boolean):
        self.state_encoder.train = boolean
        self.action_encoder.train = boolean

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

    def forward(self, z_c, z_l, z_n, z_f, action_embeddings, reward_fixation):
        # Transition loss
        transition_loss = self.distance(z_n, z_l).mean()

        next_states_sum = None
        for embedding in action_embeddings:
            if next_states_sum == None:
                next_states_sum = embedding
            else:
                next_states_sum = next_states_sum + embedding
        symmetry_loss = 0.5 * square_dist(z_c, z_c + next_states_sum / len(action_embeddings))

        reward_fixation_loss = 0.5 * square_dist(z_c, reward_fixation)

        # Negative loss
        negative_loss = None    
        for abstract_negative_state in z_f:
            if negative_loss == None:
                negative_loss = self.distance.negative_distance(z_l, abstract_negative_state)
            else:
                negative_loss = negative_loss + self.distance.negative_distance(z_l, abstract_negative_state)
        return transition_loss, negative_loss, symmetry_loss, reward_fixation_loss

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
        self.abstraction_model = Model(state_encoder, action_encoder)
        self.abstract_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.abstraction_model.parameters()), lr=self.params['abstraction_learning_rate'])
        self.abstraction_memory = collections.deque(maxlen=params['memory_size_for_abstraction'])
        self.abstraction_batch_size = params['batch_size_for_abstraction']
        self.negative_samples_size = params['negative_samples_size']
        self.loss_function = Loss(params['hinge'])
        self.abstract_state_holders = OrderedDict()
        self.symmetry_weight = params['symmetry_weight']
        self.reward_fixation_in_abstraction = {}
        self.rewards_temp = {}

    def on_new_sample(self, state, action, reward, next_state, is_done):
        if reward not in self.reward_fixation_in_abstraction:
            with torch.no_grad():
                #self.reward_fixation_in_abstraction[reward] = torch.tensor([0.5,0.5])
                self.reward_fixation_in_abstraction[reward] = torch.rand(self.params['abstract_state_space_dimmension'])
        self.memory.append((state, action, reward, next_state, is_done))
        self.abstraction_memory.append((state, action, reward, next_state, is_done))
        with torch.no_grad():
            next_state_tensor = torch.tensor(np.array(next_state)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
            abstract_next_state = self.abstraction_model.state_encoder(next_state_tensor)
            self.abstract_state_holders[(tuple(state), action)] = (abstract_next_state, next_state)
            if len(self.abstract_state_holders) > self.params['abstract_state_holders_size']:
                self.abstract_state_holders.popitem(last=False)
            self.rewards_temp[(tuple(state), action)] = reward

    def on_finished(self):
        pass

    def on_episode_start(self, episode_index):
        self.update_all_in_abstract_state_holders()
        if self.params['plot_t-sne'] == True and episode_index != 0 and (episode_index+1) % 10 == 0:
            self.draw_tsne(episode_index)

    def update_all_in_abstract_state_holders(self):
        for key, value in self.abstract_state_holders.items():
            state, action = key
            abstract_next_state, next_state = value
            with torch.no_grad():
                next_state_tensor = torch.tensor(np.array(next_state)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
                updated_abstract_next_state = self.abstraction_model.state_encoder(next_state_tensor)
                self.abstract_state_holders[key] = (updated_abstract_next_state ,next_state)

    def draw_tsne(self, episode_index):
        matplotlib.use("Qt5Agg")
        X = np.array([tensor[0].numpy().flatten() for tensor in self.abstract_state_holders.values()])

        env = gym.make("CartPole-v1", render_mode="rgb_array")
        env.reset()
        # Perform t-SNE
        tsne = TSNE(n_components=2)
        tsne_states = tsne.fit_transform(X)
        if self.params['abstract_state_space_dimmension'] == 2:
            tsne_states = X

        tx_orig, ty_orig = tsne_states[:,0], tsne_states[:,1]
        tx = (tx_orig-np.min(tx_orig)) / (np.max(tx_orig) - np.min(tx_orig))
        ty = (ty_orig-np.min(ty_orig)) / (np.max(ty_orig) - np.min(ty_orig))

        labels = list(self.abstract_state_holders.keys())
        values = list(self.abstract_state_holders.values())
        width = 16000
        height = 12000
        max_dim = 100

        full_image = Image.new('RGBA', (width, height))
        if self.params['plot_reward_fixations'] == True:
            rewards = np.array([tensor.numpy().flatten() for tensor in self.reward_fixation_in_abstraction.values()])
            labels_rewards = list(self.reward_fixation_in_abstraction.keys())
            if self.params['abstract_state_space_dimmension'] == 2:
                tsne_rewards = rewards
            else:
                tsne_rewards = tsne.fit_transform(rewards)
            tx_rewards_orig, ty_rewards_orig = tsne_rewards[:,0], tsne_rewards[:,1]
            tx_rewards = (tx_rewards_orig-np.min(tx_rewards_orig)) / (np.max(tx_rewards_orig) - np.min(tx_rewards_orig))
            ty_rewards = (ty_rewards_orig-np.min(ty_rewards_orig)) / (np.max(ty_rewards_orig) - np.min(ty_rewards_orig))
                
            tx = (tx_orig-np.min(tx_rewards_orig)) / (np.max(tx_rewards_orig) - np.min(tx_rewards_orig))
            ty = (ty_orig-np.min(ty_rewards_orig)) / (np.max(ty_rewards_orig) - np.min(ty_rewards_orig))
            for i, coord in enumerate(tsne_rewards):
                x = tx_rewards[i]
                y = ty_rewards[i]
                text = str(labels_rewards[i])

                tile = Image.new("RGB", (1000, 1000), (0,0,0))
                draw = ImageDraw.Draw(tile)
                font = ImageFont.truetype("Arial.ttf",200)  # load font
                position = (10, 10)
                text_color = (255, 255, 255)  # Use RGB values for the desired color
                draw.text(position, text, font=font, fill=text_color)
                rs = max(1, tile.width/max_dim, tile.height/max_dim)
                tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
                full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

        for i, coord in enumerate(tsne_states):
            x = tx[i]
            y = ty[i]
            abstract_state, next_state = values[i]
            state, action = labels[i]

            font_size = 80
            if self.params['t-sne_next_state'] == False:
                env.state = env.unwrapped.state = state
                text = 'v, av:'+str(round(state[1],2))+','+str(round(state[3],2)) + '\n' + 'a:' + ('<-' if action == 0 else '->')
            else:
                env.state = env.unwrapped.state = next_state
                text = 'v, av:'+str(round(next_state[1],2))+','+str(round(next_state[3],2)) + '\n' + f'r:{self.rewards_temp[(tuple(state), action)]}'
                
            #temp
            text = text + f'r:{round(self.rewards_temp[(tuple(state), action)],2)}'
            font_size = 80
                
            img = env.render()

            tile = Image.fromarray(img)
            draw = ImageDraw.Draw(tile)
            font = ImageFont.truetype("Arial.ttf", font_size)  # load font
            position = (10, 10)
            text_color = (0, 0, 0)  # Use RGB values for the desired color
            draw.text(position, text, font=font, fill=text_color)
            rs = max(1, tile.width/max_dim, tile.height/max_dim)
            tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
            full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

        plt.figure(figsize = (16,12))
        plt.title(f'{episode_index + 1}\'th episode, 1')
        plt.imshow(full_image)
        plt.show(block=True)
        return

    def create_knn_model(self, is_abstraction):
        if self.params['exploit_symmetry'] == False:
            return None
        kNNModel = FaissKNeighbors(self.params['K_for_KNN'])
        X = np.array([tensor[0].numpy().flatten() for tensor in self.abstract_state_holders.values()])
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
        for state, action, reward, next_state, is_done in minibatch:
            self.abstract_optimizer.zero_grad()

            # Abstract state
            state_tensor = torch.tensor(np.array(state)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
            z_c = self.abstraction_model.state_encoder(state_tensor)

            # Abstract action
            action_embeddings = []
            for possible_action in range(self.params['number_of_actions']):
                action_embeddings.append(self.abstraction_model.action_encoder(z_c, self.one_hot(possible_action)))
            # transition in latent space
            z_l = z_c + action_embeddings[action]

            if len(self.abstraction_memory) > self.negative_samples_size:
                negative_batch = random.sample(self.abstraction_memory, self.negative_samples_size)
            else:
                negative_batch = self.abstraction_memory
            z_f = []

            for negative_state, negative_action, negative_reward, negative_next_state, is_done in negative_batch:
                negative_state_tensor = torch.tensor(np.array(negative_state)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
                z_f.append(self.abstraction_model.state_encoder(negative_state_tensor))

            next_state_tensor = torch.tensor(np.array(next_state)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
            z_n = self.abstraction_model.state_encoder(next_state_tensor)
            self.abstract_state_holders[(tuple(state), action)] = (z_n.detach(), next_state)

            # Loss components
            trans_loss, neg_loss, symmetry_loss, reward_fixation_loss = self.loss_function(z_c, z_l, z_n,
                                                                z_f, action_embeddings, self.reward_fixation_in_abstraction[reward])
            hinge_loss = (self.params['hinge'] - min(torch.norm(action_embeddings[action]), self.params['hinge']))
            loss = reward_fixation_loss #+ trans_loss #+ symmetry_loss # + hinge_loss
            if neg_loss != None:
                loss = loss + neg_loss
            loss.backward()
            self.abstract_optimizer.step()

    def replay_mem(self, batch_size, is_decay_epsilon):
        self.replay_abstract_model()
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