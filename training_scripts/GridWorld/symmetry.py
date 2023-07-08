import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import csv
import torch
import random

from DQN_symmetry import DQNAgent, DEVICE

def define_parameters():
	params = dict()
	# Neural Network
	params['learning_rate'] = 0.001
	params['weight_decay'] = 0
	params['first_layer_size'] = 300    # neurons in the first layer
	params['second_layer_size'] = 120   # neurons in the second layer
	params['episode_length'] = 800
	params['memory_size'] = 1
	params['batch_size'] = 1
	params['gamma'] = 0.9
	params['epsilon'] = 0.1
	params['epsilon_decay'] = 0.999
	params['epsilon_minimum'] = 0.1
	params['episodes'] = 200
	params['run_times_for_performance_average'] = 10
	params['world_size'] = 13

	#reward trailing
	params['reward_trail_length'] = 5
	params['reward_trail_symmetry_threshold'] = 1.0
	params['reward_trail_symmetry_weight'] = 1.0
	return params

def plot(title, ylabel, xlabel, values, times, save_to_csv_file_name = ""):
	if save_to_csv_file_name != "":
		data = zip(values, times)
		with open(save_to_csv_file_name, 'w', newline='') as file:
			writer = csv.writer(file)
			writer.writerow([ylabel, xlabel])
			writer.writerows(data)
	plt.plot(times, values)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.show()

def reward_potential(position, goal_position, gamma):
	multiplier = (abs(position[0] - goal_position[0]) + abs(position[1] - goal_position[1]))
	return -multiplier * (gamma ** multiplier)

def plot_csv(file_name):
	with open(file_name, 'r') as file:
		reader = csv.reader(file)
		header = next(reader)  # Skip the header row
		ylabel = header[0]
		xlabel = header[1]
		values = []
		times = []
		for row in reader:
			values.append(float(row[0]))
			times.append(float(row[1]))
		plot(ylabel,xlabel,values,times)

def position_to_state(position, world_size):
	state = [0] * (world_size ** 2)
	state[position[1] * world_size + position[0]] = 1
	return tuple(state)

def train_agent_and_sample_performance(agent, params, run_iteration):
	average_rewards = []
	episodes = []
	world_size = params['world_size']
	gamma = params['gamma']
	goal_position = (random.randint(0, world_size-1), random.randint(0,world_size-1))
	position = (random.randint(0, world_size-1), random.randint(0,world_size-1))
	for i in range(params['episodes']):
		if i % 100 == 0:
			print(f'{run_iteration}th running, epidoes: {i}')
		total_reward = 0
		while position == goal_position:
			position = (random.randint(0, world_size-1), random.randint(0,world_size-1))
		for j in range(params['episode_length']):
			if position == goal_position:
				break
			old_position = position
			curr_state_with_action = agent.include_action(position_to_state(position, world_size))
			if curr_state_with_action[world_size**2] == 1: #north
				position = (position[0], position[1] + 1)
			elif curr_state_with_action[world_size**2 + 1] == 1: #east
				position = (position[0] + 1, position[1])
			elif curr_state_with_action[world_size**2 + 2] == 1: #west
				position = (position[0] - 1, position[1])
			else: # south
				position = (position[0], position[1] - 1)
			reward = reward_potential(position, goal_position, gamma) - reward_potential(old_position, goal_position, gamma) - 1
			is_done = position == goal_position
			if is_done == True:
				reward += 10
			total_reward += reward
			agent.update_reward_history_tree(curr_state_with_action, reward)
			agent.remember(curr_state_with_action, reward, position_to_state(position, world_size), is_done)
			agent.replay_mem(params['batch_size'], goal_position, i, j)
		average_reward = total_reward / j
		average_rewards.append(j)
		episodes.append(i+1)
		agent.reset_reward_trail()
	return average_rewards, episodes

params = define_parameters()
rewards = []
episodes = []
for i in range(params['run_times_for_performance_average']):
	seed = i + 200
	random.seed(seed)  # Set random seed for Python's random module
	np.random.seed(seed)  # Set random seed for NumPy's random module
	torch.manual_seed(seed)

	agent = DQNAgent(params)
	new_rewards, new_episodes = train_agent_and_sample_performance(agent, params, i)

	if len(rewards) == 0:
		rewards, episodes = new_rewards, new_episodes
	else:
		rewards = [(x + y) for x, y in zip(rewards, new_rewards)]
rewards = [x / params['run_times_for_performance_average'] for x in rewards]
plot(params['reward_trail_length'], 'average rewards', 'episods', rewards, episodes, 'DQN_symmetry.csv')