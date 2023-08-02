import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import csv
import matplotlib as mpl
import math
mpl.use('TkAgg')

# need to import gymnasium. pip install gymnaisum
# pip install "gymnasium[all]"
import gymnasium as gym

def parameters():
	params = dict()
	params['seed'] = 180
	set_seed(params['seed'])
	params['run_times_for_performance_average'] = 200
	params['episodes'] = 100
	params['episode_length'] = 500

	params['include_reward_shaping'] = True
	params['discretize_states'] = False
	params['state_discretize_bins'] = 128
	
	params['learning_rate'] = 0.001
	params['weight_decay'] = 0
	params['gamma'] = 0.99
	params['memory_size'] = 100000
	params['batch_size'] = 4
	params['epsilon'] = 1.0
	params['epsilon_decay'] = 0.98
	params['epsilon_minimum'] = 0.1
	params['episodes_with_random_policy'] = 25

	params['first_layer_size'] = 100
	params['second_layer_size'] = 100
	params['number_of_actions'] = 2
	return params

def title(params):
	runtime = params['run_times_for_performance_average']
	seed = params['seed']
	reward_shaping = params['include_reward_shaping']
	return f'{runtime} runs, seed: {seed}, rs:{reward_shaping}'

def run(params, agent_type):
	rewards = []
	episodes = []
	for i in range(params['run_times_for_performance_average']):
		agent = agent_type(params)
		new_rewards, new_episodes = train_agent_and_sample_performance(agent, params, i)

		if len(rewards) == 0:
			rewards, episodes = new_rewards, new_episodes
		else:
			rewards = [(x + y) for x, y in zip(rewards, new_rewards)]
	rewards = [x / params['run_times_for_performance_average'] for x in rewards]
	return rewards, episodes

def train_agent_and_sample_performance(agent, params, run_iteration):
	total_rewards = []
	episodes = []
	#env = gym.make("CartPole-v0",render_mode="human")
	env = gym.make("CartPole-v1")
	for i in range(params['episodes']):
		agent.on_episode_start(i)
		if i % 10 == 0:
			print(f'{run_iteration}th running, epidoes: {i}')
		current_state, info = env.reset(seed = random.randint(0,100000000000))
		if params['discretize_states'] == True:
			current_state = discretize_state(current_state, params['state_discretize_bins'])
		total_reward = 0
		is_random_policy = i < params['episodes_with_random_policy']
		for j in range(params['episode_length']):
			action = agent.select_action_index(current_state, True, is_random_policy)
			new_state, reward, terminated, truncated, info = env.step(action)
			if params['discretize_states'] == True:
				new_state = discretize_state(new_state, params['state_discretize_bins'])

			if params['include_reward_shaping'] == True:
				reward += reward_add(new_state)
			total_reward += reward
			agent.on_new_sample(current_state, action, reward, new_state, terminated)
			agent.replay_mem(params['batch_size'], not is_random_policy)
			current_state = new_state
			if terminated:
				break
		agent.on_terminated()
		total_rewards.append(total_reward)
		episodes.append(i+1)
	env.close()
	agent.on_finished()
	return total_rewards, episodes

def discretize_state(state, bins):
	def get_discretized(value, min, max):
		gap = max - min
		block = gap / bins
		return np.clip(round((value - min)/block)*block + min, min, max)
	position = get_discretized(state[0], -2.4, 2.4)
	velocity = get_discretized(state[1], -3.0, 3.0)
	angle = get_discretized(state[2], -0.5, 0.5)
	angular_velocity = get_discretized(state[3], -2.0, 2.0)
	return (position, velocity, angle, angular_velocity)

def reward_add(state):
	discretize_bins = 19
	def get_discretized(value, min, max):
		gap = max - min
		block = gap / discretize_bins
		return math.floor((2*value/block + 1)/2)
	position_d = get_discretized(state[0], -2.4, 2.4)
	velocity_d = get_discretized(state[1], -3.0, 3.0)
	angle_d = get_discretized(state[2], -0.5, 0.5)
	return 1 - abs(angle_d) / (abs(discretize_bins - 1) / 2)
	angular_velocity_d = get_discretized(state[3], -2.0, 2.0)
	#return 1 - ((angle_d**2 + position_d**2 + angular_velocity_d**2 + velocity_d**2) / ((discretize_bins - 1)**2))

def set_seed(seed):
	random.seed(seed)  # Set random seed for Python's random module
	np.random.seed(seed)  # Set random seed for NumPy's random module
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

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

def plot_csv(title, file_name):
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
		plot(title, ylabel,xlabel,values,times)