import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import csv
import matplotlib as mpl
import math
import sys
sys.path.append('/Users/minuk.lee/Desktop/Research-Dear/DeepRobots')
import Robots.WheelChair_v1
mpl.use('TkAgg')

def parameters():
	params = dict()
	params['seed'] = 0
	set_seed(params['seed'])
	params['run_times_for_performance_average'] = 50
	params['episodes'] = 50
	params['episode_length'] = 100
	
	params['learning_rate'] = 0.001
	params['weight_decay'] = 0
	params['gamma'] = 0.99
	params['memory_size'] = 10000
	params['batch_size'] = 4
	params['epsilon'] = 1.0
	params['epsilon_decay'] = 0.98
	params['epsilon_minimum'] = 0.1
	params['episodes_with_random_policy'] = 5

	params['first_layer_size'] = 64    # neurons in the first layer
	params['second_layer_size'] = 64   # neurons in the second layer

	params['state_size'] = 3 #theta
	params['action_bins'] = 10
	params['action_lowest'] = -1.0
	params['action_highest'] = 1.0
	params['number_of_actions'] = params['action_bins'] ** 2
	return params

def title(params):
	runtime = params['run_times_for_performance_average']
	seed = params['seed']
	return f'{runtime} runs, seed: {seed}'

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

def get_val_from_index(ind, low, high, n_bins):
	bin_size = (high - low)/n_bins
	true_val = ind * bin_size + low
	return true_val

def get_action_from_index(action_index, action_lowest, action_highest, action_bins):
	phidot_index = action_index//action_bins
	psidot_index = action_index % action_bins

	phidot_true = get_val_from_index(phidot_index, action_lowest, action_highest, action_bins)
	psidot_true = get_val_from_index(psidot_index, action_lowest, action_highest, action_bins)

	return phidot_true, psidot_true

def train_agent_and_sample_performance(agent, params, run_iteration):
	total_rewards = []
	episodes = []
	for i in range(params['episodes']):
		agent.on_episode_start(i)
		if i % 10 == 0:
			print(f'{run_iteration}th running, epidoes: {i}')
		robot = Robots.WheelChair_v1.WheelChairRobot(t_interval = 1.0)
		curr_x = robot.x
		current_state = robot.state
		total_reward = 0
		is_random_policy = i < params['episodes_with_random_policy']
		for j in range(params['episode_length']):
			action = agent.select_action_index(current_state, True, is_random_policy)
			phidot, psidot = get_action_from_index(action, params['action_lowest'], params['action_highest'], params['action_bins'])
			robot.move((phidot, psidot))
			reward = robot.x - curr_x
			total_reward += reward
			new_state = robot.state
			agent.on_new_sample(current_state, action, reward, new_state)
			agent.replay_mem(params['batch_size'], not is_random_policy)
			current_state = new_state
			curr_x = robot.x
		agent.on_terminated()
		total_rewards.append(total_reward)
		episodes.append(i+1)
	agent.on_finished()
	return total_rewards, episodes

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
	plt.show(block=True)

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