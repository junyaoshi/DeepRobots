import numpy as np
import torch
import random
import csv
import os
import matplotlib as mpl
import math
import sys
sys.path.append('/Users/minuk.lee/Desktop/Research-Dear/DeepRobots')
import Robots.ChaplyginBeanie
mpl.use('TkAgg')

def parameters():
	params = dict()
	params['run_times_for_performance_average'] = 100
	params['episodes'] = 400
	params['episode_length'] = 45
	
	params['learning_rate'] = 0.001
	params['weight_decay'] = 0
	params['gamma'] = 0.9
	params['memory_size'] = 20000
	params['batch_size'] = 16
	params['epsilon'] = 1.0
	params['epsilon_decay'] = 0.995
	params['epsilon_minimum'] = 0.1
	params['target_model_update_iterations'] = round(params['episode_length'] / 2)

	params['first_layer_size'] = 256    # neurons in the first layer
	params['second_layer_size'] = 256   # neurons in the second layer

	params['state_size'] = 5 #(self.x, self.y, self.theta, self.JLT, self.JRW)
	params['action_bins'] = 41
	params['action_lowest'] = -1.0
	params['action_highest'] = 1.0
	params['number_of_actions'] = params['action_bins']
	return params

def get_new_result_index(path):
	if not os.path.exists(path):
		os.makedirs(path)
	index = 0
	file_name = f"{index}.csv"
	file_path = os.path.join(path, file_name)
	while os.path.exists(file_path):
		index += 1
		file_name = f"{index}.csv"
		file_path = os.path.join(path, file_name)
	return index

def save_result(path, reward):
	index = get_new_result_index(path)
	file_name = f"{index}.csv"
	file_path = os.path.join(path, file_name)
	with open(file_path, mode='w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(reward)

def run(params, agent_type, save_file_path):
	rewards = []
	for i in range(params['run_times_for_performance_average']):
		agent = agent_type(params)
		set_seed(get_new_result_index(save_file_path)+5)
		rewards = train_agent_and_sample_performance(agent, params, i)
		save_result(save_file_path, rewards)
	return rewards

def get_val_from_index(ind, low, high, n_bins):
	bin_size = (high - low)/(n_bins-1)
	true_val = ind * bin_size + low
	return true_val

def write_(str):
    log_file = open("log_debugging.txt", "a")
    log_file.write(f'{str}\n')
    log_file.close()

def train_agent_and_sample_performance(agent, params, run_iteration):
	rewards_for_each_episode = []
	for i in range(params['episodes']):
		agent.on_episode_start(i)
		if i % 20 == 0:
			print(f'{run_iteration}th running, epidoes: {i}')
		robot = Robots.ChaplyginBeanie.ChaplyginBeanie(x=random.uniform(-50,50), y=random.uniform(-50,50), t_interval = 1.0, theta=random.uniform(-3.14, 3.14))
		curr_x = robot.x
		current_state = robot.state
		total_reward = 0
		for j in range(params['episode_length']):
			action = agent.select_action_index(current_state, True)
			phidot = get_val_from_index(action, params['action_lowest'], params['action_highest'], params['action_bins'])
			robot.move(phidot)
			reward = robot.x - curr_x
			total_reward += reward
			new_state = robot.state

			#formatted_current_state = tuple("{:.2f}".format(x) for x in current_state)
			#write_(f'{formatted_current_state} - {phidot:.2f}({action}). r:{reward}')

			agent.on_new_sample(current_state, action, reward, new_state)
			agent.replay_mem(params['batch_size'])
			current_state = new_state
			curr_x = robot.x
		agent.on_terminated()
		rewards_for_each_episode.append(total_reward)
	agent.on_finished()
	return rewards_for_each_episode

def set_seed(seed):
	random.seed(seed)  # Set random seed for Python's random module
	np.random.seed(seed)  # Set random seed for NumPy's random module
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)