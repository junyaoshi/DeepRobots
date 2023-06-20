import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import csv
import torch
import torch.optim as optim
import random

from Robots.WheelChair_v1 import WheelChairRobot
from .DQN import DQNAgent, DEVICE

def define_parameters():
	params = dict()
	# Neural Network
	params['learning_rate'] = 0.00013629
	params['weight_decay'] = 0
	params['first_layer_size'] = 32    # neurons in the first layer
	params['second_layer_size'] = 64   # neurons in the second layer
	params['third_layer_size'] = 32    # neurons in the third layer
	params['iterations'] = 10000		
	params['memory_size'] = 2500
	params['batch_size'] = 1000
	params['gamma'] = 0.9
	params['epsilon'] = 0.15
	params['action_bins'] = 30
	params['action_lowest'] = -1
	params['action_highest'] = 1
	params['memory_replay_iterations'] = 100
	params['run_times_for_performance_average'] = 50
	return params

def convert_to_index(num, low, high, n_bins):
	num_shift = num - low

	bin_size = (high - low)/n_bins

	#Account for edge case where num is high and bin_size is evenly divisible
	ind = int(min(n_bins-1, num_shift//bin_size))

	return ind

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

def select_action_index(DQN_agent, state, apply_epsilon_random):
	if apply_epsilon_random == True and random.uniform(0, 1) < params['epsilon']:
		return np.random.choice(params['action_bins'] ** 2) # phidot, psidot actions

	with torch.no_grad():
		state_tensor = torch.tensor(np.array(state)[np.newaxis, :], dtype=torch.float32).to(DEVICE)
		prediction = DQN_agent(state_tensor)
		return np.argmax(prediction.detach().cpu().numpy()[0])

def measure_performance(DQN_agent, itr = 100):
	robot = WheelChairRobot(t_interval = 1)
	x_pos = [0]
	times = [0]
	for i in range(itr):
		phidot, psidot = get_action_from_index(select_action_index(DQN_agent, robot.state, False), params['action_lowest'], params['action_highest'], params['action_bins'])
		robot.move((phidot, psidot))

		x_pos.append(robot.x)
		times.append(i)
	return x_pos, times

def plot(ylabel, xlabel, values, times, save_to_csv_file_name = ""):
	if save_to_csv_file_name != "":
		data = zip(values, times)
		with open(save_to_csv_file_name, 'w', newline='') as file:
			writer = csv.writer(file)
			writer.writerow([ylabel, xlabel])
			writer.writerows(data)
	plt.plot(times, values)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()

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

def train_agent_and_sample_performance(agent, params, run_iteration):
	robot = WheelChairRobot(t_interval = 1)
	distances = []
	iteration_times = []
	for i in range(params['iterations']):
		if i % 100 == 0:
			print(f'{run_iteration}th running, iterations: {i}')
			distances.append(measure_performance(agent)[0][-1])
			iteration_times.append(i)
			#plot('x position', 'moves', *locomote(robot, agent)) # to plot how the robot moved
		curr_x = robot.x
		curr_state = robot.state
		action_index = select_action_index(agent, curr_state, True)
		phidot, psidot = get_action_from_index(action_index, params['action_lowest'], params['action_highest'], params['action_bins'])
		robot.move((phidot, psidot))
		reward = robot.x - curr_x
		new_state = robot.state
		# train short memory base on the new action and state
		agent.train_short_memory(curr_state, action_index, reward, new_state)
		# store the new data into a long term memory
		agent.remember(curr_state, action_index, reward, new_state)
		if i % params['memory_replay_iterations'] == 0 and i != 0:
			agent.replay_mem(params['batch_size'])
	return distances, iteration_times

params = define_parameters()
distances = []
iteration_times = []
for i in range(params['run_times_for_performance_average']):
	agent = DQNAgent(params)
	agent = agent.to(DEVICE)
	agent.optimizer = optim.Adam(agent.parameters(), weight_decay=params['weight_decay'], lr=params['learning_rate'])
	new_distances, new_iteration_times = train_agent_and_sample_performance(agent, params, i)

	if len(distances) == 0:
		distances, iteration_times = new_distances, new_iteration_times
	else:
		distances = [(x + y) for x, y in zip(distances, new_distances)]
distances = [x / params['run_times_for_performance_average'] for x in distances]
plot('x distances after 100 actions', 'training iterations', distances, iteration_times, 'DQN_wheelchair_no_symmetry.csv')