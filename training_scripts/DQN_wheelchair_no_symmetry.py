import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import csv
import torch
import random

from Robots.WheelChair_v1 import WheelChairRobot
from .DQN import DQNAgent, DEVICE

def define_parameters():
	params = dict()
	# Neural Network
	params['learning_rate'] = 0.001
	params['weight_decay'] = 0
	params['first_layer_size'] = 64    # neurons in the first layer
	params['second_layer_size'] = 48   # neurons in the second layer
	params['third_layer_size'] = 32    # neurons in the third layer
	params['iterations'] = 5000
	params['memory_size'] = 1000
	params['batch_size'] = 8
	params['gamma'] = 0.98
	params['epsilon'] = 0.1
	params['epsilon_decay'] = 0.995
	params['epsilon_minimum'] = 0.1
	params['action_bins'] = 20
	params['action_lowest'] = -2.0
	params['action_highest'] = 2.0
	params['robot_reset_iterations'] = 100
	params['target_model_update_iterations'] = 100
	params['run_times_for_performance_average'] = 1
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

def include_reward_signal(robot_state, reward_signal):
	return robot_state+ (reward_signal, )

def measure_performance(DQN_agent, itr = 300):
	robot = WheelChairRobot(t_interval = 0.25)
	for i in range(itr):
		phidot, psidot = get_action_from_index(DQN_agent.select_action_index(include_reward_signal(robot.state,0), True), params['action_lowest'], params['action_highest'], params['action_bins'])
		robot.move((phidot, psidot))
	distance = robot.x
	robot.reset()
	for i in range(itr):
		phidot, psidot = get_action_from_index(DQN_agent.select_action_index(include_reward_signal(robot.state,1), True), params['action_lowest'], params['action_highest'], params['action_bins'])
		robot.move((phidot, psidot))
	distance += -robot.x
	robot.reset()
	for i in range(itr):
		phidot, psidot = get_action_from_index(DQN_agent.select_action_index(include_reward_signal(robot.state,2), True), params['action_lowest'], params['action_highest'], params['action_bins'])
		robot.move((phidot, psidot))
	distance += robot.y
	robot.reset()
	for i in range(itr):
		phidot, psidot = get_action_from_index(DQN_agent.select_action_index(include_reward_signal(robot.state,3), True), params['action_lowest'], params['action_highest'], params['action_bins'])
		robot.move((phidot, psidot))
	distance += -robot.y
	return distance

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
	robot = WheelChairRobot(t_interval = 0.25)
	distances = []
	iteration_times = []
	reward_signal = random.randint(0,3)
	for i in range(params['iterations']):
		if i % 100 == 0:
			print(f'{run_iteration}th running, iterations: {i}')
			distances.append(measure_performance(agent))
			iteration_times.append(i)
		curr_x = robot.x
		curr_y = robot.y
		curr_state = include_reward_signal(robot.state,reward_signal)
		action_index = agent.select_action_index(curr_state, True)
		phidot, psidot = get_action_from_index(action_index, params['action_lowest'], params['action_highest'], params['action_bins'])
		robot.move((phidot, psidot))
		if reward_signal == 0:
			reward = robot.x - curr_x
		elif reward_signal == 1:
			reward = -(robot.x - curr_x)
		elif reward_signal == 2:
			reward = robot.y - curr_y
		else:
			reward = -(robot.y - curr_y)

		new_state = include_reward_signal(robot.state, reward_signal)
		# store the new data into a long term memory
		agent.remember(curr_state, action_index, reward, new_state)
		agent.replay_mem(params['batch_size'], i)
		if i % params['robot_reset_iterations'] == 0 and i != 0:
			robot.reset()
			reward_signal = random.randint(0,3)
	return distances, iteration_times

params = define_parameters()
distances = []
iteration_times = []
for i in range(params['run_times_for_performance_average']):
	seed = i + 5
	random.seed(seed)  # Set random seed for Python's random module
	np.random.seed(seed)  # Set random seed for NumPy's random module
	torch.manual_seed(seed)

	agent = DQNAgent(params)
	new_distances, new_iteration_times = train_agent_and_sample_performance(agent, params, i)

	if len(distances) == 0:
		distances, iteration_times = new_distances, new_iteration_times
	else:
		distances = [(x + y) for x, y in zip(distances, new_distances)]
distances = [x / params['run_times_for_performance_average'] for x in distances]
plot('no symmetry', 'distances after 300 actions', 'training iterations', distances, iteration_times, 'DQN_wheelchair_no_symmetry.csv')