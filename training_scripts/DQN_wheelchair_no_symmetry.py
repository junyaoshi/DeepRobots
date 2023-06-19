import numpy as np
from math import pi
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import random

from Robots.WheelChair_v1 import WheelChairRobot
from .DQN import DQNAgent, DEVICE

def define_parameters():
	params = dict()
	# Neural Network
	params['learning_rate'] = 0.00013629
	params['first_layer_size'] = 200    # neurons in the first layer
	params['second_layer_size'] = 20   # neurons in the second layer
	params['third_layer_size'] = 50    # neurons in the third layer
	params['iterations'] = 2000		
	params['memory_size'] = 1500
	params['batch_size'] = 300
	params['gamma'] = 0.9
	params['epsilon'] = 0.2
	params['action_bins'] = 30
	params['action_lowest'] = -1
	params['action_highest'] = 1
	params['memory_replay_iterations'] = 100
    # Settings
	params['weights_path'] = 'weights/weights.h5'
	params['train'] = True
	params['load_weights'] = False
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

def locomote(robot, DQN_agent, itr = 100):
	robot.reset()

	x_pos = [0]
	times = [0]
	for i in range(itr):
		phidot, psidot = get_action_from_index(select_action_index(DQN_agent, robot.state, False), params['action_lowest'], params['action_highest'], params['action_bins'])
		robot.move((phidot, psidot))

		x_pos.append(robot.x)
		times.append(i)
	return x_pos, times

def plot(ylabel, xlabel, values, times):
	plt.plot(times, values)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()

params = define_parameters()
agent = DQNAgent(params)
agent = agent.to(DEVICE)
agent.optimizer = optim.Adam(agent.parameters(), weight_decay=0, lr=params['learning_rate'])
robot = WheelChairRobot(t_interval = 1)

distances = [0]
times = [0]
for i in range(params['iterations']):
	if i % 100 == 0:
		print(f'iterations: {i}')
		distances.append(locomote(robot, agent)[0][-1])
		times.append(i)
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
if params['train']:
    model_weights = agent.state_dict()
    torch.save(model_weights, params["weights_path"])
plot('x distances after 100 actions', 'training iterations', distances, times)