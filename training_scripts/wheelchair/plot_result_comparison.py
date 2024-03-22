import os
import csv
import numpy as np
import matplotlib.pyplot as plt

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

def read_rewards(file_path):
	with open(file_path, 'r') as file:
		reader = csv.reader(file)
		for row in reader:
			return row

action = 9
num_episodes = 200
print_all = True
no_symmetry_path="wheelchair_result/9_no_symmetry"
number_of_results = 100#get_new_result_index(no_symmetry_path)

episodes = np.arange(1, num_episodes+1)
font_size = 15
def add_to_plot(ax, path, label, color, number_of_results):
	symmetry_rewards = []
	for i in range(number_of_results):
		file_name = f"{i}.csv"
		symmetry_rewards.append(read_rewards(os.path.join(path, file_name)))
	symmetry_rewards = np.array([inner_array[:num_episodes] for inner_array in symmetry_rewards], dtype=float)
	symmetry_mean = np.mean(symmetry_rewards, axis=0)
	symmetry_std_deviation = np.std(symmetry_rewards, axis=0)
	plt.rcParams.update({'font.size': font_size})  # Change the font size to 12 for all elements
	ax.plot(episodes, symmetry_mean, label=label, color=color)
	#if color == 'r' or color == 'b':
	#ax.fill_between(episodes, symmetry_mean - symmetry_std_deviation, symmetry_mean + symmetry_std_deviation, alpha=0.3, color=color)

if False: #print only no_symmetry
	fig, ax = plt.subplots()
	number_of_results = min(get_new_result_index(no_symmetry_path),get_new_result_index(no_symmetry_path))
	add_to_plot(ax, no_symmetry_path, "no symmetry mean", 'b', number_of_results)
	plt.xlabel("episodes")
	plt.ylabel("rewards")
	plt.title(f'Rewards averaged from {number_of_results} runs')
	ax.legend()
	plt.show(block=True)
if print_all:
	fig, ax = plt.subplots()
	add_to_plot(ax, no_symmetry_path, "Naive DQN", 'b', number_of_results)
	index = 0
	colors = ["c", "g", "y", "r", "m", "k", "0.5", "orange", "purple", "lime", "maroon"]
	for k in [5, 11]:
		for reward_filter, weight in [(False, 0.6),(True,1.0)]:
			add_to_plot(ax, f'wheelchair_result/{action}_symmetry({k})-{weight},filter({reward_filter})', f"k:{k},equivalence:{weight},reward_filter:{reward_filter}", colors[index], number_of_results)
			index = index + 1
	plt.xlabel("episodes", fontsize=font_size)
	plt.ylabel("rewards", fontsize=font_size)
	plt.title(f'Rewards averaged from {number_of_results} runs')
	ax.legend()
	plt.show(block=True)
else:
	for k in [5,11]:
		for reward_filter, weight in [(False, 0.6), (True,1.0)]:
			fig, ax = plt.subplots()
			symmetry_path = f'wheelchair_result/{action}_symmetry({k})-{weight},filter({reward_filter})_no_mean'
			number_of_results = min(get_new_result_index(symmetry_path), get_new_result_index(no_symmetry_path))
			add_to_plot(ax, f'wheelchair_result/{action}_symmetry({k})-{weight},filter({reward_filter})_no_mean', f"k:{k},weight:{weight},filter:{reward_filter}_no_mean", 'r', number_of_results)
			add_to_plot(ax, no_symmetry_path, "no equivalent", 'b', number_of_results)
			plt.xlabel("episodes")
			plt.ylabel("rewards")
			plt.title(f'Rewards averaged from {number_of_results} runs')
			ax.legend()
			plt.show(block=True)