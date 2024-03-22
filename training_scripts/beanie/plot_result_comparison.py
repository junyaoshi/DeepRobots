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

num_episodes = 400
action = 41
print_all = False
no_symmetry_path=f"symmetry_result/{action}_no_symmetry"
no_symmetry_path_2=f"symmetry_result/{action}_no_symmetry_reduced_no_position"
number_of_results = min(get_new_result_index(no_symmetry_path),get_new_result_index(no_symmetry_path_2))

episodes = np.arange(1, num_episodes+1)
def add_to_plot(ax, path, label, color, number_of_results):
	symmetry_rewards = []
	for i in range(number_of_results):
		file_name = f"{i}.csv"
		symmetry_rewards.append(read_rewards(os.path.join(path, file_name)))
	symmetry_rewards = np.array([inner_array[:num_episodes] for inner_array in symmetry_rewards], dtype=float)
	symmetry_mean = np.mean(symmetry_rewards, axis=0)
	symmetry_std_deviation = np.std(symmetry_rewards, axis=0)
	ax.plot(episodes, symmetry_mean, label=label, color=color)
	ax.fill_between(episodes, symmetry_mean - symmetry_std_deviation, symmetry_mean + symmetry_std_deviation, alpha=0.3, color=color)

if True: #print only no_symmetry
	fig, ax = plt.subplots()
	add_to_plot(ax, no_symmetry_path, "no symmetry mean", 'b', number_of_results)
	add_to_plot(ax, no_symmetry_path_2, "no symmetry mean reduced no position", 'y', number_of_results)
	plt.xlabel("episodes")
	plt.ylabel("rewards")
	plt.title(f'Rewards averaged from {number_of_results} runs')
	ax.legend()
	plt.show(block=True)
if print_all:
	fig, ax = plt.subplots()
	number_of_results = 4
	add_to_plot(ax, no_symmetry_path, "Naive DQN", 'b', number_of_results)
	index = 0
	colors = ["r", "g", "y", "c", "m", "k", "w", "0.5", "orange", "purple", "lime", "maroon"]
	for k in [10,20,30,40,50,60]:
		for reward_filter, weight in [(False, 0.5),(True,1.0)]:
			add_to_plot(ax, f'symmetry_result/{action}_symmetry({k})-{weight},filter({reward_filter})_mnn', f"k:{k},equivalence:{weight},reward_filter:{reward_filter}", colors[index], number_of_results)
			index = index + 1
	plt.xlabel("episodes")
	plt.ylabel("rewards")
	plt.title(f'Rewards averaged from {number_of_results} runs')
	ax.legend()
	plt.show(block=True)
else:
	for k in [5,9,13]:
		for reward_filter, weight in [(False, 1.0),(True,1.0)]:
			fig, ax = plt.subplots()
			symmetry_path = f'symmetry_result/{action}_symmetry({k})-{weight},filter({reward_filter})_mature_50'
			number_of_results = get_new_result_index(symmetry_path)
			add_to_plot(ax, f'symmetry_result/{action}_symmetry({k})-{weight},filter({reward_filter})_mature_50', f"k:{k},weight:{weight},filter:{reward_filter}_mature_50", 'r', number_of_results)
			#add_to_plot(ax, no_symmetry_path_2, "no symmetry mean reduced", 'y', number_of_results)
			add_to_plot(ax, no_symmetry_path, "no symmetry mean", 'b', number_of_results)
			plt.xlabel("episodes")
			plt.ylabel("rewards")
			plt.title(f'Rewards averaged from {number_of_results} runs')
			ax.legend()
			plt.show(block=True)