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

num_episodes = 200
print_all = True
no_symmetry_path="symmetry_result/9_no_symmetry"
symmetry_path="symmetry_result/9_symmetry(15)-1.0,filter(True)"
number_of_results = get_new_result_index(symmetry_path)
number_of_results = 100

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
	#ax.fill_between(episodes, symmetry_mean - symmetry_std_deviation, symmetry_mean + symmetry_std_deviation, alpha=0.3, color=color)

if False: #print only no_symmetry
	fig, ax = plt.subplots()
	number_of_results = get_new_result_index(no_symmetry_path)
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
	colors = ["r", "g", "y", "c", "m", "k", "w", "0.5", "orange", "purple", "lime", "maroon"]
	for k in [3, 5, 7, 11]:
		for reward_filter, weight in [(False, 0.6),(True,1.0)]:
			action = 9
			add_to_plot(ax, f'symmetry_result/{action}_symmetry({k})-{weight},filter({reward_filter})', f"k:{k},equivalence:{weight},reward_filter:{reward_filter}", colors[index], number_of_results)
			index = index + 1
	for k in [13]:
		for reward_filter, weight in [(False, 0.5),(True,1.0)]:
			action = 9
			add_to_plot(ax, f'symmetry_result/{action}_symmetry({k})-{weight},filter({reward_filter})', f"k:{k},equivalence weight:{weight},reward_filter:{reward_filter}", colors[index], number_of_results)
			index = index + 1
	for k in [15]:#[13,15]:
		for reward_filter, weight in [(False, 0.5),(True,1.0)]:
			action = 9
			add_to_plot(ax, f'symmetry_result/{action}_symmetry({k})-{weight},filter({reward_filter})', f"k:{k},equivalence weight:{weight},reward_filter:{reward_filter}", colors[index], number_of_results)
			index = index + 1
	plt.xlabel("episodes")
	plt.ylabel("rewards")
	plt.title(f'Rewards averaged from {number_of_results} runs')
	ax.legend()
	plt.show(block=True)
else:
	for k in [3,5,7,11]:
		for reward_filter, weight in [(False, 0.6), (False, 1.0),(True,1.0)]:
			fig, ax = plt.subplots()
			action = 9
			symmetry_path = f'symmetry_result/{action}_symmetry({k})-{weight},filter({reward_filter})'
			number_of_results = get_new_result_index(symmetry_path)
			add_to_plot(ax, f'symmetry_result/{action}_symmetry({k})-{weight},filter({reward_filter})', f"k:{k},weight:{weight},filter:{reward_filter}", 'r', number_of_results)
			add_to_plot(ax, no_symmetry_path, "no symmetry mean", 'b', number_of_results)
			plt.xlabel("episodes")
			plt.ylabel("rewards")
			plt.title(f'Rewards averaged from {number_of_results} runs')
			ax.legend()
			plt.show(block=True)
	for k in [13,15]:
		for reward_filter, weight in [(False, 0.5),(True,1.0)]:
			fig, ax = plt.subplots()
			action = 9
			symmetry_path = f'symmetry_result/{action}_symmetry({k})-{weight},filter({reward_filter})'
			number_of_results = get_new_result_index(symmetry_path)
			add_to_plot(ax, f'symmetry_result/{action}_symmetry({k})-{weight},filter({reward_filter})', f"k:{k},weight:{weight},filter:{reward_filter}", 'r', number_of_results)
			add_to_plot(ax, no_symmetry_path, "no symmetry mean", 'b', number_of_results)
			plt.xlabel("episodes")
			plt.ylabel("rewards")
			plt.title(f'Rewards averaged from {number_of_results} runs')
			ax.legend()
			plt.show(block=True)
				
				