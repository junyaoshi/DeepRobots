import json
import os
import numpy as np
import matplotlib.pyplot as plt

algo = "SAC"

trials = ['trial_6_06', 'trial_6_07', 'trial_6_08', 'trial_6_09', 'trial_6_10']

x_poss = np.zeros(101)
y_poss = np.zeros(101)
thetas = np.zeros(101)
times = np.zeros(101)
a1s = np.zeros(101)
a2s = np.zeros(101)
a1dots = np.zeros(101)
a2dots = np.zeros(101)

x_poss_std = []
y_poss_std = []
thetas_std = []
times_std = []
a1s_std = []
a2s_std = []
a1dots_std = []
a2dots_std = []

for trial in trials:
	results_dir = os.path.join("results", "LearningResults", algo + "_IdealFluidSwimmer", trial)
	with open(results_dir + '/' + algo + '_data_' + trial + '.txt') as file:
		data_dict = json.load(file)
		x_poss = np.add(x_poss, data_dict['x_poss'])
		y_poss = np.add(y_poss, data_dict['y_poss'])
		thetas = np.add(thetas, data_dict['thetas'])
		a1s = np.add(a1s, data_dict['a1s'])
		a2s = np.add(a2s, data_dict['a2s'])
		a1dots = np.add(a1dots, data_dict['a1dots'])
		a2dots = np.add(a2dots, data_dict['a2dots'])
		times = np.array(data_dict['times'])
		x_poss_std.append(data_dict['x_poss'])
		y_poss_std.append(data_dict['y_poss'])
		thetas_std.append(data_dict['thetas'])
		a1s_std.append(data_dict['a1s'])
		a2s_std.append(data_dict['a2s'])
		a1dots_std.append(data_dict['a1dots'])
		a2dots_std.append(data_dict['a2dots'])

x_poss_std = np.std(np.array(x_poss_std), axis=0)
y_poss_std = np.std(np.array(y_poss_std), axis=0)
thetas_std = np.std(np.array(thetas_std), axis=0)
a1s_std = np.std(np.array(a1s_std), axis=0)
a2s_std = np.std(np.array(a2s_std), axis=0)
a1dots_std = np.std(np.array(a1dots_std), axis=0)
a2dots_std = np.std(np.array(a2dots_std), axis=0)

x_poss = x_poss/5
y_poss = y_poss/5
thetas = thetas/5
a1s = a1s/5
a2s = a2s/5
a1dots = a1dots/5
a2dots = a2dots/5

plots_dir = os.path.join(algo + '_avg')

plot_style = "--bo"
marker_size = 3

plt.plot(x_poss, y_poss, plot_style, markersize=marker_size)
plt.xlabel('x')
plt.ylabel('y')
plt.title('y vs x')
plt.savefig(os.path.join(plots_dir, 'y vs x' + '.png'))
plt.show()

plt.plot(times, a1s, plot_style, markersize=marker_size)
plt.ylabel('a1 displacements')
plt.xlabel('time')
plt.fill_between(times, a1s+a1s_std, a1s-a1s_std, alpha=0.5)
plt.savefig(os.path.join(plots_dir, 'a1 displacements' + '.png'))
plt.show()

plt.plot(times, a2s, plot_style, markersize=marker_size)
plt.ylabel('a2 displacements')
plt.xlabel('time')
plt.fill_between(times, a2s+a2s_std, a2s-a2s_std, alpha=0.5)
plt.savefig(os.path.join(plots_dir, 'a2 displacements' + '.png'))
plt.show()

plt.plot(times, x_poss, plot_style, markersize=marker_size)
plt.ylabel('x positions')
plt.xlabel('time')
plt.fill_between(times, x_poss+x_poss_std, x_poss-x_poss_std, alpha=0.5)
plt.savefig(os.path.join(plots_dir, 'x positions' + '.png'))
plt.show()

plt.plot(times, y_poss, plot_style, markersize=marker_size)
plt.ylabel('y positions')
plt.xlabel('time')
plt.fill_between(times, y_poss+y_poss_std, y_poss-y_poss_std, alpha=0.5)
plt.savefig(os.path.join(plots_dir, 'y positions' + '.png'))
plt.show()

plt.plot(times, thetas, plot_style, markersize=marker_size)
plt.ylabel('thetas')
plt.xlabel('time')
plt.fill_between(times, thetas+thetas_std, thetas-thetas_std, alpha=0.5)
plt.savefig(os.path.join(plots_dir, 'thetas' + '.png'))
plt.show()

plt.plot(times, a1dots, plot_style, markersize=marker_size)
plt.ylabel('a1dot')
plt.xlabel('time')
plt.fill_between(times, a1dots+a1dots_std, a1dots-a1dots_std, alpha=0.5)
plt.savefig(os.path.join(plots_dir, 'a1dot' + '.png'))
plt.show()

plt.plot(times, a2dots, plot_style, markersize=marker_size)
plt.ylabel('a2dot')
plt.xlabel('time')
plt.fill_between(times, a2dots+a2dots_std, a2dots-a2dots_std, alpha=0.5)
plt.savefig(os.path.join(plots_dir, 'a2dot' + '.png'))
plt.show()
plt.close()



