import numpy as np
from math import cos, sin, pi
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

from Robots.WheelChair_v1 import WheelChairRobot

from utils.csv_generator import generate_csv

N_BINS = 30

A_LOWER = -1; A_UPPER = 1

def construct_table(
	theta_bins = N_BINS,
	phi_bins = N_BINS,
	psi_bins = N_BINS,
	phidot_bins = N_BINS,
	psidot_bins = N_BINS):
	
	q_table = np.zeros(
		(theta_bins, 
		phi_bins, 
		psi_bins, 
		phidot_bins * psidot_bins)
	)

	return q_table

def convert_to_index(num, low, high, n_bins):
	num_shift = num - low
	high_shift = high - low

	bin_size = (high - low)/n_bins

	#Account for edge case where num is high and bin_size is evenly divisible
	ind = int(min(n_bins-1, num_shift//bin_size))

	return ind

def get_val_from_index(ind, low, high, n_bins):
	bin_size = (high - low)/n_bins
	true_val = ind * bin_size + low
	return true_val


def get_action_from_index(ind, low, high, phidot_bins, psidot_bins):
	phidot_ind = ind//phidot_bins
	psidot_ind = ind % phidot_bins

	phidot_true = get_val_from_index(phidot_ind, low, high, phidot_bins)
	psidot_true = get_val_from_index(psidot_ind, low, high, psidot_bins)

	return phidot_true, psidot_true

def get_policy_score(q_table):
	tot = 0
	
	for theta_ind in range(N_BINS):
		for phi_ind in range(N_BINS):
			for psi_ind in range(N_BINS):
				theta = get_val_from_index(theta_ind, -pi, pi, N_BINS)
				phi = get_val_from_index(phi_ind, -pi, pi, N_BINS)
				psi = get_val_from_index(psi_ind, -pi, pi, N_BINS)

				robot = WheelChairRobot(theta=theta, phi=phi, psi=psi)
				
				action_ind = np.argmax(q_table[theta_ind, phi_ind, psi_ind, :])
				phidot_true, psidot_true = get_action_from_index(action_ind, A_LOWER, A_UPPER, N_BINS, N_BINS)
				action = (phidot_true, psidot_true)
				
				robot.move(action)

				tot += robot.x

	return tot

def get_sym_score_theta(mat, theta_bins):
	#NOTE: Right now assumes theta_bins is even
	mid_ind = int(theta_bins/2)
	sum_across_mid = mat[0:mid_ind, :,] + mat[mid_ind:, :]
	var_along_col = np.var(sum_across_mid, axis=0)
	sym_score = np.average(var_along_col, axis=None)
	return sym_score


def e_greedy(eps, theta, phi, psi, q_table):
	a_vals = q_table[theta, phi, psi, :]
	#print(a_vals)

	max_val = np.max(a_vals)
	#print(max_val)
	
	max_indices = np.squeeze(np.argwhere(a_vals == max_val), axis = 1)	

	rand_num = np.random.uniform()

	if(rand_num <= eps):
		ind = np.random.choice(max_indices)
	else:
		indices = np.squeeze(np.argwhere(a_vals < max_val), axis = 1)
		if(indices.size == 0):
			indices = max_indices
		ind = np.random.choice(indices)
	
	phidot_true, psidot_true = get_action_from_index(ind, A_LOWER, A_UPPER, N_BINS, N_BINS)
	return ind, phidot_true, psidot_true

#Assumes state is initialized as all 0
def q_learn(q_table, robot, itr = 300000, gamma = 0.85, eps = 0.8, alpha = 0.3):
	#Initialize action (starting state assumed to be 0)
	for i in range(itr):
		if(i % 10000 == 0 and i != 0):
			print("iteration:", i)
			print(np.max(q_table))
		#Update s to current config (equivalent to s <- s' step)
		curr_state = robot.state
		#print(curr_state)
		theta = curr_state[0]; phi = curr_state[1]; psi = curr_state[2] #s
		theta_ind = convert_to_index(theta, -pi, pi, N_BINS)
		phi_ind = convert_to_index(phi, -pi, pi, N_BINS)
		psi_ind = convert_to_index(psi, -pi, pi, N_BINS)
		
		#Get curr_x for reward calculation
		curr_x = robot.x

		#Move to s'
		action_ind, phidot_val, psidot_val = e_greedy(eps, theta_ind, phi_ind, psi_ind, q_table)
		action = (phidot_val, psidot_val)
		robot.move(action)

		#Do reward calculation
		new_x = robot.x
		reward = new_x - curr_x

		#Get s'
		new_state = robot.state
		theta_p = new_state[0]; phi_p = new_state[1]; psi_p = new_state[2]
		theta_p_ind = convert_to_index(theta_p, -pi, pi, N_BINS)
		phi_p_ind = convert_to_index(phi_p, -pi, pi, N_BINS)
		psi_p_ind = convert_to_index(psi_p, -pi, pi, N_BINS)

		#Find Q(s, a)
		# print(theta_ind)
		# print(phi_ind)
		# print(psi_ind)
		q_s_a = q_table[theta_ind, phi_ind, psi_ind, action_ind]

		#From s', find max a' Q(s', a')
		q_s_a_p = np.max(q_table[theta_p_ind, phi_p_ind, psi_p_ind, :])

		#Update Q-value
		q_table[theta_ind, phi_ind, psi_ind, action_ind] = q_s_a + alpha * (reward + gamma * q_s_a_p - q_s_a)

def locomote(robot, q_table, itr = 100, save_csv = True):
	robot.reset()
	curr_state = (0, 0, 0)

	x_pos = [0]
	y_pos = [0]
	thetas = [0]
	phis = [0]
	psis = [0]
	times = [0]

	robot_params = []

	for i in range(itr):
		theta = curr_state[0]; phi = curr_state[1]; psi = curr_state[2]
		theta_ind = convert_to_index(theta, -pi, pi, N_BINS)
		phi_ind = convert_to_index(phi, -pi, pi, N_BINS)
		psi_ind = convert_to_index(psi, -pi, pi, N_BINS)

		action_ind = np.argmax(q_table[theta_ind, phi_ind, psi_ind, :])
		#print(max(q_table[theta_ind, phi_ind, psi_ind, :]))
		phidot, psidot = get_action_from_index(action_ind, A_LOWER, A_UPPER, N_BINS, N_BINS)

		action = (phidot, psidot)
		robot.move(action)

		curr_state = robot.state

		x_pos.append(robot.x)
		y_pos.append(robot.y)
		thetas.append(robot.theta)
		phis.append(robot.phi)
		psis.append(robot.psi)
		times.append(i)

		robot_param = [robot.x,
					   robot.y,
					   robot.theta,
					   float(robot.phi),
					   float(robot.psi),
					   robot.phidot,
					   robot.psidot]
		robot_params.append(robot_param)
		
	if(save_csv):
		generate_csv(robot_params, "./results/wheelchair_results/" + "qlearn_result_test.csv")

	return x_pos, y_pos, thetas, phis, psis, times

def plot_trajectories(x_pos, y_pos, thetas, phis, psis, times):

	plt.plot(x_pos, y_pos)
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('y vs x')
	plt.show()

	plt.plot(times, phis)
	plt.ylabel('phi displacements')
	plt.show()

	plt.plot(times, psis)
	plt.ylabel('psi displacements')
	plt.show()

	plt.plot(times, x_pos)
	plt.ylabel('x positions')
	plt.show()

	plt.plot(times, y_pos)
	plt.ylabel('y positions')
	plt.show()

	plt.plot(times, thetas)
	plt.ylabel('thetas')
	plt.show()
	plt.close()


def plot_heat_map(table):
	thetas = np.arange(N_BINS)
	phis = np.arange(N_BINS)
	psis = np.arange(N_BINS)

	theta_phi_heat = np.zeros((N_BINS, N_BINS))
	theta_psi_heat = np.zeros((N_BINS, N_BINS))
	phi_psi_heat = np.zeros((N_BINS, N_BINS))

	for i in range(len(thetas)):
		for j in range(len(phis)):
			theta_phi_heat[i, j] = np.max(table[i, j, :, :])

	for i in range(len(thetas)):
		for j in range(len(phis)):
			theta_psi_heat[i, j] = np.max(table[i, :, j, :])

	for i in range(len(phis)):
		for j in range(len(psis)):
			phi_psi_heat[i, j] = np.max(table[:, i, j, :])

	#plt.imshow(theta_phi_heat, cmap='inferno')
	print("theta_phi sym_score", get_sym_score_theta(theta_phi_heat, N_BINS))
	plt.matshow(theta_phi_heat)
	plt.colorbar()
	plt.show()

	#plt.imshow(theta_psi_heat, cmap='inferno')
	print("theta_psi sym_score", get_sym_score_theta(theta_psi_heat, N_BINS))
	plt.matshow(theta_psi_heat)
	plt.colorbar()
	plt.show()

	plt.matshow(phi_psi_heat)
	plt.colorbar()
	plt.show()



table = construct_table()
init_score = get_policy_score(table)
robot = WheelChairRobot(t_interval = 1)

q_learn(table, robot)

final_score = get_policy_score(table)

print("initial score", init_score)
print("final score", final_score)


x_pos, y_pos, thetas, phis, psis, times = locomote(robot, table, save_csv = False)

plot_trajectories(x_pos, y_pos, thetas, phis, psis, times)
print(x_pos)
plot_heat_map(table)





