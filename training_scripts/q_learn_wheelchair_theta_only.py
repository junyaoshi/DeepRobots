import numpy as np
from math import cos, sin, pi
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

from Robots.WheelChair_v1 import WheelChairRobot

from utils.csv_generator import generate_csv

#========Experiment Params========

#Discretization Factor
N_BINS = 30

#Action input limits
A_LOWER = -1; A_UPPER = 1

#Number of experiments 
N_SEEDS = 1

#Number of Iterations
ITR = 100000

#Save rollout data?
SAVE_CSV = False

#Generate trajectory and q-val plots?
GEN_PLOTS = True

#dump x pos data and intermediate training info to console?
VERBOSE = True

TRIAL_NAME = "euc"

#======================

SEEDS = 10 * np.arange(N_SEEDS) 


def construct_table(
	theta_bins = N_BINS,
	phidot_bins = N_BINS,
	psidot_bins = N_BINS):
	
	q_table = np.zeros(
		(theta_bins,
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

def get_phipsi_inds(action_ind, phidot_bins):
	phidot_ind = action_ind//phidot_bins
	psidot_ind =  action_ind % phidot_bins
	return phidot_ind, psidot_ind

def get_policy_score(q_table):
	tot = 0
	
	for theta_ind in range(N_BINS):
		theta = get_val_from_index(theta_ind, -pi, pi, N_BINS)

		robot = WheelChairRobot(theta=theta, t_interval = 1)
		
		curr_theta_ind = theta_ind
		
		for i in range(10):
			action_ind = np.argmax(q_table[curr_theta_ind, :])
			phidot_true, psidot_true = get_action_from_index(action_ind, A_LOWER, A_UPPER, N_BINS, N_BINS)
			action = (phidot_true, psidot_true)
			robot.move(action)
			curr_theta_ind = convert_to_index(robot.theta, -pi, pi, N_BINS)

		tot += robot.x

	return tot/N_BINS

def get_sym_score_theta(mat, theta_bins):
	pass


def e_greedy(eps, theta, q_table):
	a_vals = q_table[theta, :]

	max_val = np.max(a_vals)
	
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


def get_opp_theta_ind(theta_ind, n_bins):
	opp_theta_ind = None
	if(theta_ind == 0 or theta_ind == n_bins//2):
		opp_theta_ind = theta_ind
	else:
		mid = n_bins//2
		if(theta_ind > mid):
			diff = theta_ind - mid
			opp_theta_ind = mid - diff
		else:
			diff = mid - theta_ind
			opp_theta_ind = mid + diff
	
	return opp_theta_ind


#Assumes state is initialized as all 0
def q_learn(
	q_table, 
	robot, 
	itr = 1000000, 
	gamma = 0.8, 
	eps = 0.8, 
	alpha = 0.9, 
	use_sym = False,
	use_euc = False
	):
	
	#Initialize action (starting state assumed to be 0)
	policy_scores = []
	
	for i in range(itr):
		
		if(i % 10000 == 0 and i != 0 and VERBOSE):
			print("iteration:", i)
			print("policy score:", get_policy_score(q_table))

		if(i % 500 == 0):
			policy_scores.append(get_policy_score(q_table))
		
		#Update s to current config (equivalent to s <- s' step)
		curr_state = robot.state
		theta = curr_state[0] #s
		theta_ind = convert_to_index(theta, -pi, pi, N_BINS)
		
		#Get curr_x for reward calculation
		curr_x = robot.x

		#Get curr_y for euclidean distance reward alculation 
		curr_y = robot.y

		#Move to s'
		action_ind, phidot_val, psidot_val = e_greedy(eps, theta_ind, q_table)
		action = (phidot_val, psidot_val)
		robot.move(action)

		#Do reward calculation
		new_x = robot.x
		new_y = robot.y

		reward = new_x - curr_x

		if use_euc:
			reward = np.sqrt( (new_x - curr_x)**2 + (new_y - curr_y)**2 )

		#Get s'
		new_state = robot.state
		theta_p = new_state[0]
		theta_p_ind = convert_to_index(theta_p, -pi, pi, N_BINS)

		#Find Q(s, a)
		q_s_a = q_table[theta_ind, action_ind]

		#From s', find max a' Q(s', a')
		q_s_a_p = np.max(q_table[theta_p_ind, :])

		#Update Q-value
		q_table[theta_ind, action_ind] = q_s_a + alpha * (reward + gamma * q_s_a_p - q_s_a)

		if use_sym:
			opp_theta_ind = get_opp_theta_ind(theta_ind, N_BINS)
			if opp_theta_ind != theta_ind:
				phidot_ind, psidot_ind = get_phipsi_inds(action_ind, N_BINS)
				rev_action_ind = psidot_ind * N_BINS + phidot_ind
				q_table[opp_theta_ind, rev_action_ind] = q_table[theta_ind, action_ind]

	return policy_scores


def locomote(
	q_table, 
	theta = 0, 
	itr = 100, 
	save_csv = True,
	csv_name = None
	):
	robot = WheelChairRobot(theta=theta, t_interval = 1)

	curr_state = (theta, 0, 0)

	x_pos = [0]
	y_pos = [0]
	thetas = [theta]
	phis = [0]
	psis = [0]
	times = [0]

	robot_params = []

	for i in range(itr):
		theta = curr_state[0]
		theta_ind = convert_to_index(theta, -pi, pi, N_BINS)

		action_ind = np.argmax(q_table[theta_ind, :])
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
		
	if(save_csv and csv_name != None):
		generate_csv(robot_params, "./results/wheelchair_results/" + "qlearn_result_theta_" + str(csv_name) + ".csv")

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


def plot_theta_q_vals(table):
	theta_ind = []
	theta_q_vals = []

	for i in range(N_BINS):
		theta_ind.append(i)
		theta_q_vals.append(np.max(table[i, :]))


	plt.plot(theta_ind, theta_q_vals)
	plt.xlabel('theta ind')
	plt.ylabel('max q val')
	plt.show()


figure = None; axis = None
if N_SEEDS > 1:
	figure, axis = plt.subplots(2, N_SEEDS//2, figsize = (15, 15))
	figure.tight_layout(pad = 4.0)

for i in range(N_SEEDS):
	seed = SEEDS[i]

	#Q-learning without symmetry boost 

	np.random.seed(seed)

	table = construct_table()
	init_score = get_policy_score(table)
	robot = WheelChairRobot(t_interval = 1)

	policy_scores = q_learn(table, robot, itr = ITR, use_sym = False)

	final_score = get_policy_score(table)

	print("initial score", init_score)
	print("final score", final_score)

	csv_name = "no_sym_" + TRIAL_NAME
	x_pos, y_pos, thetas, phis, psis, times = locomote(table, theta = 0, save_csv = SAVE_CSV, csv_name = csv_name)

	if VERBOSE:
		print(x_pos)

	if GEN_PLOTS:
		plot_trajectories(x_pos, y_pos, thetas, phis, psis, times)
		plot_theta_q_vals(table)

	##############################################################################

	#Q-learning with symmetry boost

	np.random.seed(seed)

	table = construct_table()
	init_score = get_policy_score(table)
	robot = WheelChairRobot(t_interval = 1)

	policy_scores2 = q_learn(table, robot, itr = ITR, use_sym = True)

	final_score = get_policy_score(table)

	print("initial score", init_score)
	print("final score", final_score)

	csv_name = "yes_sym_" + TRIAL_NAME
	x_pos, y_pos, thetas, phis, psis, times = locomote(table, theta = 0, save_csv = SAVE_CSV, csv_name = csv_name)

	if VERBOSE:
		print(x_pos)

	if GEN_PLOTS:
		plot_trajectories(x_pos, y_pos, thetas, phis, psis, times)
		plot_theta_q_vals(table)

	##############################################################################

	#Update policy score graphs

	x_vals = np.arange(len(policy_scores))
	
	if N_SEEDS > 1:
		col = i % (N_SEEDS//2)
		row = 0
		if(i >= N_SEEDS//2):
			row = 1

		axis[row, col].plot(x_vals, policy_scores, label = "no sym")
		axis[row, col].plot(x_vals, policy_scores2, label = "yes sym")
		axis[row, col].legend(loc="lower right")
	
	else:
		plt.plot(x_vals, policy_scores, label = "no sym")
		plt.plot(x_vals, policy_scores2, label = "yes sym")
		plt.legend(loc="lower right")
		plt.show()


if N_SEEDS > 1:
	plt.show()






