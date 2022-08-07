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

#X and Y limits

XY_LOWER = -60; XY_UPPER = 60

#Number of experiments 
N_SEEDS = 1 #TODO: support for > 1 experiments 

#Number of Iterations
ITR = 10000

print("NUMBER OF ITERATIONS: ", str(ITR))

#Save rollout data?
SAVE_CSV = False

#Generate trajectory and q-val plots?
GEN_PLOTS = False

#dump x pos data and intermediate training info to console?
VERBOSE = True

TRIAL_NAME = "100k_2" #Only really matters if SAVE_CSV is True

ITR_SCALE = 500

MAX_ROLLOUT_STEPS = 10

#======================

SEEDS = 10 * np.arange(N_SEEDS) 


def construct_table(
	theta_bins = N_BINS,
	x_bins = N_BINS,
	y_bins = N_BINS,
	phidot_bins = N_BINS,
	psidot_bins = N_BINS):
	
	q_table = np.zeros(
		(theta_bins,
		x_bins,
		y_bins,
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

#Need modification 
def get_policy_score(q_table, use_thresh = False, thresh_val = 23, steps = 3):
	tot = 0

	num_over_thresh = 0


	print("getting policy score...")
	
	for x_ind in range(N_BINS):
		for y_ind in range(N_BINS):
			for theta_ind in range(N_BINS):
				theta = get_val_from_index(theta_ind, -pi, pi, N_BINS)

				robot = WheelChairRobot(theta=theta, t_interval = 1)
				
				curr_theta_ind = theta_ind
				curr_x_ind = x_ind
				curr_y_ind = y_ind
				
				for i in range(steps):
					action_ind = np.argmax(q_table[curr_theta_ind, curr_x_ind, curr_y_ind, :])
					phidot_true, psidot_true = get_action_from_index(action_ind, A_LOWER, A_UPPER, N_BINS, N_BINS)
					action = (phidot_true, psidot_true)
					robot.move(action)

					curr_theta_ind = convert_to_index(robot.theta, -pi, pi, N_BINS)
					curr_x_ind = convert_to_index(robot.x, XY_LOWER, XY_UPPER, N_BINS)
					curr_y_ind = convert_to_index(robot.y, XY_LOWER, XY_UPPER, N_BINS)

				tot += robot.x

				if robot.x >= thresh_val:
					num_over_thresh += 1


	ret = tot/(N_BINS * N_BINS * N_BINS)

	if use_thresh:
		ret = num_over_thresh/(N_BINS * N_BINS * N_BINS)


	return ret

def get_sym_score_theta(mat, theta_bins):
	pass


#Need modification
def e_greedy(eps, theta, x, y, q_table):
	a_vals = q_table[theta, x, y, :]

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

def get_opp_angle_dot_ind(angle_ind, n_bins):
	opp_angle_ind = None
	if(angle_ind == n_bins//2):
		opp_angle_ind = angle_ind
	elif angle_ind == 0:
		opp_angle_ind = n_bins - 1
	else:
		mid = n_bins//2
		if(angle_ind > mid):
			diff = angle_ind - mid
			opp_angle_ind = mid - diff
		else:
			diff = mid - angle_ind
			opp_angle_ind = mid + diff
	
	return opp_angle_ind

def get_opp_dot_inds(action_ind, n_bins):
	phidot_ind, psidot_ind = get_phipsi_inds(action_ind, n_bins)
	#print("old phidot:", get_val_from_index(phidot_ind, -1, 1, 30))
	#print("old psidot:", get_val_from_index(psidot_ind, -1, 1, 30))
	phidot_ind = get_opp_angle_dot_ind(phidot_ind, n_bins)
	psidot_ind = get_opp_angle_dot_ind(psidot_ind, n_bins)

	return phidot_ind, psidot_ind

def get_lr_sym(theta, action_ind, n_bins):
	#New theta +/- pi (whatever keeps it in range)
	new_theta = None
	if(theta >= 0):
		new_theta = theta - pi
	else:
		new_theta = theta + pi

	new_theta_ind = convert_to_index(new_theta, -pi, pi, n_bins)

	#print("old theta:", theta)
	

	#Actions are negated
	new_phidot_ind, new_psidot_ind = get_opp_dot_inds(action_ind, n_bins)

	#print("new phidot:", get_val_from_index(new_phidot_ind, -1, 1, 30))
	#print("new psidot:", get_val_from_index(new_psidot_ind, -1, 1, 30))
	#print("new theta:", get_val_from_index(new_theta_ind, -pi, pi, 30))

	new_action_ind = new_psidot_ind * n_bins + new_phidot_ind 

	return new_theta_ind, new_action_ind


#Need modification
#Assumes state is initialized as all 0
def q_learn(
	q_table, 
	robot, 
	itr = 1000000, 
	gamma = 0.75, 
	eps = 0.8, 
	alpha = 0.9, 
	use_x_sym = False,
	use_y_sym = False,
	use_both = False,
	use_euc = False
	):
	
	#Initialize action (starting state assumed to be 0)
	policy_scores = []
	
	for i in range(itr):
		
		if(i % 10000 == 0 and i != 0 and VERBOSE):
			print("iteration:", i)
			print("policy score:", get_policy_score(q_table))

		if(i % ITR_SCALE == 0):
			temp = []
			for step_num in range(1, MAX_ROLLOUT_STEPS + 1):
				#print(step_num)
				temp.append(get_policy_score(q_table, steps = step_num))
			policy_scores.append(temp)

		#Get curr_x for reward calculation
		curr_x = robot.x

		#Get curr_y for euclidean distance reward alculation 
		curr_y = robot.y
		
		#Update s to current config (equivalent to s <- s' step)
		curr_state = robot.state
		theta = curr_state[0] #s
		theta_ind = convert_to_index(theta, -pi, pi, N_BINS)
		x_ind = convert_to_index(curr_x, XY_LOWER, XY_UPPER, N_BINS)
		y_ind = convert_to_index(curr_y, XY_LOWER, XY_UPPER, N_BINS)

		#Move to s'
		action_ind, phidot_val, psidot_val = e_greedy(eps, theta_ind, x_ind, y_ind, q_table)
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
		x_p_ind = convert_to_index(new_x, XY_LOWER, XY_UPPER, N_BINS)
		y_p_ind = convert_to_index(new_y, XY_LOWER, XY_UPPER, N_BINS)

		#Find Q(s, a)
		q_s_a = q_table[theta_ind, x_ind, y_ind, action_ind]

		#From s', find max a' Q(s', a')
		q_s_a_p = np.max(q_table[theta_p_ind, x_p_ind, y_p_ind, :])

		#Update Q-value
		q_table[theta_ind, x_ind, y_ind, action_ind] = q_s_a + alpha * (reward + gamma * q_s_a_p - q_s_a)

		if use_x_sym:
			q_table[theta_ind, :, y_ind, action_ind] = q_table[theta_ind, x_ind, y_ind, action_ind]

		if use_y_sym:
			q_table[theta_ind, x_ind,:, action_ind] = q_table[theta_ind, x_ind, y_ind, action_ind]

		if use_both:
			q_table[theta_ind, :, :, action_ind] = q_table[theta_ind, x_ind, y_ind, action_ind]

		#Since x and y are bounded, it's possible during training that the robot will significantly exceed the bounds
		#So we reset if this happens
		if robot.x > XY_UPPER or robot.x < XY_LOWER:
			robot.reset_x()

		if robot.y > XY_UPPER or robot.y < XY_LOWER:
			robot.reset_y()


	return policy_scores

#Need modification
def locomote(
	q_table, 
	theta = 0, 
	itr = 12, 
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
		x_ind = convert_to_index(robot.x, XY_LOWER, XY_UPPER, N_BINS)
		y_ind = convert_to_index(robot.y, XY_LOWER, XY_UPPER, N_BINS)

		action_ind = np.argmax(q_table[theta_ind, x_ind, y_ind, :])
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

#Need modification
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


scores = []
scores_lr = []
scores_ud = []
scores_both = []

for i in range(N_SEEDS):
	seed = SEEDS[i]

	#Q-learning without symmetry boost 

	np.random.seed(seed)

	table = construct_table()
	init_score = get_policy_score(table)
	robot = WheelChairRobot(t_interval = 1)

	policy_scores = q_learn(table, robot, itr = ITR, use_x_sym = False, use_y_sym = False)
	scores.append(policy_scores)

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

	#Q-learning with x symmetry boost

	np.random.seed(seed)

	table = construct_table()
	init_score = get_policy_score(table)
	robot = WheelChairRobot(t_interval = 1)

	policy_scores_lr = q_learn(table, robot, itr = ITR, use_x_sym = True, use_y_sym = False)
	scores_lr.append(policy_scores_lr)

	final_score = get_policy_score(table)

	print("initial score", init_score)
	print("final score", final_score)

	csv_name = "lr_sym_" + TRIAL_NAME
	x_pos, y_pos, thetas, phis, psis, times = locomote(table, theta = 0, save_csv = SAVE_CSV, csv_name = csv_name)

	if VERBOSE:
		print(x_pos)

	if GEN_PLOTS:
		plot_trajectories(x_pos, y_pos, thetas, phis, psis, times)
		plot_theta_q_vals(table)


	##############################################################################

	#Q-learning with y symmetry boost

	np.random.seed(seed)

	table = construct_table()
	init_score = get_policy_score(table)
	robot = WheelChairRobot(t_interval = 1)

	policy_scores_ud = q_learn(table, robot, itr = ITR, use_x_sym = False, use_y_sym = True)
	scores_ud.append(policy_scores_ud)

	final_score = get_policy_score(table)

	print("initial score", init_score)
	print("final score", final_score)

	csv_name = "ud_sym_" + TRIAL_NAME
	x_pos, y_pos, thetas, phis, psis, times = locomote(table, theta = 0, save_csv = SAVE_CSV, csv_name = csv_name)

	if VERBOSE:
		print(x_pos)

	if GEN_PLOTS:
		plot_trajectories(x_pos, y_pos, thetas, phis, psis, times)
		plot_theta_q_vals(table)


	##############################################################################

	#Q-learning with both symmetry boost

	np.random.seed(seed)

	table = construct_table()
	init_score = get_policy_score(table)
	robot = WheelChairRobot(t_interval = 1)

	policy_scores_both = q_learn(table, robot, itr = ITR, use_both = True)
	scores_both.append(policy_scores_both)

	final_score = get_policy_score(table)

	print("initial score", init_score)
	print("final score", final_score)

	csv_name = "both_sym_" + TRIAL_NAME
	x_pos, y_pos, thetas, phis, psis, times = locomote(table, theta = 0, save_csv = SAVE_CSV, csv_name = csv_name)

	if VERBOSE:
		print(x_pos)

	if GEN_PLOTS:
		plot_trajectories(x_pos, y_pos, thetas, phis, psis, times)
		plot_theta_q_vals(table)

##############################################################################

#Make policy score graphs

x_vals = ITR_SCALE * np.arange(len(policy_scores))

if N_SEEDS > 1:
	figure, axis = plt.subplots(2, N_SEEDS//2, figsize = (15, 15))
	figure.tight_layout(pad = 4.0)

	for i in range(N_SEEDS):
		col = i % (N_SEEDS//2)
		row = 0
		if(i >= N_SEEDS//2):
			row = 1

		axis[row, col].plot(x_vals, scores[i], label = "no sym")
		axis[row, col].plot(x_vals, scores_lr[i], label = "x_sym")
		axis[row, col].plot(x_vals, scores_ud[i], label = "y_sym")
		axis[row, col].plot(x_vals, scores_both[i], label = "both_sym")
		axis[row, col].set_xlabel("iterations")
		axis[row, col].set_ylabel("score")
		axis[row, col].legend(loc="lower right")

else:
	x_vals = np.arange(1, MAX_ROLLOUT_STEPS + 1)
	iters = ITR_SCALE * np.arange(len(policy_scores))
	for i in range(len(iters)):
		plt.plot(x_vals, scores[0][i], label = "no sym")
		plt.plot(x_vals, scores_lr[0][i], label = "x_sym")
		plt.plot(x_vals, scores_ud[0][i], label = "y_sym")
		plt.plot(x_vals, scores_both[0][i], label = "both_sym")
		plt.xlabel("num rollout steps")
		plt.ylabel("score")
		plt.legend(loc="lower right")
		plt.title("score vs rollout steps at " + str(i * ITR_SCALE) + " iterations")
		plt.savefig("./training_scripts/movie_files/" + "img_" + str(i))
		plt.close()


#plt.show()


# if N_SEEDS > 1:
# 	plt.show()






