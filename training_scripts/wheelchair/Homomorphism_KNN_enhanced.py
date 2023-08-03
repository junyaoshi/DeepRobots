import Shared
from DQN_Homomorphism_KNN_enhanced import DQNAgent, DEVICE

def define_parameters():
	params = Shared.parameters()
	params['memory_size_for_abstraction'] = params['episode_length'] * 3
	params['batch_size_for_abstraction'] = 8
	params['abstraction_learning_rate'] = 0.001
	params['hinge'] = 0.05
	params['abstract_state_space_dimmension'] = 2
	params['K_for_KNN'] = 3
	params['symmetry_weight'] = 0.4
	params['exploit_symmetry'] = True
	params['abstract_state_holders_size'] = params['memory_size']
	params['average_reward'] = False

	# not implemented yet
	params['plot_t-sne'] = True
	params['t-sne_next_state'] = True
	params['plot_reward_fixations'] = True
	return params

params = define_parameters()
rewards, episodes = Shared.run(params, DQNAgent)
average_reward = params['average_reward']
Shared.plot(f'wheel hom_knn_enh. a_r:{average_reward}' + Shared.title(params), 'total rewards', 'episodes', rewards, episodes, 'DQN_symmetry.csv')