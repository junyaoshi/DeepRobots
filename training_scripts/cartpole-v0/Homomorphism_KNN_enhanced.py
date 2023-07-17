import Shared
from DQN_Homomorphism_KNN_enhanced import DQNAgent, DEVICE

def define_parameters():
	params = Shared.parameters()
	params['memory_size_for_abstraction'] = params['episode_length'] * 3
	params['batch_size_for_abstraction'] = 4
	params['abstraction_learning_rate'] = 0.001
	params['negative_samples_size'] = 2
	params['hinge'] = 0.05
	params['abstract_state_space_dimmension'] = 50
	params['K_for_KNN'] = 3
	params['symmetry_weight'] = 0.4
	params['exploit_symmetry'] = True
	params['abstract_state_holders_size'] = params['memory_size']
	
	params['plot_t-sne'] = False
	params['t-sne_next_state'] = False
	params['plot_reward_fixations'] = True
	return params

params = define_parameters()
rewards, episodes = Shared.run(params, DQNAgent)
weight = params['symmetry_weight']
dimension = params['abstract_state_space_dimmension']
Shared.plot(f'hom_knn_enh. weight, dimension:{weight, dimension}' + Shared.title(params), 'total rewards', 'episodes', rewards, episodes, 'DQN_symmetry.csv')