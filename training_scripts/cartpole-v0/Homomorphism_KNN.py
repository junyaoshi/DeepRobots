import Shared
from DQN_Homomorphism_KNN import DQNAgent, DEVICE

def define_parameters():
	params = Shared.parameters()
	params['memory_size_for_abstraction'] = params['episode_length'] * 3
	params['batch_size_for_abstraction'] = 4
	params['abstraction_learning_rate'] = 0.001
	params['negative_samples_size'] = 1
	params['hinge'] = 1.0
	params['abstract_state_space_dimmension'] = 2
	params['K_for_KNN'] = 3
	params['symmetry_weight'] = 0.6
	params['exploit_symmetry'] = False
	params['t-sne_next_state'] = False
	return params

params = define_parameters()
rewards, episodes = Shared.run(params, DQNAgent)
Shared.plot('Homomorphism_KNN(10 runs)', 'total rewards', 'episodes', rewards, episodes, 'DQN_symmetry.csv')