import Shared
from DQN_Homomorphism_KNN import DQNAgent, DEVICE

def define_parameters():
	params = Shared.parameters()
	params['memory_size_for_abstraction'] = params['episode_length'] * 3
	params['batch_size_for_abstraction'] = 4
	params['n_neg_samples'] = 5
	params['hinge'] = 1.0
	params['abstract_state_space_dimmension'] = 50
	return params

params = define_parameters()
rewards, episodes = Shared.run(params, DQNAgent)
Shared.plot('Homomorphism_KNN', 'total rewards', 'episodes', rewards, episodes, 'DQN_symmetry.csv')