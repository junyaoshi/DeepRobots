import Shared
from DQN_Homomorphism_KNN import DQNAgent, DEVICE

def define_parameters():
	params = Shared.parameters()
	params['memory_size_for_abstraction']
	params['batch_size_for_abstraction']
	params['n_neg_samples']
	params['hinge']
	params['abstract_state_space_dimmension']
	return params

params = define_parameters()
rewards, episodes = Shared.run(params, DQNAgent)
Shared.plot('Homomorphism_KNN', 'total rewards', 'episodes', rewards, episodes, 'DQN_symmetry.csv')