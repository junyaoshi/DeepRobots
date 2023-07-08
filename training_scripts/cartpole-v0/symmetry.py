import Shared
from DQN_symmetry import DQNAgent, DEVICE

def define_parameters():
	params = Shared.parameters()
	#reward trailing
	params['reward_trail_length'] = 5
	params['reward_trail_symmetry_threshold'] = 0.8
	params['reward_trail_symmetry_weight'] = 1.0
	return params

params = define_parameters()
rewards, episodes = Shared.run(params, DQNAgent)
Shared.plot('symmetry', 'total rewards', 'episodes', rewards, episodes, 'DQN_symmetry.csv')