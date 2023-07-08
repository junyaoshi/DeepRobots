import Shared

# need to import gymnasium. pip install gymnaisum
# pip install "gymnasium[all]"
import gymnasium as gym

from DQN_symmetry import DQNAgent, DEVICE

def define_parameters():
	params = Shared.parameters()
	# Neural Network
	params['first_layer_size'] = 100    # neurons in the first layer
	params['second_layer_size'] = 100   # neurons in the second layer
	params['epsilon'] = 1.0
	params['epsilon_decay'] = 0.98
	params['epsilon_minimum'] = 0.1
	params['episodes_with_random_policy'] = 25

	#reward trailing
	params['reward_trail_length'] = 5
	params['reward_trail_symmetry_threshold'] = 0.8
	params['reward_trail_symmetry_weight'] = 1.0
	return params

params = define_parameters()
rewards, episodes = Shared.run(params, DQNAgent)
Shared.plot('symmetry', 'total rewards', 'episodes', rewards, episodes, 'DQN_symmetry.csv')