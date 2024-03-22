import Shared
from DQN import DQNAgent, DEVICE

params = Shared.parameters()
action = params['action_bins']
rewards = Shared.run(params, DQNAgent, f"symmetry_result/{action}_no_symmetry_reduced_no_position")