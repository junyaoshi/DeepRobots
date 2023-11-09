import Shared
from DQN_Homomorphism_KNN_enhanced import DQNAgent, DEVICE
import threading
import copy

def define_parameters():
	params = Shared.parameters()
	params['memory_size_for_abstraction'] = params['memory_size']
	params['batch_size_for_abstraction'] = 16
	params['abstraction_learning_rate'] = 0.001
	params['abstract_state_space_dimmension'] = 2
	params['K_for_KNN'] = 11
	params['symmetry_weight'] = 1.0
	params['abstract_state_holders_size'] = params['memory_size']
	params['reward_filter'] = True

	# not implemented yet
	params['plot_t-sne'] = False
	params['t-sne_next_state'] = True
	params['plot_reward_fixations'] = False
	return params

params = define_parameters()
action = params['action_bins']
if params['plot_t-sne'] == True:	
	Shared.run(params, DQNAgent, f'symmetry_result/t-sne')

# previous(3,7) ones are 16 batch size. current(5,11) ones are 32 batch size
threads = []
for k in [13]:
	for reward_filter, weight in [(False,0.5), (True,1.0)]:
		params['K_for_KNN'] = k
		params['symmetry_weight'] = weight
		params['reward_filter'] = reward_filter
		thread = threading.Thread(target=Shared.run, args=(copy.deepcopy(params), DQNAgent, f'symmetry_result/{action}_symmetry({k})-{weight},filter({reward_filter})'))
		thread.start()
		threads.append(thread)

for thread in threads:
	thread.join()