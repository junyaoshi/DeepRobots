import Shared
from DQN_Homomorphism_KNN_enhanced_beanie import DQNAgent, DEVICE
import threading
import copy

def define_parameters():
	params = Shared.parameters()
	params['memory_size_for_abstraction'] = params['memory_size']
	params['batch_size_for_abstraction'] = 32
	params['abstraction_learning_rate'] = 0.0005
	params['abstract_state_space_dimmension'] = 50
	params['K_for_KNN'] = 5
	params['symmetry_weight'] = 1.0
	params['abstract_state_holders_size'] = params['memory_size']
	params['reward_filter'] = False
	params['abstract_KNN_interval'] = 1
	params['equivalent_exploitation_beginning_episode'] = 30
	params['equivalent_use_mean_target'] = False
	params['preserve_reward_model'] = True

	# not implemented yet
	params['plot_t-sne'] = False
	params['t-sne_next_state'] = True
	params['plot_reward_fixations'] = False
	params['t-sne-interval'] = 50
	return params

params = define_parameters()
action = params['action_bins']
if params['plot_t-sne'] == True:	
	Shared.run(params, DQNAgent, f'wheelchair_result/t-sne')

# previous(3,7) ones are 16 batch size. current(5,11) ones are 32 batch size
threads = []
for k in [5]:
	#for reward_filter, weight in [(True,1.0)]:
	for reward_filter, weight in [(False,1.0)]:
		params['K_for_KNN'] = k
		params['symmetry_weight'] = weight
		params['reward_filter'] = reward_filter
		thread = threading.Thread(target=Shared.run, args=(copy.deepcopy(params), DQNAgent, f'wheelchair_result/{action}_symmetry({k})-{weight},filter({reward_filter})_preserve_reward_model_no_mean'))
		thread.start()
		threads.append(thread)

for thread in threads:
	thread.join()