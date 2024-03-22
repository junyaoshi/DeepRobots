import Shared
from DQN_Homomorphism_KNN_enhanced import DQNAgent, DEVICE
import threading
import copy
import wandb

def define_parameters():
	params = Shared.parameters()
	params['memory_size_for_abstraction'] = params['memory_size']
	params['batch_size_for_abstraction'] = 64
	params['abstraction_learning_rate'] = 0.0005
	params['abstract_state_space_dimmension'] = 50
	params['K_for_KNN'] = 9
	params['symmetry_weight'] = 1.0
	params['abstract_state_holders_size'] = params['memory_size']
	params['reward_filter'] = False
	params['abstract_KNN_interval'] = 3
	params['equivalent_exploitation_beginning_episode'] = 50

	# not implemented yet
	params['plot_t-sne'] = False
	params['t-sne_next_state'] = True
	params['plot_reward_fixations'] = False
	params['t-sne-interval'] = 350
	return params

params = define_parameters()
action = params['action_bins']
if params['plot_t-sne'] == True:
	wandb.init(
		project="debugging_mode",
	)
	Shared.run(params, DQNAgent, f'symmetry_result/t-sne')
	wandb.finish()

# previous(3,7) ones are 16 batch size. current(5,11) ones are 32 batch size
threads = []
for k in [13]:
	for reward_filter, weight in [(True,1.0)]:
	#for reward_filter, weight in [(False,1.0)]:
	#for reward_filter, weight in [(False,0.6), (True,1.0)]:
		params['K_for_KNN'] = k	
		params['symmetry_weight'] = weight
		params['reward_filter'] = reward_filter
		id = f'symmetry_result/{action}_symmetry({k})-{weight},filter({reward_filter})_mature_50'
		wandb.init(
			project=id.split('/')[1],
		)
		thread = threading.Thread(target=Shared.run, args=(copy.deepcopy(params), DQNAgent, id))
		thread.start()
		threads.append(thread)

for thread in threads:
	thread.join()
wandb.finish()