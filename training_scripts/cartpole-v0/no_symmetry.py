import math
import Shared

# need to import gymnasium. pip install gymnaisum
# pip install "gymnasium[all]"
import gymnasium as gym

from DQN import DQNAgent, DEVICE

def define_parameters():
	params = Shared.parameters()
	# Neural Network
	params['first_layer_size'] = 100    # neurons in the first layer
	params['second_layer_size'] = 100   # neurons in the second layer
	params['epsilon'] = 1.0
	params['epsilon_decay'] = 0.98
	params['epsilon_minimum'] = 0.1
	params['episodes_with_random_policy'] = 25
	return params

def train_agent_and_sample_performance(agent, params, run_iteration):
	total_rewards = []
	episodes = []
	gamma = params['gamma']
	#env = gym.make("CartPole-v0",render_mode="human")
	env = gym.make("CartPole-v0")
	for i in range(params['episodes']):
		if i % 10 == 0:
			print(f'{run_iteration}th running, epidoes: {i}')
		current_state, info = env.reset()
		total_reward = 0
		is_random_policy = i < params['episodes_with_random_policy']
		for j in range(params['episode_length']):
			action = agent.select_action_index(current_state, True, is_random_policy)
			new_state, reward, terminated, truncated, info = env.step(action)
			reward += Shared.reward_add(new_state)
			total_reward += reward
			agent.remember(current_state, action, reward, new_state, terminated)
			agent.replay_mem(params['batch_size'], not is_random_policy)
			current_state = new_state
			if terminated:
				break
		total_rewards.append(total_reward)
		episodes.append(i+1)
	env.close()
	return total_rewards, episodes

params = define_parameters()
rewards = []
episodes = []
for i in range(params['run_times_for_performance_average']):
	Shared.set_seed(params['seed'])

	agent = DQNAgent(params)
	new_rewards, new_episodes = train_agent_and_sample_performance(agent, params, i)

	if len(rewards) == 0:
		rewards, episodes = new_rewards, new_episodes
	else:
		rewards = [(x + y) for x, y in zip(rewards, new_rewards)]
rewards = [x / params['run_times_for_performance_average'] for x in rewards]
Shared.plot('no symmetry(batch 8, no reward shape)', 'total rewards', 'episodes', rewards, episodes, 'DQN_no_symmetry.csv')