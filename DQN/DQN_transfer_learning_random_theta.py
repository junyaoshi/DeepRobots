from DQN_swimming import *
import os, sys

# Edit the system path as needed
sys.path.append('/home/jackshi/DeepRobots')

json_name = '/home/jackshi/DeepRobots/Trials/DQN_Swimming_Trial_16/30000 th episode model.json'
h5_name = '/home/jackshi/DeepRobots/Trials/DQN_Swimming_Trial_16/30000 th episode weights.h5'

# transfer agent
agent = DQNAgent(gamma=0.98, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.9999994,
                 memory_size=20000, actions_params=(-pi/8, pi/8, pi/8), learning_rate=0.001)
agent.load_model(json_name=json_name, h5_name=h5_name)

# specify program information
TIMESTAMP = str(datetime.datetime.now()).replace(' ', '_')[:-7]
TRIAL_NAME = 'DQN_Swimming_Transfer_Random_Theta'
TRIAL_NUM = 19
PATH = 'Trials/' + TRIAL_NAME + '_Trial_' + str(TRIAL_NUM) + "_" + TIMESTAMP

# create directory
os.mkdir(PATH)
os.chmod(PATH, 0o0777)

# # Perform DQN
# learning_results = perform_DQN(agent, episodes=2, iterations=100,
#                                batch_size=8, C=200, randomize_theta=True, path=PATH)
# agent, num_episodes, avg_rewards, std_rewards, avg_losses, std_losses = learning_results
#
# # Loss Plot
# make_loss_plot(num_episodes, avg_losses, std_losses, path=PATH)
#
# # Learning Curve Plot
# make_learning_plot(num_episodes, avg_rewards, std_rewards, path=PATH)

# Policy Rollout
policy_rollout(agent=agent, path=PATH)



