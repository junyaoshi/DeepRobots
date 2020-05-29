import sys
import json

# Edit the system path as needed
sys.path.append('/home/jackshi/DeepRobots')

from DQN.DQN_agent_v2 import DQN_Agent
from Robots.PhysicalRobot import PhysicalRobot
from utils.learning_helper import physical_forward_reward_function

trial_dir = 'Trial_70_DQN_swimming_forward_12500_iters_updated_nn_2020-05-22-11-37'
param_fname = trial_dir + 'params.json'
model_json_fname = trial_dir + '50_th_episode_model.json'
model_h5_fname = trial_dir + '50_th_episode_weights.h5'

# load param file
with open(param_fname) as param_file:
    params = json.load(param_file)

# transfer agent
robot = PhysicalRobot(params['delay'])
a_lower = params['a_lower']
a_upper = params['a_upper']
a_interval = params['a_interval']
action_params = (a_lower, a_upper, a_interval)
episodes = params['episodes']
iterations = params['iterations']
total_iterations = episodes * iterations
dqn_agent = DQN_Agent(robot=robot,
                      reward_function=physical_forward_reward_function,
                      trial_name="transfer_learning_test",
                      trial_num=0,
                      episodes=params['episodes'],
                      iterations=params['iterations'],
                      network_update_freq=params['network_update_freq'],
                      check_singularity=False,
                      is_physical_robot=True,
                      actions_params=action_params,
                      model_architecture=params['model_architecture'],
                      memory_size=total_iterations // 3,  # 10
                      memory_buffer_coef=20,  # 20
                      randomize_theta=False,
                      batch_size=params['batch_size'],
                      gamma=0.99,
                      epsilon=1.0,
                      epsilon_min=params['epsilon_min'],
                      epsilon_decay=params['epsilon_decay'],
                      learning_rate=params['learning_rate'],
                      params=params)
dqn_agent.load_model(json_name=model_json_fname, h5_name=model_h5_fname)

# Policy Rollout
dqn_agent.policy_rollout(timesteps=50)
