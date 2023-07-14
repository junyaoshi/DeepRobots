import Shared
from DQN import DQNAgent, DEVICE

params = Shared.parameters()
rewards, episodes = Shared.run(params, DQNAgent)
runtime = params['run_times_for_performance_average']
Shared.plot('no symmetry ' + Shared.title(params), 'total rewards', 'episodes', rewards, episodes, 'DQN_no_symmetry.csv')