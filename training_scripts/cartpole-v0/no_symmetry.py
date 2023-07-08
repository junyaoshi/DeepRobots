import Shared
from DQN import DQNAgent, DEVICE

params = Shared.parameters()
rewards, episodes = Shared.run(params, DQNAgent)
Shared.plot('no symmetry', 'total rewards', 'episodes', rewards, episodes, 'DQN_no_symmetry.csv')