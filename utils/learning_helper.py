from math import pi
import numpy as np
import csv


def save_learning_data(path, num_episodes, avg_rewards, std_rewards, avg_losses, std_losses, avg_Qs, std_Qs):
    """
    saving learning results to csv
    """
    rows = zip(num_episodes, avg_rewards, std_rewards, avg_losses, std_losses, avg_Qs, std_Qs)
    with open(path + '/learning_data.csv', 'w') as f:
        w = csv.writer(f)
        w.writerows(rows)

# Perform DQN
def get_random_edge_states(robot):
    num = np.random.rand()
    theta = robot.theta
    if num < 0.2:
        pass
    elif num < 0.4:
        print('edge case 1!')
        robot.set_state(theta=theta, a1=-pi/2, a2=pi/2)
    elif num < 0.6:
        print('edge case 2!')
        robot.set_state(theta=theta, a1=pi/2, a2=pi/2)
    elif num < 0.8:
        print('edge case 3!')
        robot.set_state(theta=theta, a1=pi/2, a2=-pi/2)
    else:
        print('edge case 4')
        robot.set_state(theta=theta, a1=-pi/2, a2=-pi/2)
    return robot

