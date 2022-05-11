import os, sys
sys.path.insert(0, os.path.abspath(".."))
import csv
import numpy as np
from math import sin, cos, pi, sqrt

from envs.IdealFluidSwimmerEnv import IdealFluidSwimmerEnv
from Robots.IdealFluidSwimmer_v3 import IdealFluidSwimmer
from stable_baselines.common.vec_env import DummyVecEnv

from stable_baselines import SAC

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

def find_min_config(model_file, step_size=6, lower=-3, upper=3):
    search_arr_1 = np.linspace(lower, 0, step_size//2, endpoint = False)
    search_arr_2 = np.linspace(upper,0, step_size//2, endpoint = False)
    zero = np.array([0])
    search_arr = np.concatenate((search_arr_1, zero, search_arr_2))

    max_x = -float('inf')
    max_a1 = 0; max_a2 = 0;
    for a1_ind in range(len(search_arr)):
        for a2_ind in range(len(search_arr)):
            env = IdealFluidSwimmerEnv(reset_theta = False)
            #env.snake_robot.a1 = search_arr[a1_ind]
            #env.snake_robot.a2 = search_arr[a2_ind]
            final_x = rollout_by_model(model_file, env, search_arr[a1_ind], search_arr[a2_ind])
            print('a1: ', search_arr[a1_ind])
            print('a2: ', search_arr[a2_ind])
            print('x: ', final_x)
            if(max_x < final_x):
                max_a1 = search_arr[a1_ind]
                max_a2 = search_arr[a2_ind]
                max_x = final_x
    return max_x, max_a1, max_a2



def rollout_by_model(model_file, env, a1, a2, do_plot = False):
    
    model = SAC.load(model_file)
    theta_init = 0
    obs_prev = env.reset_w_theta(theta_init)
    env.snake_robot.a1 = a1
    env.snake_robot.a2 = a2
    obs_prev[1] = a1
    obs_prev[2] = a2
    #obs_prev = env.reset_w_theta()
#obs_prev = env.reset()

    x_poss = [env.snake_robot.x]
    y_poss = [env.snake_robot.y]
    thetas = [env.snake_robot.theta]
    times = [0]
    a1s = [env.snake_robot.a1]
    a2s = [env.snake_robot.a2]
    a1dots = [env.snake_robot.a1dot]
    a2dots = [env.snake_robot.a2dot]
    euc_dist = [0]

    robot_params = []

    plot_style= "--bo"
    marker_size = 3

    for i in range(200):
        x_prev = env.snake_robot.x
        action, _states = model.predict(obs_prev, deterministic = True)
        obs, rewards, dones, info = env.step(action)
        x = env.snake_robot.x

        #print("Timestep: {} | State: {} | Action a1dot: {}; a2dot: {};| "
              #"Reward: {} | dX: {}".format(i, obs_prev, action[0], action[1], rewards, x - x_prev))
        obs_prev = obs
        x_poss.append(env.snake_robot.x)
        y_poss.append(env.snake_robot.y)
        thetas.append(env.snake_robot.theta)
        times.append(i)
        a1s.append(env.snake_robot.a1)
        a2s.append(env.snake_robot.a2)
        a1dots.append(env.snake_robot.a1dot)
        a2dots.append(env.snake_robot.a2dot)

        euc_dist.append( sqrt( (env.snake_robot.x)*(env.snake_robot.x) + (env.snake_robot.y)*(env.snake_robot.y)) )

        #cs.append(env.snake_robot.c)
        robot_param = [env.snake_robot.x,
                       env.snake_robot.y,
                       env.snake_robot.theta,
                       float(env.snake_robot.a1),
                       float(env.snake_robot.a2),
                       env.snake_robot.a1dot,
                       env.snake_robot.a2dot]
        robot_params.append(robot_param)

    if(do_plot == True):
        plt.plot(x_poss, y_poss, plot_style, markersize=marker_size)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('y vs x')
        plt.show()

        plt.plot(times, euc_dist, plot_style, markersize=marker_size)
        plt.xlabel('time')
        plt.ylabel('euclidean distance')
        plt.show()

        plt.plot(a1s, a2s, plot_style, markersize=marker_size)
        plt.xlabel('a1')
        plt.ylabel('a2')
        plt.title('a2 vs a1')
        plt.show()

        plt.plot(times, a1s, plot_style, markersize=marker_size)
        plt.ylabel('a1 displacements')
        plt.xlabel('time')
        plt.show()

        plt.plot(times, a2s, plot_style, markersize=marker_size)
        plt.ylabel('a2 displacements')
        plt.xlabel('time')
        plt.show()

        plt.plot(times, x_poss, plot_style, markersize=marker_size)
        plt.ylabel('x positions')
        plt.xlabel('time')
        plt.show()

        plt.plot(times, y_poss, plot_style, markersize=marker_size)
        plt.ylabel('y positions')
        plt.xlabel('time')
        plt.show()

        plt.plot(times, thetas, plot_style, markersize=marker_size)
        plt.ylabel('thetas')
        plt.xlabel('time')
        plt.show()

        plt.plot(times, a1dots, plot_style, markersize=marker_size)
        plt.ylabel('a1dot')
        plt.xlabel('time')
        plt.show()

        plt.plot(times, a2dots, plot_style, markersize=marker_size)
        plt.ylabel('a2dot')
        plt.xlabel('time')
        plt.show()

    return x_poss[-1]

def rollout_action(csv_name, yes = True):
    robot = IdealFluidSwimmer(a1=0, a2=0, a_upper = 7/8*pi, a_lower = -7/8*pi)

    env = IdealFluidSwimmerEnv(reset_theta = False)
    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor
    #vec_env = DummyVecEnv([lambda: raw_env])
    #env = vec_env.envs[0]

    env.enable_test_mode()

    a1dots_ = []
    a2dots_ = []
    x_real = []
    y_real = []
    a1_real = []
    a2_real = []
    theta_real = []
    with open(csv_name, newline='') as rollout_file:
        reader = csv.reader(rollout_file)
        for row in reader:
            a1dots_.append(float(row[5]))
            a2dots_.append(float(row[6]))
            x_real.append(row[0])
            y_real.append(row[1])
            theta_real.append(row[2])
            a1_real.append(row[3])
            a2_real.append(row[4])

    if(yes == False):
        x_pos = []
        y_pos = []
        thetas = []
        time = [0]
        a1 = []
        a2 = []
        a1dotl = []
        a2dotl = []

        x_poss = [env.snake_robot.x]
        y_poss = [env.snake_robot.y]
        thetas = [env.snake_robot.theta]
        times = [0]
        a1s = [env.snake_robot.a1]
        a2s = [env.snake_robot.a2]
        a1dots = [env.snake_robot.a1dot]
        a2dots = [env.snake_robot.a2dot]

        obs_prev = env.reset_w_theta(0)

        robot_params = []
        for i in range(200):
            x_prev = env.snake_robot.x
            action = [a1dots_[i], a2dots_[i]]
            obs, rewards, dones, info = env.step(action)
            x = env.snake_robot.x

            print("Timestep: {} | Action a1dot: {}; a2dot: {};| "
                  "Reward: {} | dX: {}".format(i, action[0], action[1], rewards, x - x_prev))
            obs_prev = obs
            x_poss.append(env.snake_robot.x)
            y_poss.append(env.snake_robot.y)
            thetas.append(env.snake_robot.theta)
            times.append(i)
            a1s.append(env.snake_robot.a1)
            a2s.append(env.snake_robot.a2)
            a1dots.append(env.snake_robot.a1dot)
            a2dots.append(env.snake_robot.a2dot)

            #cs.append(env.snake_robot.c)
            robot_param = [env.snake_robot.x,
                           env.snake_robot.y,
                           env.snake_robot.theta,
                           float(env.snake_robot.a1),
                           float(env.snake_robot.a2),
                           env.snake_robot.a1dot,
                           env.snake_robot.a2dot]
            robot_params.append(robot_param)

        plot_style = "--bo"
        marker_size = 3

        plt.plot(x_poss, y_poss, plot_style, markersize=marker_size)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('y vs x')
        plt.show()

        plt.plot(times, a1s, plot_style, markersize=marker_size)
        plt.ylabel('a1 displacements')
        plt.xlabel('time')
        plt.show()

        plt.plot(times, a2s, plot_style, markersize=marker_size)
        plt.ylabel('a2 displacements')
        plt.xlabel('time')
        plt.show()

        plt.plot(times, x_poss, plot_style, markersize=marker_size)
        plt.ylabel('x positions')
        plt.xlabel('time')
        plt.show()

        plt.plot(times, y_poss, plot_style, markersize=marker_size)
        plt.ylabel('y positions')
        plt.xlabel('time')
        plt.show()

        plt.plot(times, thetas, plot_style, markersize=marker_size)
        plt.ylabel('thetas')
        plt.xlabel('time')
        plt.show()

        plt.plot(times, a1dots, plot_style, markersize=marker_size)
        plt.ylabel('a1dot')
        plt.xlabel('time')
        plt.show()

        plt.plot(times, a2dots, plot_style, markersize=marker_size)
        plt.ylabel('a2dot')
        plt.xlabel('time')
        plt.show()

    if(yes == True):
        x_pos = [robot.x]
        y_pos = [robot.y]
        thetas = [robot.theta]
        time = [0]
        a1 = [robot.a1]
        a2 = [robot.a2]
        a1dotl = [robot.a1dot]
        a2dotl = [robot.a2dot]


        robot_params = []

        print('initial x y theta a1 a2: ', robot.x, robot.y, robot.theta, robot.a1, robot.a2)
        for t in range(len(a1dots_)):
            print(t + 1, 'th iteration')
            a1dot = float(a1dots_[t])
            a2dot = float(a2dots_[t])
            # a1dot = 1 / 3 * cos(t / 5)
            # a2dot = 1 / 3 * sin(t / 5)
            action = (a1dot, a2dot)
            robot.move(action)
            print('action taken(a1dot, a2dot): ', action)
            print('robot x y theta a1 a2: ', robot.x, robot.y, robot.theta, robot.a1, robot.a2)
            print('supposed x y theta a1 a2: ', x_real[t], y_real[t], theta_real[t], a1_real[t], a2_real[t])
            x_pos.append(robot.x)
            y_pos.append(robot.y)
            thetas.append(robot.theta)
            time.append(t + 1)
            a1.append(robot.a1)
            a2.append(robot.a2)
            a1dotl.append(robot.a1dot)
            a2dotl.append(robot.a2dot)
            robot_param = [robot.x,
                           robot.y,
                           robot.theta,
                           float(robot.a1),
                           float(robot.a2),
                           robot.a1dot,
                           robot.a2dot]
            robot_params.append(robot_param)

        plot_style = "--bo"
        marker_size = 3

        plt.plot(x_pos, y_pos, plot_style, markersize = marker_size)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('y vs x')
        plt.show()

        plt.plot(time, a1, plot_style, markersize = marker_size)
        plt.ylabel('a1 displacements')
        plt.show()

        plt.plot(time, a2, plot_style, markersize = marker_size)
        plt.ylabel('a2 displacements')
        plt.show()

        plt.plot(time, a1dotl, plot_style, markersize = marker_size)
        plt.ylabel('a1dot')
        plt.show()

        plt.plot(time, a2dotl, plot_style, markersize = marker_size)
        plt.ylabel('a2dot')
        plt.show()

        plt.plot(time, x_pos, plot_style, markersize = marker_size)
        plt.ylabel('x positions')
        plt.show()

        plt.plot(time, y_pos, plot_style, markersize = marker_size)
        plt.ylabel('y positions')
        plt.show()

        plt.plot(time, thetas, plot_style, markersize = marker_size)
        plt.ylabel('thetas')
        plt.show()
        plt.close()

if __name__ == "__main__":
    rollout_action('rollout_7_8.csv')
    #rollout_by_model('sac_model_trial_12_41_5_6_PI.zip', IdealFluidSwimmerEnv(reset_theta = False), 0, 0, True)
    #x, a1, a2 = find_min_config('sac_model_trial_12_41_5_6_PI.zip')
    #print('================')
    #print('a1: ', a1)
    #print('a2: ', a2)
    #print('x: ', x)