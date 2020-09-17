import gym
import os

from envs.IdealFluidSwimmerWithSpringEnv import IdealFluidSwimmerWithSpringEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ppo2.ppo2 import PPO2
import matplotlib.pyplot as plt

env = IdealFluidSwimmerWithSpringEnv()

dir_name = "results\LearningResults\PPO_WheeledRobotPybullet"
tensorboard_dir = dir_name + "\\tensorboard"
model_dir = dir_name + "\\model"
model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=tensorboard_dir, full_tensorboard_log=True)
model.learn(total_timesteps=25, tb_log_name="test")
model.save(model_dir)

obs_prev = env.reset()

x_poss = [env.snake_robot.x]
y_poss = [env.snake_robot.y]
thetas = [env.snake_robot.theta]
times = [0]
a1s = [env.snake_robot.a1]
a2s = [env.snake_robot.a2]
a1dots = [env.snake_robot.a1dot]
a2dots = [env.snake_robot.a2dot]
a1ddots = [env.snake_robot.a1ddot]
ks = [env.snake_robot.k]
cs = [env.snake_robot.c]
# robot_params = []
for i in range(100):
    x_prev = env.snake_robot.x
    action, _states = model.predict(obs_prev)
    obs, rewards, dones, info = env.step(action)
    x = env.snake_robot.x
    print("Timestep: {} | State: {} | Action a1ddot: {}; k: {}; c: {}| "
          "Reward: {} | dX: {}".format(i, obs_prev, action[0], action[1] * 10000, action[2] * 10000, rewards, x - x_prev))
    obs_prev = obs
    x_poss.append(env.snake_robot.x)
    y_poss.append(env.snake_robot.y)
    thetas.append(env.snake_robot.theta)
    times.append(i)
    a1s.append(env.snake_robot.a1)
    a2s.append(env.snake_robot.a2)
    a1dots.append(env.snake_robot.a1dot)
    a2dots.append(env.snake_robot.a2dot)
    a1ddots.append(env.snake_robot.a1ddot)
    ks.append(env.snake_robot.k)
    cs.append(env.snake_robot.c)
    # robot_param = [env.swimmer.x,
    #                env.swimmer.y,
    #                env.swimmer.theta,
    #                float(env.swimmer.a1),
    #                float(env.swimmer.a2),
    #                env.swimmer.a1dot,
    #                env.swimmer.a2dot]
    # robot_params.append(robot_param)

plots_dir = dir_name + "\\PolicyRolloutPlots\\"
if not os.path.isdir(plots_dir):
    os.mkdir(plots_dir)

# view results
# print('x positions are: ' + str(x_pos))
# print('y positions are: ' + str(y_pos))
# print('thetas are: ' + str(thetas))

plot_style = "--bo"
marker_size = 3

plt.plot(x_poss, y_poss, plot_style, markersize=marker_size)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(plots_dir + 'y vs x' + '.png')
plt.close()

plt.plot(times, a1s, plot_style, markersize=marker_size)
plt.ylabel('a1 displacements')
plt.xlabel('time')
plt.savefig(plots_dir + 'a1 displacements' + '.png')
plt.close()

plt.plot(times, a2s, plot_style, markersize=marker_size)
plt.ylabel('a2 displacements')
plt.xlabel('time')
plt.savefig(plots_dir + 'a2 displacements' + '.png')
plt.close()

plt.plot(times, x_poss, plot_style, markersize=marker_size)
plt.ylabel('x positions')
plt.xlabel('time')
plt.savefig(plots_dir + 'x positions' + '.png')
plt.close()

plt.plot(times, y_poss, plot_style, markersize=marker_size)
plt.ylabel('y positions')
plt.xlabel('time')
plt.savefig(plots_dir + 'y positions' + '.png')
plt.close()

plt.plot(times, thetas, plot_style, markersize=marker_size)
plt.ylabel('thetas')
plt.xlabel('time')
plt.savefig(plots_dir + 'thetas' + '.png')
plt.close()

plt.plot(times, a1dots, plot_style, markersize=marker_size)
plt.ylabel('a1dot')
plt.xlabel('time')
plt.savefig(plots_dir + 'a1dot' + '.png')
plt.close()

plt.plot(times, a2dots, plot_style, markersize=marker_size)
plt.ylabel('a2dot')
plt.xlabel('time')
plt.savefig(plots_dir + 'a2dot' + '.png')
plt.close()

# plt.plot(times, ks, plot_style, markersize=marker_size)
# plt.ylabel('k')
# plt.xlabel('time')
# plt.savefig(plots_dir + 'k' + '.png')
# plt.close()
#
# plt.plot(times, cs, plot_style, markersize=marker_size)
# plt.ylabel('c')
# plt.xlabel('time')
# plt.savefig(plots_dir + 'c' + '.png')
# plt.close()

# plt.plot(times, a1ddots, plot_style, markersize=marker_size)
# plt.ylabel('a1ddot')
# plt.xlabel('time')
# plt.savefig(dir_name + 'a1ddot' + '.png')
# plt.close()