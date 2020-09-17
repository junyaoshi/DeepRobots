import gym

from envs.IdealFluidSwimmerWithSpringEnv import IdealFluidSwimmerWithSpringEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ppo2.ppo2 import PPO2
import matplotlib.pyplot as plt

env = IdealFluidSwimmerWithSpringEnv()

tensorboard_dir = "results/LearningResults/PPO_IdealFluidSwimmerWithSpring"
model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=tensorboard_dir)
model.learn(total_timesteps=25, tb_log_name="trial_run")

obs_prev = env.reset()

x_poss = [env.swimmer.x]
y_poss = [env.swimmer.y]
thetas = [env.swimmer.theta]
times = [0]
a1s = [env.swimmer.a1]
a2s = [env.swimmer.a2]
a1dots = [env.swimmer.a1dot]
a2dots = [env.swimmer.a2dot]
a1ddots = [env.swimmer.a1ddot]
ks = [env.swimmer.k]
cs = [env.swimmer.c]
# robot_params = []
for i in range(100):
    x_prev = env.swimmer.x
    action, _states = model.predict(obs_prev)
    obs, rewards, dones, info = env.step(action)
    x = env.swimmer.x
    print("Timestep: {} | State: {} | Action a1ddot: {}; k: {}; c: {}| "
          "Reward: {} | dX: {}".format(i, obs_prev, action[0], action[1], action[2], rewards, x - x_prev))
    obs_prev = obs
    x_poss.append(env.swimmer.x)
    y_poss.append(env.swimmer.y)
    thetas.append(env.swimmer.theta)
    times.append(i)
    a1s.append(env.swimmer.a1)
    a2s.append(env.swimmer.a2)
    a1dots.append(env.swimmer.a1dot)
    a2dots.append(env.swimmer.a2dot)
    a1ddots.append(env.swimmer.a1ddot)
    ks.append(env.swimmer.k)
    cs.append(env.swimmer.c)
    # robot_param = [env.swimmer.x,
    #                env.swimmer.y,
    #                env.swimmer.theta,
    #                float(env.swimmer.a1),
    #                float(env.swimmer.a2),
    #                env.swimmer.a1dot,
    #                env.swimmer.a2dot]
    # robot_params.append(robot_param)

dir_name = tensorboard_dir + "/PolicyRolloutPlots"

# view results
# print('x positions are: ' + str(x_pos))
# print('y positions are: ' + str(y_pos))
# print('thetas are: ' + str(thetas))

plot_style = "--bo"
marker_size = 3

plt.plot(x_poss, y_poss, plot_style, markersize=marker_size)
plt.xlabel('x')
plt.ylabel('y')
plt.title('y vs x')
plt.savefig(dir_name + 'y vs x' + '.png')
plt.show()

plt.plot(times, a1s, plot_style, markersize=marker_size)
plt.ylabel('a1 displacements')
plt.xlabel('time')
plt.savefig(dir_name + 'a1 displacements' + '.png')
plt.show()

plt.plot(times, a2s, plot_style, markersize=marker_size)
plt.ylabel('a2 displacements')
plt.xlabel('time')
plt.savefig(dir_name + 'a2 displacements' + '.png')
plt.show()

plt.plot(times, x_poss, plot_style, markersize=marker_size)
plt.ylabel('x positions')
plt.xlabel('time')
plt.savefig(dir_name + 'x positions' + '.png')
plt.show()

plt.plot(times, y_poss, plot_style, markersize=marker_size)
plt.ylabel('y positions')
plt.xlabel('time')
plt.savefig(dir_name + 'y positions' + '.png')
plt.show()

plt.plot(times, thetas, plot_style, markersize=marker_size)
plt.ylabel('thetas')
plt.xlabel('time')
plt.savefig(dir_name + 'thetas' + '.png')
plt.show()

plt.plot(times, a1dots, plot_style, markersize=marker_size)
plt.ylabel('a1dot')
plt.xlabel('time')
plt.savefig(dir_name + 'a1dot' + '.png')
plt.show()

plt.plot(times, a2dots, plot_style, markersize=marker_size)
plt.ylabel('a2dot')
plt.xlabel('time')
plt.savefig(dir_name + 'a2dot' + '.png')
plt.show()
plt.close()

plt.plot(times, ks, plot_style, markersize=marker_size)
plt.ylabel('k')
plt.xlabel('time')
plt.savefig(dir_name + 'k' + '.png')
plt.show()

plt.plot(times, cs, plot_style, markersize=marker_size)
plt.ylabel('c')
plt.xlabel('time')
plt.savefig(dir_name + 'c' + '.png')
plt.show()

plt.plot(times, a1ddots, plot_style, markersize=marker_size)
plt.ylabel('a1ddot')
plt.xlabel('time')
plt.savefig(dir_name + 'a1ddot' + '.png')
plt.show()

plt.close()