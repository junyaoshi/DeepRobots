import gym

from envs.IdealFluidSwimmerWithSpringFixedKCEnv import IdealFluidSwimmerWithSpringEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ppo2.ppo2 import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import os

def addDateTime(s = ""):
    """
    Adds the current date and time at the end of a string.
    Inputs:
        s -> string
    Output:
        S = s_Dyymmdd_HHMM
    """
    import datetime
    date = str(datetime.datetime.now())
    date = date[2:4] + date[5:7] + date[8:10] + '_' + date[11:13] + date[14:16] + date[17:19]
    return s + '_D' + date


raw_env = IdealFluidSwimmerWithSpringEnv()
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
vec_env = DummyVecEnv([lambda: raw_env])

trial_name = "trial_1.24"
trial_name = addDateTime(trial_name)
results_dir = os.path.join("..", "results")
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)
results_dir = os.path.join(results_dir, "PPO_IdealFluidSwimmerWithSpringFixedKC")
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)
results_dir = os.path.join(results_dir, trial_name)
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)
model_path = os.path.join(results_dir, "model")

tensorboard_dir = os.path.join(results_dir, "tensorboard")
model = PPO2(MlpPolicy, vec_env, verbose=1, tensorboard_log=tensorboard_dir)
model.learn(total_timesteps=30000, tb_log_name="trial_run")
model.save(model_path)

env = vec_env.envs[0]
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
for i in range(1000):
    x_prev = env.snake_robot.x
    action, _states = model.predict(obs_prev)
    obs, rewards, dones, info = env.step(action)
    x = env.snake_robot.x
    print("Timestep: {} | State: {} | Action a1ddot: {} | "
          "Reward: {} | dX: {}".format(i, obs_prev, action[0], rewards, x - x_prev))
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
    # robot_param = [env.snake_robot.x,
    #                env.snake_robot.y,
    #                env.snake_robot.theta,
    #                float(env.snake_robot.a1),
    #                float(env.snake_robot.a2),
    #                env.snake_robot.a1dot,
    #                env.snake_robot.a2dot]
    # robot_params.append(robot_param)

plots_dir = os.path.join(results_dir, "PolicyRolloutPlots")
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
plt.title('y vs x')
plt.savefig(os.path.join(plots_dir, 'y vs x' + '.png'))
# plt.show()
plt.close()

plt.plot(times, a1s, plot_style, markersize=marker_size)
plt.ylabel('a1 displacements')
plt.xlabel('time')
plt.savefig(os.path.join(plots_dir, 'a1 displacements' + '.png'))
# plt.show()
plt.close()

plt.plot(times, a2s, plot_style, markersize=marker_size)
plt.ylabel('a2 displacements')
plt.xlabel('time')
plt.savefig(os.path.join(plots_dir, 'a2 displacements' + '.png'))
# plt.show()
plt.close()

plt.plot(times, x_poss, plot_style, markersize=marker_size)
plt.ylabel('x positions')
plt.xlabel('time')
plt.savefig(os.path.join(plots_dir, 'x positions' + '.png'))
# plt.show()
plt.close()

plt.plot(times, y_poss, plot_style, markersize=marker_size)
plt.ylabel('y positions')
plt.xlabel('time')
plt.savefig(os.path.join(plots_dir, 'y positions' + '.png'))
# plt.show()
plt.close()

plt.plot(times, thetas, plot_style, markersize=marker_size)
plt.ylabel('thetas')
plt.xlabel('time')
plt.savefig(os.path.join(plots_dir, 'thetas' + '.png'))
# plt.show()
plt.close()

plt.plot(times, a1dots, plot_style, markersize=marker_size)
plt.ylabel('a1dot')
plt.xlabel('time')
plt.savefig(os.path.join(plots_dir, 'a1dot' + '.png'))
# plt.show()
plt.close()

plt.plot(times, a2dots, plot_style, markersize=marker_size)
plt.ylabel('a2dot')
plt.xlabel('time')
plt.savefig(os.path.join(plots_dir, 'a2dot' + '.png'))
# plt.show()
plt.close()

plt.plot(times, ks, plot_style, markersize=marker_size)
plt.ylabel('k')
plt.xlabel('time')
plt.savefig(os.path.join(plots_dir, 'k' + '.png'))
# plt.show()
plt.close()

plt.plot(times, cs, plot_style, markersize=marker_size)
plt.ylabel('c')
plt.xlabel('time')
plt.savefig(os.path.join(plots_dir, 'c' + '.png'))
# plt.show()
plt.close()

plt.plot(times, a1ddots, plot_style, markersize=marker_size)
plt.ylabel('a1ddot')
plt.xlabel('time')
plt.savefig(os.path.join(plots_dir, 'a1ddot' + '.png'))
# plt.show()
plt.close()