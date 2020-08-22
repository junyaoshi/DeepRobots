import gym

from envs.IdealFluidSwimmerEnv import IdealFluidSwimmerEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ppo2.ppo2 import PPO2
import matplotlib.pyplot as plt

env = IdealFluidSwimmerEnv()

model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="results/PPO_IdealFluidSwimmer")
model.learn(total_timesteps=20000, tb_log_name="trial_run")

obs_prev = env.reset()

x_pos = [env.swimmer.x]
y_pos = [env.swimmer.y]
thetas = [env.swimmer.theta]
time = [0]
a1 = [env.swimmer.a1]
a2 = [env.swimmer.a2]
a1dot = [env.swimmer.a1dot]
a2dot = [env.swimmer.a2dot]
robot_params = []
for i in range(100):
    x_prev = env.swimmer.x
    action, _states = model.predict(obs_prev)
    obs, rewards, dones, info = env.step(action)
    x = env.swimmer.x
    print("Timestep: {} | State: {} | Action: {} | Reward: {} | dX: {}".format(i, obs_prev, action, rewards, x - x_prev))
    obs_prev = obs
    x_pos.append(env.swimmer.x)
    y_pos.append(env.swimmer.y)
    thetas.append(env.swimmer.theta)
    time.append(i)
    a1.append(env.swimmer.a1)
    a2.append(env.swimmer.a2)
    a1dot.append(env.swimmer.a1dot)
    a2dot.append(env.swimmer.a2dot)
    robot_param = [env.swimmer.x,
                   env.swimmer.y,
                   env.swimmer.theta,
                   float(env.swimmer.a1),
                   float(env.swimmer.a2),
                   env.swimmer.a1dot,
                   env.swimmer.a2dot]
    robot_params.append(robot_param)

dir_name = "/Users/jackshi/Desktop/DeepRobotsResultPics/test/"

# view results
# print('x positions are: ' + str(x_pos))
# print('y positions are: ' + str(y_pos))
# print('thetas are: ' + str(thetas))

plt.plot(x_pos, y_pos)
plt.xlabel('x')
plt.ylabel('y')
plt.title('y vs x')
plt.savefig(dir_name + 'y vs x' + '.png')
plt.show()

plt.plot(time, a1)
plt.ylabel('a1 displacements')
plt.xlabel('time')
plt.savefig(dir_name + 'a1 displacements' + '.png')
plt.show()

plt.plot(time, a2)
plt.ylabel('a2 displacements')
plt.xlabel('time')
plt.savefig(dir_name + 'a2 displacements' + '.png')
plt.show()

plt.plot(time, x_pos)
plt.ylabel('x positions')
plt.xlabel('time')
plt.savefig(dir_name + 'x positions' + '.png')
plt.show()

plt.plot(time, y_pos)
plt.ylabel('y positions')
plt.xlabel('time')
plt.savefig(dir_name + 'y positions' + '.png')
plt.show()

plt.plot(time, thetas)
plt.ylabel('thetas')
plt.xlabel('time')
plt.savefig(dir_name + 'thetas' + '.png')
plt.show()

plt.plot(time, a1dot)
plt.ylabel('a1dot')
plt.xlabel('time')
plt.savefig(dir_name + 'a1dot' + '.png')
plt.show()

plt.plot(time, a2dot)
plt.ylabel('a2dot')
plt.xlabel('time')
plt.savefig(dir_name + 'a2dot' + '.png')
plt.show()
plt.close()

env.close()