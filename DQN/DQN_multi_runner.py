import subprocess

if __name__ == "__main__":
    episodes = 100
    iterations = 500
    trial_num = 36
    model_architecture = "50_10"
    t_interval = 4
    for robot_type in ["wheeled"]: #, "wheeled"
        for reward_func in ["forward", "left"]:
            arg = "python DQN/DQN_runner.py"
            arg += " --trial_num {}".format(trial_num)
            arg += " --episodes {}".format(episodes)
            arg += " --iterations {}".format(iterations)
            arg += " --robot_type {}".format(robot_type)
            arg += " --reward_func {}".format(reward_func)
            arg += " --model_architecture {}".format(model_architecture)
            arg += " --t_interval {}".format(t_interval)

            # print(arg)
            p = subprocess.Popen(arg, shell=True)
            p.communicate()
            trial_num += 1
