import subprocess

if __name__ == "__main__":

    '''
    Testing: t_interval = 4
    '''
    # episodes =100
    iterations = 500
    trial_num = 66
    trial_note = "left1"
    model_architecture = "150_30"
    t_interval = 1
    batch_size = 16
    C = 100
    lr = 2e-4
    for episodes in [20, 40, 60]:
        for i in range(1):
            for robot_type in ["wheeled"]:  #
                for reward_func in ["left"]:  #, "forward"
                    arg = "python DQN/DQN_runner.py"
                    arg += " --trial_num {}".format(trial_num)
                    arg += " --episodes {}".format(episodes)
                    arg += " --iterations {}".format(iterations)
                    arg += " --robot_type {}".format(robot_type)
                    arg += " --reward_func {}".format(reward_func)
                    arg += " --model_architecture {}".format(model_architecture)
                    arg += " --t_interval {}".format(t_interval)
                    arg += " --batch_size {}".format(batch_size)
                    arg += " --network_update_freq {}".format(C)
                    arg += " --learning_rate {}".format(lr)
                    arg += " --trial_note {}".format(trial_note)

                    # print(arg)
                    p = subprocess.Popen(arg, shell=True)
                    p.communicate()
                    trial_num += 1

    '''
    Testing: t_interval = 1, wheeled
    
    episodes = 2000
    iterations = 500
    model_architecture = "300_100_10"
    t_interval = 1
    batch_size = 8
    lr = 1e-4
    C = 200
    for i in range(1):
        for robot_type in ["wheeled"]:  #"swimming",
            for reward_func in ["forward"]:
                arg = "python DQN/DQN_runner.py"
                arg += " --trial_num {}".format(trial_num)
                arg += " --episodes {}".format(episodes)
                arg += " --iterations {}".format(iterations)
                arg += " --robot_type {}".format(robot_type)
                arg += " --reward_func {}".format(reward_func)
                arg += " --model_architecture {}".format(model_architecture)
                arg += " --t_interval {}".format(t_interval)
                arg += " --batch_size {}".format(batch_size)
                arg += " --network_update_freq {}".format(C)
                arg += " --learning_rate {}".format(lr)

                # print(arg)
                p = subprocess.Popen(arg, shell=True)
                p.communicate()
                trial_num += 1
    '''
