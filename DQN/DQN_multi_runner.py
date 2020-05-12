import subprocess

if __name__ == "__main__":

    '''
    Testing: t_interval = 4
    '''
    # episodes =100
    iterations = 250
    trial_num = 67
    trial_note = "forward_t_interval_"
    model_architecture = "50_20"
    # t_interval = 1
    batch_size = 16
    C = 100
    lr = 2e-4
    for episodes in [20]:
        for i in range(1):
            for robot_type in ["swimming"]:  #
                for reward_func in ["forward"]:  #, "forward"
                    for t_interval in [8, 1]:
                        arg = "python3 DQN_runner.py"
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
                        arg += " --trial_note {}".format(trial_note + str(t_interval))

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
