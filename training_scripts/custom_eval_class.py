from stable_baselines.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import os
from math import pi
from utils.csv_generator import generate_csv

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0, reset_freq = 100, vec_test_env=None, trial_name=None):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.reset_freq = reset_freq
        self.curr_step = 0
        self.vec_env = vec_test_env
        self.trial_name = trial_name
        print("initialize custom callback")

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        self.curr_step += 1
        
        if self.curr_step % self.reset_freq != 0:
            return True

        env = self.vec_env.envs[0]
        env.enable_test_mode()

        for j in range(1):
            results_dir = os.path.join("results", "LearningResults", "PPO_IdealFluidSwimmer", self.trial_name, str(self.curr_step) + "_theta_int_" + str(j))
            if not os.path.isdir(results_dir):
                os.mkdir(results_dir)
            theta_init = (j*2*pi)/8
            obs_prev = env.reset_w_theta(theta_init)

            x_poss = [env.snake_robot.x]
            y_poss = [env.snake_robot.y]
            thetas = [env.snake_robot.theta]
            times = [0]
            a1s = [env.snake_robot.a1]
            a2s = [env.snake_robot.a2]
            a1dots = [env.snake_robot.a1dot]
            a2dots = [env.snake_robot.a2dot]

            robot_params = []
            for i in range(10):
                x_prev = env.snake_robot.x
                action, _states = self.model.predict(obs_prev)
                obs, rewards, dones, info = env.step(action)
                x = env.snake_robot.x

                print("Timestep: {} | State: {} | Action a1dot: {}; a2dot: {};| "
                      "Reward: {} | dX: {}".format(i, obs_prev, action[0], action[1], rewards, x - x_prev))
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
            #plt.show()
            plt.close()

            plt.plot(times, a1s, plot_style, markersize=marker_size)
            plt.ylabel('a1 displacements')
            plt.xlabel('time')
            plt.savefig(os.path.join(plots_dir, 'a1 displacements' + '.png'))
            #plt.show()
            plt.close()

            plt.plot(times, a2s, plot_style, markersize=marker_size)
            plt.ylabel('a2 displacements')
            plt.xlabel('time')
            plt.savefig(os.path.join(plots_dir, 'a2 displacements' + '.png'))
            #plt.show()
            plt.close()

            plt.plot(times, x_poss, plot_style, markersize=marker_size)
            plt.ylabel('x positions')
            plt.xlabel('time')
            plt.savefig(os.path.join(plots_dir, 'x positions' + '.png'))
            #plt.show()
            plt.close()

            plt.plot(times, y_poss, plot_style, markersize=marker_size)
            plt.ylabel('y positions')
            plt.xlabel('time')
            plt.savefig(os.path.join(plots_dir, 'y positions' + '.png'))
            #plt.show()
            plt.close()

            plt.plot(times, thetas, plot_style, markersize=marker_size)
            plt.ylabel('thetas')
            plt.xlabel('time')
            plt.savefig(os.path.join(plots_dir, 'thetas' + '.png'))
            #plt.show()
            plt.close()

            plt.plot(times, a1dots, plot_style, markersize=marker_size)
            plt.ylabel('a1dot')
            plt.xlabel('time')
            plt.savefig(os.path.join(plots_dir, 'a1dot' + '.png'))
            #plt.show()
            plt.close()

            plt.plot(times, a2dots, plot_style, markersize=marker_size)
            plt.ylabel('a2dot')
            plt.xlabel('time')
            plt.savefig(os.path.join(plots_dir, 'a2dot' + '.png'))
            #plt.show()
            plt.close()

            csv_path = os.path.join(plots_dir, 'rollout.csv')
            generate_csv(robot_params, csv_path)
        
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass