from stable_baselines.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import os
from math import pi
from utils.csv_generator import generate_csv

class ExpLogger(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0, reset_freq = 100, vec_test_env=None, trial_name=None):
        super(ExpLogger, self).__init__(verbose)
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
        self.path = "/Users/brandonzhang/Desktop/rl_research/sac_log/"
        self.filename = trial_name + "_log.txt"
        self.file = open(self.path + self.filename, 'w')
        self.file.write('x,y,theta,ahead,atail\n')

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
        env = self.model.get_env().envs[0]
        model_robot = env.snake_robot
        x = model_robot.get_position()[0]
        y = model_robot.get_position()[1]
        state = model_robot.state
        theta = state[0]
        ahead = state[1]
        atail = state[2]

        data_str = str(x) + ',' + str(y) + ',' + str(theta) + ',' + str(ahead) + ',' + str(atail) + '\n'
        if(env.is_test == False):
            if(env.did_reset):
                self.file.write('BREAK\n')
            self.file.write(data_str)
        
        #print('x, y, theta, ahead, atail: ', x,y,state,theta,ahead,atail)

        self.curr_step += 1
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
        self.file.close()