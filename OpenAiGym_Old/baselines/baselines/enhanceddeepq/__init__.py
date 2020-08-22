from baselines.enhanceddeepq import models  # noqa
from baselines.enhanceddeepq.build_graph import build_act, build_train  # noqa
from baselines.enhanceddeepq.enhanceddeepq import learn, load_act  # noqa
from baselines.enhanceddeepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa

def wrap_atari_dqn(env):
    from baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=False)
