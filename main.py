"""
This is a file includes the main function for gym atari reinforcement learning.
"""
import os
import gym
import numpy as np

from argparse_decorate import init_parser, add_arg

from deepQLearning.deep_q_learning import DQLAtari
from discreteSoftActorCirtics.sac_discrete import SoftActorCriticsDiscrete

os.system('Xvfb :97 -screen 0 640x480x24 &')


@init_parser()
@add_arg('--start_episode', type=int, default=0, help='A number for start episode index.')
@add_arg('--eval', type=bool, default=False, help='True means evaluate model only.')
@add_arg('--game_index', type=int, default=1, choices=[0, 1, 2],
         help='Represent Breakout, MsPacman and Pong respectively.')
@add_arg('--env_name', type=str, default=None, help='The name of the gym atari environment.')
@add_arg('--memory_size', type=int, default=100000, help='The size of the memory space.')
@add_arg('--start_epsilon', type=float, default=1.0, help='The probability for random actions.')
@add_arg('--min_epsilon', type=float, default=0.05, help='The probability for random actions.')
@add_arg('--reward_clip', type=bool, default=False, help='Clip reward in [-1, 1] range if True.')
@add_arg('--live_penalty', type=bool, default=True, help='Penalties when agent lose a life in the game.')
@add_arg('--agent', type=str, default='dsac', choices=['dql', 'dsac'],
         help='Deep Q-learning and discrete soft Actor-Critics algorithms.')
def main(**kwargs):
    """
    The main function for gym atari reinforcement learning.
    """
    atari_game = ['BreakoutNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'PongNoFrameskip-v4']
    atari_game_action = [['NOOP', 'FIRE', 'RIGHT', 'LEFT'],
                         ['NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT'], ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']]
    env_name = kwargs['env_name'] if kwargs['env_name'] is not None else atari_game[kwargs['game_index']]
    action_name_list = atari_game_action[kwargs['game_index']]
    dirs = './' + env_name
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    img_size = (4, 84, 84)
    env = gym.make(env_name)
    memory_par = (kwargs['memory_size'], img_size)
    action_space = np.array([i for i in range(env.action_space.n)], dtype=np.uint8)
    game = (env_name, env, kwargs['live_penalty'])

    if kwargs['agent'] == 'dql':
        agent = DQLAtari(memory_par=memory_par,
                         action_space=action_space,
                         action_name_list=action_name_list,
                         game=game,
                         reward_clip=kwargs['reward_clip'],
                         epsilon=(kwargs['start_epsilon'], kwargs['min_epsilon']))
    elif kwargs['agent'] == 'dsac':
        agent = SoftActorCriticsDiscrete(memory_par=memory_par,
                                         action_space=action_space,
                                         game=game,
                                         reward_clip=kwargs['reward_clip'])
    else:
        raise Exception('The agent does not exist.')

    agent.simulate(net_path=dirs, start_episodes=kwargs['start_episode'], eval=kwargs['eval'], start_frames=0)


if __name__ == '__main__':
    main()
