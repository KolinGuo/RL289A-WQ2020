'''
## Play ##
# Run a trained DQN on an Open AI gym environment and observe its performance on screen
@author: Mark Sinton (msinto93@gmail.com)
'''

import json, os, sys, argparse, logging, random, time
import numpy as np
import gym, gym_sokoban
import matplotlib.pyplot as plt
import tensorflow as tf

import matplotlib.pyplot as plt
%matplotlib inline
from IPython import display
from train import get_train_args
from utils.state_buffer import StateBuffer
from utils.network import DQNModel
from utils.utils import preprocess_observation, reset_env_and_state_buffer


def get_play_args(train_args):
    play_args = argparse.ArgumentParser()

    # Environment parameters (First 4 params must be same as those used in training)
    play_args.add_argument("--env", type=str, default='Sokoban-v0', help="Environment to use for training")
    play_args.add_argument("--num_surfaces", type=int, default=7, help="Number of room surfaces for one-hot encoding")
    play_args.add_argument("--max_step", type=int, default=200, help="Maximum number of steps in a single game episode")
    play_args.add_argument("--grid_width", type=int, default=10, help="Grid width")
    play_args.add_argument("--grid_height", type=int, default=10, help="Grid height")
    play_args.add_argument("--grids_per_state", type=int, default=4, help="Sequence of grids which constitutes a single state")

    # Play parameters
    play_args.add_argument("--num_eps", type=int, default=5, help="Number of episodes to run for")
    play_args.add_argument("--max_ep_length", type=int, default=2000, help="Maximum number of steps per episode")
    play_args.add_argument("--max_initial_random_steps", type=int, default=4, help="Maximum number of random steps to take at start of episode to ensure random starting point")

    # Files/directories
    play_args.add_argument("--checkpoint_dir", type=str, default='./checkpoints', help="Directory for saving/loading checkpoints")
    play_args.add_argument("--checkpoint_file", type=str, default=None, help="Checkpoint file to load and resume training from (if None, train from scratch)")

    return play_args.parse_args()


def play(args):
    ACTION_SPACE = np.array([1, 2, 3, 4], dtype=np.uint8)
    # Function to get a random action
    def sample_action_space():
        return random.choice(ACTION_SPACE)

    # Function to convert actionID (1, 2, 3, 4) to actionQID (0, 1, 2, 3)
    def actionID_to_actionQID(actionID):
        return actionID-1

    # Function to convert actionQID (0, 1, 2, 3) to actionID (1, 2, 3, 4)
    def actionQID_to_actionID(actionQID):
        return actionQID+1

    # Create environment
    env = gym.make(args.env)
    num_actions = env.action_space.n


    state_buf = StateBuffer(args)
    state_shape = (args.grid_height, args.grid_width, args.num_surfaces, args.grids_per_state)
    load_model_path = None
    if args.checkpoint_file is not None:    # Resume training
        load_model_path = os.path.join(args.checkpoint_dir, args.checkpoint_file)

    DQN_target = DQNModel(state_shape, num_actions, load_model_path=load_model_path, name='DQN_target')

    for ep in range(0, args.num_eps):
        # Reset environment and state buffer for next episode
        reset_env_and_state_buffer(env, state_buf, args)
        step = 0
        ep_done = False
        initial_steps = np.random.randint(1, args.max_initial_random_steps+1)

        while not ep_done:
            time.sleep(0.05)
            img = env.render(mode='rgb_array')
            plt.imshow(img)
            display.clear_output(wait=True)
            display.display(plt.gcf())
            #Choose random action for initial steps to ensure every episode has a random start point. Then choose action with highest Q-value according to network's current policy.
            if step < initial_steps:
                actionID = sample_action_space()
            else:
                state = tf.convert_to_tensor(state_buf.get_state())
                state = state[tf.newaxis, ...]      # Add an axis for batch
                actionQID = DQN_target.predict(state)
                actionID = actionQID_to_actionID(int(actionQID))    # convert from Tensor to int

            observation, reward, terminal, _ = env.step(actionID, observation_mode='rgb_array')
            grid = preprocess_observation(args, observation)
            state_buf.add(grid)
            step += 1

            # Episode can finish either by reaching terminal state or max episode steps
            if terminal or step == args.max_ep_length:
                ep_done = True




if  __name__ == '__main__':
    train_args = get_train_args()
    play_args = get_play_args(train_args)
    play(play_args)
