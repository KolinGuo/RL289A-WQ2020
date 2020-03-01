'''
## Train ##
# Code to train Deep Q Network on gym-sokoban environment
# Adapted from: https://github.com/msinto93/DQN_Atari/blob/master/train.py
'''

from datetime import datetime
import json
import os
import argparse
import gym
import gym_sokoban
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def get_train_args():
    train_args = argparse.ArgumentParser()

    # Environment parameters
    train_args.add_argument("--env", type=str, default='Boxoban-Train-v0', help="Environment to use for training")
    train_args.add_argument("--num_room_states", type=int, default=7, help="Number of room states for one-hot encoding")
    train_args.add_argument("--render", type=bool, default=False, help="Whether or not to display the environment during training")
    train_args.add_argument("--grid_width", type=int, default=10, help="Grid width")
    train_args.add_argument("--grid_height", type=int, default=10, help="Grid height")
    train_args.add_argument("--steps_per_state", type=int, default=4, help="Sequence of steps which constitutes a single state")

    # Training parameters
    train_args.add_argument("--num_steps_train", type=int, default=5e7, help="Number of steps to train for")
    train_args.add_argument("--batch_size", type=int, default=32, help="Batch size of state transitions")
    train_args.add_argument("--learning_rate", type=float, default=0.00025, help="Learning rate")
    train_args.add_argument("--replay_mem_size", type=int, default=1e6, help="Maximum number of steps in replay memory buffer")
    train_args.add_argument("--initial_replay_mem_size", type=int, default=5e4, help="Initial number of steps in replay memory (populated by random actions) before learning can start")
    train_args.add_argument("--epsilon_start", type=float, default=1.0, help="Exploration rate at the beginning of training")
    train_args.add_argument("--epsilon_end", type=float, default=0.1, help="Fixed exploration rate at the end of epsilon decay")
    train_args.add_argument("--epsilon_decay_step", type=float, default=1e6, help="After how many steps to stop decaying the exploration rate")
    train_args.add_argument("--discount_rate", type=float, default=0.99, help="Discount rate (gamma) for future rewards")
    train_args.add_argument("--update_target_step", type=int, default=1e4, help="Copy current network parameters to target network every N steps")
    train_args.add_argument("--save_checkpoint_step", type=int, default=25e4, help="Save checkpoint every N steps")
    train_args.add_argument("--save_log_step", type=int, default=1000, help="Save logs every N steps")

    # Files/directories
    train_args.add_argument("--checkpoint_dir", type=str, default='./checkpoints', help="Directory for saving/loading checkpoints")
    train_args.add_argument("--checkpoint_file", type=str, default=None, help="Checkpoint file to load and resume training from (if None, train from scratch)")
    train_args.add_argument("--log_dir", type=str, default='./logs/train', help="Directory for saving logs")
    log_file_suffix = datetime.now().strftime("_%Y%m%d_%H%M%S")
    train_args.add_argument("--log_file_suffix", type=str, default=log_file_suffix, help="Log file suffix (current timestamp) DON'T MODIFY")

    return train_args.parse_args()

def log_train_args(args):
    # Create summary writer to write summaries to disk
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    log_file_name = 'Training_args' + args.log_file_suffix + '.log'
    # Write the training arguments
    with open(os.path.join(args.log_dir, log_file_name), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

def train(args):
    pass

if __name__ == '__main__':
    # Change back to repository directory
    os.chdir(os.path.realpath(os.path.join(os.path.abspath(__file__), '../../')))
    train_args = get_train_args()
    log_train_args(train_args)
    train(train_args)

