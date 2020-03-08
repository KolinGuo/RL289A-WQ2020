'''
## Train ##
# Code to train Deep Q Network on gym-sokoban environment
# Adapted from: https://github.com/msinto93/DQN_Atari/blob/master/train.py
'''

from datetime import datetime
import json, os, sys, argparse, logging
import gym, gym_sokoban
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from utils.experience_replay import ReplayMemory
from utils.state_buffer import StateBuffer
from utils.network import DQNModel

def get_train_args():
    train_args = argparse.ArgumentParser()

    # Environment parameters
    train_args.add_argument("--env", type=str, default='Boxoban-Train-v0', help="Environment to use for training")
    train_args.add_argument("--num_room_states", type=int, default=7, help="Number of room states for one-hot encoding")
    train_args.add_argument("--render", type=bool, default=False, help="Whether or not to display the environment during training")
    train_args.add_argument("--grid_width", type=int, default=10, help="Grid width")
    train_args.add_argument("--grid_height", type=int, default=10, help="Grid height")
    train_args.add_argument("--grids_per_state", type=int, default=4, help="Sequence of grids which constitutes a single state")

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
    log_filename = datetime.now().strftime("%Y%m%d_%H%M%S.log")
    train_args.add_argument("--log_filename", type=str, default=log_filename, help="Log file name (current timestamp) DON'T MODIFY")

    return train_args.parse_args()

def log_train_args(args):
    # Create summary writer to write summaries to disk
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Set up logging to file
    log_filepath = os.path.join(args.log_dir, args.log_filename)
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s] [%(name)-8s] [%(levelname)-8s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=log_filepath,
                        filemode='w')

    # define a Handler which writes INFO messages or higher to the sys.stdout
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('[%(name)-8s] [%(levelname)-8s] %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # Get logger for training args
    logger = logging.getLogger('args')

    # Write the training arguments
    for key, value in vars(args).items():
        logger.debug('{%s: %s}', key, value)

def train(args):
    # Get logger for training args
    logger = logging.getLogger('train')

    # Check if GPU is available
    logger.info("Num GPUs Available: %d", len(tf.config.experimental.list_physical_devices('GPU')))

    # Create environment
    env = gym.make(args.env)
    num_actions = 4     # Push (up, down, left, right): 1, 2, 3, 4

    # Initialize replay memory and state buffer
    replay_mem = ReplayMemory(args)
    state_buf = StateBuffer(args)

    # Instantiate DQN and DQN_target
    state_shape = (args.grid_height, args.grid_width, args.num_room_states, args.grids_per_state)
    load_model_path = None
    if args.checkpoint_file is not None:    # Resume training
        load_model_path = os.path.join(args.checkpoint_dir, args.checkpoint_file)

    DQN = DQNModel(state_shape, num_actions, args.learning_rate, load_model_path=load_model_path, name='DQN')
    DQN_target = DQNModel(state_shape, num_actions, load_model_path=load_model_path, name='DQN_target')

    # TODO: save loss and accuracy while training

    ## Begin training
    env.reset()

    # Populate replay memory to initial_replay_mem_size
    

    # Start training







if __name__ == '__main__':
    # Change back to repository directory
    os.chdir(os.path.realpath(os.path.join(os.path.abspath(__file__), '../../')))
    train_args = get_train_args()
    log_train_args(train_args)
    train(train_args)

