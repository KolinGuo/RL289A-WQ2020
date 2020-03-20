'''
## Test ##
# Test a trained DQN. This can be run alongside training by running 'run_every_new_ckpt.sh'.
@author: Mark Sinton (msinto93@gmail.com) 
'''
from datetime import datetime

import os
import sys
import argparse
import gym
import tensorflow as tf
import numpy as np
import scipy.stats as ss
import random
import re
import pdb

from train import get_train_args
from utils.utils import preprocess_observation, reset_env_and_state_buffer
from utils.state_buffer import StateBuffer
from utils.network import DQNModel
    
def get_test_args(train_args):
    test_params = argparse.ArgumentParser()

    # Environment parameters
    test_params.add_argument('--env', '-e', metavar='env', help='Environment to load (default: Boxoban-Val-v0)', default='Boxoban-Val-v0')
    test_params.add_argument("--num_surfaces", type=int, default=7, help="Number of room surfaces for one-hot encoding")
    test_params.add_argument("--max_step", type=int, default=200, help="Maximum number of steps in a single game episode")
    test_params.add_argument("--render", type=bool, default=False, help="Whether or not to display the environment during training")
    test_params.add_argument("--grid_width", type=int, default=10, help="Grid width")
    test_params.add_argument("--grid_height", type=int, default=10, help="Grid height")
    test_params.add_argument("--grids_per_state", type=int, default=4, help="Sequence of grids which constitutes a single state")
    test_params.add_argument("--random_seed", type=int, default=4321, help="Random seed for reproducability")

    # Environment rewards
    test_params.add_argument("--env_penalty_for_step", type=float, default=-0.1, help="Reward of performing a step")
    test_params.add_argument("--env_reward_box_on_target", type=float, default=1.0, help="Reward of pushing a box on target")
    test_params.add_argument("--env_penalty_box_off_target", type=float, default=-1.0, help="Reward of pushing a box off target")
    test_params.add_argument("--env_reward_finished", type=float, default=10.0, help="Reward of winning (pushed all boxes on targets)")
    
    # Testing parameters
    test_params.add_argument("--num_eps", type=int, default=2, help="Number of episodes to test for")
    test_params.add_argument("--max_initial_random_steps", type=int, default=10, help="Maximum number of random steps to take at start of episode to ensure random starting point")
    test_params.add_argument("--epsilon_value", type=float, default=0.05, help="Exploration rate for the play")

    # Files/directories
    test_params.add_argument("--checkpoint_dir", type=str, default='./ckpts', help="Directory for saving/loading checkpoints")
    test_params.add_argument("--checkpoint_file", type=str, default=None, help="Checkpoint file to load (if None, load latest ckpt)")
    test_params.add_argument("--checkpoint_list", type=str, default=None, help="Text file of list of checkpoints to load")
    test_params.add_argument("--results_dir", type=str, default='./test_results', help="Directory for saving txt file of results")
    results_filename = datetime.now().strftime("%Y%m%d_%H%M%S_results.txt")
    test_params.add_argument("--results_file", type=str, default=results_filename, help="Text file of test results (if None, do not save results), (current timestamp) DON'T MODIFY")
    
    return test_params.parse_args()


class Counts():
    def __init__(self):
        self.on = 0
        self.off = 0
        self.win = 0

    def reward_update(self, reward):
        if reward == 0.9:
            self.on = self.on + 1
        elif reward == -1.1:
            self.off = self.off + 1
        elif reward == 9.9:
            self.win = 1
        elif reward != -0.1:
            print("Weird Reward: " + str(reward))
            exit(0)

    def update_all(self, newCounts):
        self.on = self.on + newCounts.on
        self.off = self.off + newCounts.off
        self.win = self.win + newCounts.win

    def get (self):
        return (self.on, self.off, self.win)

    def get_str(self):
        return '\rMoved block on target: {:d} \tMoved block off target: {:d} \t Win: {:d}\n\r'.format(self.on, self.off, self.win)
        #'\x1b[2K\rMoved block on target: {:d} \tMoved block off target: {:d} \t Win: {:d} \t'.format(self.on, self.off, self.win)


def get_checkpoint_paths(args):
    checkpoint_paths = []

    #Load single checkpoint path
    if args.checkpoint_file is not None: 
        load_model_path = os.path.join(args.checkpoint_dir, args.checkpoint_file)
        checkpoint_paths.append(load_model_path)

    #Load checkpoint paths to array
    elif args.checkpoint_dir is not None: 
        directory = os.listdir(args.checkpoint_dir)

        checkpoint_paths.append(None) #Put in starting case with no cp

        #Get decimal cp codes and sort so tests are done in order
        cp_codes = []
        for file_in in directory:
            match = re.search("ckpt\-(\d+)\.index", file_in)
            if match:
                cp_codes.append(int(match.group(1)))
        cp_codes.sort()

        #create sorted path names
        for code in cp_codes:
            file_out = "ckpt-" + str(code)
            load_model_path = os.path.join(args.checkpoint_dir, file_out)
            checkpoint_paths.append(load_model_path)

    #No arguments, just the default starting case with no cp
    else:
        checkpoint_paths.append(None)

    return checkpoint_paths


def output(out_str, out_file):
    print(out_str)
    out_file.write(out_str)


def test(args):
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

    def get_actionID(step, initial_steps, epsilon):
    #Choose random action for initial steps to ensure every episode has a random start point. Then choose action with highest Q-value according to network's current policy.
        if step < initial_steps:
            actionID = sample_action_space()
        else:
            if random.random() < epsilon:   # Take random action
                actionID = sample_action_space()
            else:   # Take greedy action
                state = tf.convert_to_tensor(state_buf.get_state(), dtype=tf.float32)
                state = state[tf.newaxis, ...]      # Add an axis for batch
                actionQID = DQN_target.predict(state)
                actionID = actionQID_to_actionID(int(actionQID))    # convert from Tensor to int
        return actionID


    # Create environment
    env = gym.make(args.env)
    

    # Set random seeds for reproducability
    env.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)

    state_buf = StateBuffer(args)
    state_shape = (args.grid_height, args.grid_width, args.num_surfaces, args.grids_per_state)
    num_actions = 4

    epsilons = [0.1]# [0.9, 0.5, 0.28, 0.2, 0.15, 0.1, 0.05]
    checkpoint_paths = get_checkpoint_paths(args)
    num_checkpoints = len(checkpoint_paths)

    out_file = open(args.results_file, "a+")
    out_file.write("pathStr, epsilon, mean_reward, error_reward, mean_step, ons, offs, wins\n\r")

    for epsilon in epsilons:
        for cp_id in range(0, num_checkpoints):
            path = checkpoint_paths[cp_id]

            out_str = "Starting Checkpoint test: {} \t {}/{} \t Epsilon: {}\n\r".format(path, cp_id + 1, num_checkpoints, epsilon)
            #output(out_str, out_file)
            print(out_str)

            #if args.checkpoint_list is not None:
            DQN_target = DQNModel(state_shape, num_actions, load_model_path=path, name='DQN_target')

            #Begin Testing
            rewards = []
            step_totals = []
            cp_totals = Counts()
            for ep in range(0, args.num_eps):

                # Reset environment and state buffer for next episode
                reset_env_and_state_buffer(env, state_buf, args)
                ep_reward = 0
                ep_totals = Counts()

                step = 0
                ep_done = False
                initial_steps = np.random.randint(1, args.max_initial_random_steps+1)
                
                while not ep_done:
                    if args.render:
                        env.render()
                    else:
                        env.render(mode='tiny_rgb_array')

                    actionID = get_actionID(step, initial_steps, epsilon)

                    observation, reward, terminal, _ = env.step(actionID, observation_mode='tiny_rgb_array')

                    grid = preprocess_observation(args, observation)
                    state_buf.add(grid)

                    step += 1
                    ep_reward += reward
                    ep_totals.reward_update(reward)

                    # Episode can finish either by reaching terminal state or max episode steps
                    if terminal or step == args.max_step:
                        cp_totals.update_all(ep_totals)
                        step_totals.append(step)

                        out_str = 'Test ep {:d}/{:d} \t Steps = {:d} \t Reward = {:.2f} \t\n\r'.format(ep + 1, args.num_eps, step, ep_reward, actionID)
                        #output(out_str, out_file)
                        print(out_str)

                        out_str =  ep_totals.get_str()
                        #output(out_str, out_file)
                        print(out_str)



                        rewards.append(ep_reward)
                        ep_done = True   

            mean_step = np.mean(step_totals)
            mean_reward = np.mean(rewards)
            error_reward = ss.sem(rewards)
            
            if not path:
                pathStr = "Beginning"
            else:
                pathStr = path

            out_str = pathStr + ' Checkpoint Testing complete \n\r'
            #output(out_str, out_file)
            print(out_str)

            out_str = 'Average reward = {:.2f} +/- {:.2f} /ep\t Average steps: {}\n\r'.format(mean_reward, error_reward, mean_step)
            #output(out_str, out_file)
            print(out_str)

            out_str = 'Totals: ' + cp_totals.get_str() + '\tEpsilon: ' + str(epsilon) + '\n\r\n\r'
            print(out_str)
            #output(out_str, out_file)

            out_str = '{},{},{:.2f},{:.2f},{:.2f},{},{},{},\n\r'.format(pathStr, epsilon, mean_reward, error_reward, mean_step, cp_totals.on, cp_totals.off, cp_totals.win)
            output(out_str, out_file)
    out_file.close()
    env.close()  


if  __name__ == '__main__':
    train_args = get_train_args()
    test_args = get_test_args(train_args)
    test(test_args)     