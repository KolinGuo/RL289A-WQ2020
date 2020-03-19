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
    test_params.add_argument("--env_reward_box_on_target", type=float, default=10.0, help="Reward of pushing a box on target")
    test_params.add_argument("--env_penalty_box_off_target", type=float, default=-10.0, help="Reward of pushing a box off target")
    test_params.add_argument("--env_reward_finished", type=float, default=100.0, help="Reward of winning (pushed all boxes on targets)")
    
    # Testing parameters
    test_params.add_argument("--num_eps", type=int, default=20, help="Number of episodes to test for")
    test_params.add_argument("--max_initial_random_steps", type=int, default=10, help="Maximum number of random steps to take at start of episode to ensure random starting point")
    test_params.add_argument("--epsilon_value", type=float, default=0.05, help="Exploration rate for the play")

    # Files/directories
    test_params.add_argument("--checkpoint_dir", type=str, default='./ckpts', help="Directory for saving/loading checkpoints")
    test_params.add_argument("--checkpoint_file", type=str, default=None, help="Checkpoint file to load (if None, load latest ckpt)")
    test_params.add_argument("--results_dir", type=str, default='./test_results', help="Directory for saving txt file of results")
    results_filename = datetime.now().strftime("%Y%m%d_%H%M%S_results.txt")
    test_params.add_argument("--results_file", type=str, default=results_filename, help="Text file of test results (if None, do not save results), (current timestamp) DON'T MODIFY")
    
    return test_params.parse_args()


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

    # Create environment
    env = gym.make(args.env)
    num_actions = env.action_space.n
    #env.unwrapped.set_maxsteps(args.max_step) #TODO may not need
    #env.unwrapped.set_rewards(
    #    [args.env_penalty_for_step, 
    #        args.env_reward_box_on_target, 
    #        args.env_penalty_box_off_target, 
    #        args.env_reward_finished])

    # Set random seeds for reproducability
    env.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)

    state_buf = StateBuffer(args)
    state_shape = (args.grid_height, args.grid_width, args.num_surfaces, args.grids_per_state)
    load_model_path = None


    # Resume from checkpoint
    if args.checkpoint_file is not None:    
        load_model_path = os.path.join(args.checkpoint_dir, args.checkpoint_file)

    DQN_target = DQNModel(state_shape, num_actions, load_model_path=load_model_path, name='DQN_target')

    #Begin Testing
    rewards = []  
    for ep in range(0, args.num_eps):

        # Reset environment and state buffer for next episode
        reset_env_and_state_buffer(env, state_buf, args)
        ep_reward = 0
        step = 0
        ep_done = False
        initial_steps = np.random.randint(1, args.max_initial_random_steps+1)
        sys.stdout.write('\n')   
        sys.stdout.flush()
        
        while not ep_done:
            if args.render:
                env.render()
            else:
                env.render(mode='tiny_rgb_array')

            #Choose random action for initial steps to ensure every episode has a random start point. Then choose action with highest Q-value according to network's current policy.
            if step < initial_steps:
                actionID = sample_action_space()
            else:
                if random.random() < args.epsilon_value:   # Take random action
                    actionID = sample_action_space()
                    print("Random Action\n")
                else:   # Take greedy action
                    state = tf.convert_to_tensor(state_buf.get_state(), dtype=tf.float32)
                    state = state[tf.newaxis, ...]      # Add an axis for batch
                    actionQID = DQN_target.predict(state)
                    actionID = actionQID_to_actionID(int(actionQID))    # convert from Tensor to int
                    print("Greedy Action\n")

            observation, reward, terminal, _ = env.step(actionID, observation_mode='tiny_rgb_array')
            grid = preprocess_observation(args, observation)
            state_buf.add(grid)

            step += 1
            ep_reward += reward

            sys.stdout.write('\x1b[2K\rTest ep {:d}/{:d} \t Steps = {:d} \t Reward = {:.2f} \t'.format(ep, args.num_eps, step, ep_reward, actionID))
            sys.stdout.flush() 

            # Episode can finish either by reaching terminal state or max episode steps
            if terminal or step == args.max_step:
                rewards.append(ep_reward)
                ep_done = True   

    mean_reward = np.mean(rewards)
    error_reward = ss.sem(rewards)
            
    sys.stdout.write('\n\nTesting complete \t Average reward = {:.2f} +/- {:.2f} /ep \n\n'.format(mean_reward, error_reward))
    sys.stdout.flush() 

    # Log average episode reward for Tensorboard visualisation
    #summary_str = sess.run(summary_op, {reward_var: mean_reward}) #TODO change tensor stuff
    #summary_writer.add_summary(summary_str, train_ep) #TODO need?
     
    # Write results to file        
    #if args.results_file is not None:
        #if not os.path.exists(args.results_dir):
            #os.makedirs(args.results_dir)
        #output_file = open(args.results_dir + '/' + args.results_file, 'a')
        #output_file.write('Training Episode {}: \t Average reward = {:.2f} +/- {:.2f} /ep \n\n'.format(ep, mean_reward, error_reward))
        #output_file.flush()
        #sys.stdout.write('Results saved to file \n\n')
        #sys.stdout.flush()      
    
    env.close()  


if  __name__ == '__main__':
    train_args = get_train_args()
    test_args = get_test_args(train_args)
    test(test_args)     