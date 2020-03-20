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
#%matplotlib inline
from IPython import display
from train import get_train_args
from utils.state_buffer import StateBuffer
from utils.network import DQNModel
from utils.utils import preprocess_observation, reset_env_and_state_buffer
from PIL import Image


def get_play_args(train_args):
    play_args = argparse.ArgumentParser()

    # Environment parameters (First 4 params must be same as those used in training)
    play_args.add_argument('--env', '-e', metavar='env', help='Environment to load (default: Boxoban-Val-v0)', default='Boxoban-Val-v0')
    play_args.add_argument("--num_surfaces", type=int, default=7, help="Number of room surfaces for one-hot encoding")
    play_args.add_argument("--grid_width", type=int, default=10, help="Grid width")
    play_args.add_argument("--grid_height", type=int, default=10, help="Grid height")
    play_args.add_argument("--grids_per_state", type=int, default=4, help="Sequence of grids which constitutes a single state")

    # Play parameters
    play_args.add_argument("--num_eps", type=int, default=5, help="Number of episodes to run for")
    play_args.add_argument("--max_ep_length", type=int, default=200, help="Maximum number of steps per episode")
    play_args.add_argument("--max_initial_random_steps", type=int, default=4, help="Maximum number of random steps to take at start of episode to ensure random starting point")
    play_args.add_argument("--epsilon_value", type=float, default=0.05, help="Exploration rate for the play")

    # Files/directories
    play_args.add_argument("--checkpoint_dir", type=str, default='./checkpoints', help="Directory for saving/loading checkpoints")
    play_args.add_argument("--checkpoint_file", type=str, default=None, help="Checkpoint file to load and resume training from (if None, train from scratch)")

    #Render
    play_args.add_argument('--gifs', default=True, action='store_true', help='Generate Gif files from images')
    play_args.add_argument('--save', default=True, action='store_true', help='Save images of single steps')

    return play_args.parse_args()



def play(args):
    generate_gifs = args.gifs
    save_images = args.save or args.gifs
    print(save_images)
    render_mode = 'rgb_array'
    scale_image = 16

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

    # Creating target directory if images are to be stored
    if save_images and not os.path.exists('images'):
        try:
            os.makedirs('images')
            os.chdir('images')
            os.makedirs('steps')
            os.makedirs('gif')

        except OSError:
            print('Error: Creating images target directory. ')
    else:
        try:
            os.chdir('images')
            if not os.path.exists('steps'):
                try:
                    os.makedirs('steps')
                except OSError:
                    print('Error: Creating steps target directory. ')


            if not os.path.exists('gif'):
                try:
                    os.makedirs('gif')
                except OSError:
                    print('Error: Creating gif target directory. ')

        except OSError:
            print('Error: Entering images target directory. ')


    if args.checkpoint_file is not None:    # Resume training
        load_model_path = os.path.join(args.checkpoint_dir, args.checkpoint_file)

    DQN_target = DQNModel(state_shape, num_actions, args.learning_rate, load_model_path=load_model_path, name='DQN')

    for ep in range(0, args.num_eps):
        # Reset environment and state buffer for next episode
        reset_env_and_state_buffer(env, state_buf, args)
        step = 0
        ep_done = False
        initial_steps = np.random.randint(1, args.max_initial_random_steps+1)

        while not ep_done:
            time.sleep(0.05)
            img = env.render(mode = render_mode)
            plt.imshow(img)
            display.clear_output(wait=True)
            display.display(plt.gcf())
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

            if save_images:
                img = Image.fromarray(np.array(env.render(render_mode, scale=scale_image)), 'RGB')
                img.save(os.path.join('steps', 'observation_{}_{}.png'.format(ep, step)))

            # Episode can finish either by reaching terminal state or max episode steps
            if terminal or step == args.max_ep_length:
                ep_done = True

            if generate_gifs:
                print('')
                import imageio

                with imageio.get_writer(os.path.join('gif', 'episode_{}.gif'.format(ep)), mode='I', fps=1) as writer:

                    for t in range(args.max_ep_length):
                        try:
                            filename = os.path.join('steps', 'observation_{}_{}.png'.format(ep, t))
                            image = imageio.imread(filename)
                            writer.append_data(image)
                        except:
                            pass


if  __name__ == '__main__':
    train_args = get_train_args()
    play_args = get_play_args(train_args)
    play(play_args)
