'''
## Train ##
# Code to train Deep Q Network on gym-sokoban environment
@author: Kolin Guo
'''

from datetime import datetime
import json, os, sys, argparse, logging, random, time, shutil
import gym, gym_sokoban
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from utils.experience_replay import ReplayMemory
from utils.state_buffer import StateBuffer
from utils.network import DQNModel
from utils.utils import preprocess_observation, reset_env_and_state_buffer

def get_train_args():
    train_args = argparse.ArgumentParser()

    # Random Seed
    train_args.add_argument("--random_seed", type=int, default=1234, help="Random seed for reproducability")

    # Environment parameters
    train_args.add_argument("--env", type=str, default='Boxoban-Train-v0', help="Environment to use for training")
    train_args.add_argument("--num_surfaces", type=int, default=7, help="Number of room surfaces for one-hot encoding")
    train_args.add_argument("--max_step", type=int, default=200, help="Maximum number of steps in a single game episode")
    train_args.add_argument("--render", type=bool, default=False, help="Whether or not to display the environment during training")
    train_args.add_argument("--grid_width", type=int, default=10, help="Grid width")
    train_args.add_argument("--grid_height", type=int, default=10, help="Grid height")
    train_args.add_argument("--grids_per_state", type=int, default=4, help="Sequence of grids which constitutes a single state")

    # Environment rewards
    train_args.add_argument("--env_penalty_for_step", type=float, default=-0.1, help="Reward of performing a step")
    train_args.add_argument("--env_reward_box_on_target", type=float, default=10.0, help="Reward of pushing a box on target")
    train_args.add_argument("--env_penalty_box_off_target", type=float, default=-10.0, help="Reward of pushing a box off target")
    train_args.add_argument("--env_reward_finished", type=float, default=100.0, help="Reward of winning (pushed all boxes on targets)")

    # Training parameters
    train_args.add_argument("--save_tb_trace", type=bool, default=False, help="Save TensorBoard trace for first 1000 step")
    train_args.add_argument("--num_steps_train", type=int, default=50000000, help="Number of steps to train for")
    train_args.add_argument("--batch_size", type=int, default=32, help="Batch size of state transitions")
    train_args.add_argument("--learning_rate", type=float, default=0.00025, help="Learning rate")
    train_args.add_argument("--replay_mem_size", type=int, default=1000000, help="Maximum number of steps in replay memory buffer")
    train_args.add_argument("--initial_replay_mem_size", type=int, default=50000, help="Initial number of steps in replay memory (populated by random actions) before learning can start")
    train_args.add_argument("--epsilon_start", type=float, default=1.0, help="Exploration rate at the beginning of training")
    train_args.add_argument("--epsilon_end", type=float, default=0.1, help="Fixed exploration rate at the end of epsilon decay")
    train_args.add_argument("--epsilon_decay_step", type=int, default=1000000, help="After how many steps to stop decaying the exploration rate")
    train_args.add_argument("--discount_rate", type=float, default=0.99, help="Discount rate (gamma) for future rewards")
    train_args.add_argument("--update_target_step", type=int, default=10000, help="Copy current network parameters to target network every N steps")
    train_args.add_argument("--save_checkpoint_step", type=int, default=100000, help="Save checkpoint every N steps")
    train_args.add_argument("--save_log_step", type=int, default=1000, help="Save logs (training_time, avg_reward, num_episodes) every N steps")

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
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] [%(name)-12s] [%(levelname)-8s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=log_filepath,
                        filemode='w')

    # define a Handler which writes INFO messages or higher to the sys.stdout
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('[%(name)-12s] [%(levelname)-8s] %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # Get logger for training args
    logger = logging.getLogger('args')

    # Write the training arguments
    for key, value in vars(args).items():
        logger.info('{%s: %s}', key, value)

def train(args):
    ACTION_SPACE = np.array([1, 2, 3, 4], dtype=np.uint8)
    # Function to get a random actionID
    def sample_action_space():
        return random.choice(ACTION_SPACE)

    # Function to convert actionID (1, 2, 3, 4) to actionQID (0, 1, 2, 3)
    def actionID_to_actionQID(actionID):
        return actionID-1

    # Function to convert actionQID (0, 1, 2, 3) to actionID (1, 2, 3, 4)
    def actionQID_to_actionID(actionQID):
        return actionQID+1

    # Function to return epsilon based on current step
    def get_epsilon(current_step, epsilon_start, epsilon_end, epsilon_decay_step):
        if current_step < epsilon_decay_step:
            return epsilon_start + (epsilon_end - epsilon_start) / float(epsilon_decay_step) * current_step
        else:
            return epsilon_end

    # Get logger for training
    logger = logging.getLogger('train')

    # Check if GPU is available
    logger.info("Num GPUs Available: %d", len(tf.config.experimental.list_physical_devices('GPU')))

    # Create environment
    env = gym.make(args.env)
    num_actions = 4     # Push (up, down, left, right): 1, 2, 3, 4
    env.unwrapped.set_maxsteps(args.max_step)
    env.unwrapped.set_rewards(
            [args.env_penalty_for_step, 
                args.env_reward_box_on_target, 
                args.env_penalty_box_off_target, 
                args.env_reward_finished])

    # Set random seeds for reproducability
    random.seed(args.random_seed)
    env.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)

    # Initialize replay memory and state buffer
    replay_mem = ReplayMemory(args)
    state_buf = StateBuffer(args)

    # Check if resume from training
    load_model_path = None
    if args.checkpoint_file is not None:    # Resume training
        load_model_path = os.path.join(args.checkpoint_dir, args.checkpoint_file)
        assert os.path.exists(load_model_path+'.index'), 'Path "{}" does not exist!'.format(load_model_path+'.index')

        start_step = args.checkpoint_file.split('/')[-1].split('-')[-1]
        assert len(start_step)>0, "Invalid checkpoint file for extracting start_step"
        start_step = int(start_step)
    else:   # Train from scratch
        # Create another directory for this training
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.log_filename.split('.')[0])
        start_step = 0

    # Create checkpoint directory
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # Instantiate DQN and DQN_target
    state_shape = (args.grid_height, args.grid_width, args.num_surfaces, args.grids_per_state)
    DQN = DQNModel(state_shape, num_actions, args.learning_rate, load_model_path=load_model_path, name='DQN')
    DQN_target = DQNModel(state_shape, num_actions, load_model_path=load_model_path, name='DQN_target')

    ## Begin training
    env.reset()

    # Populate replay memory to initial_replay_mem_size
    logger.info("Populating replay memory with random actions...")

    for si in range(args.initial_replay_mem_size):
        if args.render:
            env.render(mode='human')
        else:
            env.render(mode='tiny_rgb_array')

        actionID = sample_action_space()
        observation, reward, terminal, _ = env.step(actionID, observation_mode='tiny_rgb_array')
        grid = preprocess_observation(args, observation)
        replay_mem.add(actionID, reward, grid, terminal)

        if terminal:
            env.reset()

        sys.stdout.write('\x1b[2K\rStep {:d}/{:d}'.format(si+1, args.initial_replay_mem_size))
        sys.stdout.flush()

    # Start training
    reward_one_episode = 0
    reward_episodes = []
    step_one_episode = 0
    step_episodes = []
    Qval_steps = []
    duration_steps = []

    # Create tf summary writer to write summaries to disk
    # ./logs/train/20200318_120026
    tf_train_log_dir = os.path.join(args.log_dir.replace('train', 'tf_train'), args.log_filename.split('.')[0])
    if not os.path.exists(tf_train_log_dir):
        os.makedirs(tf_train_log_dir)
    train_summary_writer = tf.summary.create_file_writer(tf_train_log_dir)
    train_summary_writer.set_as_default()
    if args.save_tb_trace:
        # Model graphs
        tf.summary.trace_on(graph=True, profiler=True)

    reset_env_and_state_buffer(env, state_buf, args)
    logger.info("Start training...")
    for si in range(start_step+1, args.num_steps_train+1):
        start_time = time.time()

        ## Playing Step
        # Perform a step
        if args.render:
            env.render(mode='human')
        else:
            env.render(mode='tiny_rgb_array')

        # Select a random action based on epsilon-greedy algorithm
        epsilon = get_epsilon(si, args.epsilon_start, args.epsilon_end, args.epsilon_decay_step)
        if random.random() < epsilon:   # Take random action
            actionID = sample_action_space()
        else:   # Take greedy action
            state = tf.convert_to_tensor(state_buf.get_state(), dtype=tf.float32)
            state = state[tf.newaxis, ...]      # Add an axis for batch
            actionQID = DQN.predict(state)
            actionID = actionQID_to_actionID(int(actionQID))    # convert from Tensor to int

        # Take the action and store state transition
        observation, reward, terminal, _ = env.step(actionID, observation_mode='tiny_rgb_array')
        grid = preprocess_observation(args, observation)
        state_buf.add(grid)
        replay_mem.add(actionID, reward, grid, terminal)
        # Accumulate reward and increment step
        reward_one_episode += reward
        step_one_episode += 1

        if terminal:
            # Save the accumulate reward for this episode
            reward_episodes.append(reward_one_episode)
            reward_one_episode = 0
            # Save the number of steps for this episode
            step_episodes.append(step_one_episode)
            step_one_episode = 0
            # Reset environment and state buffer
            reset_env_and_state_buffer(env, state_buf, args)

        ## Training Step
        # Sample a random minibatch of transitions from ReplayMemory
        states_batch, actionID_batch, rewards_batch, next_states_batch, terminals_batch = replay_mem.getMinibatch()
        actionQID_batch = actionID_to_actionQID(actionID_batch)
        # Infer DQN_target for Q(S', A)
        next_states_batch = tf.convert_to_tensor(next_states_batch, dtype=tf.float32)
        next_states_Qvals = DQN_target.infer(next_states_batch)
        max_next_states_Qvals = tf.math.reduce_max(next_states_Qvals, axis=1, name='maxQ')
        assert max_next_states_Qvals.shape == (args.batch_size,), "Wrong dimention for predicted next state Q vals"
        # Set Q(S', A) for all terminal state S'
        max_next_states_Qvals = tf.math.multiply(max_next_states_Qvals, np.invert(terminals_batch), name='remove_terminals')
        # Save average maximum predicted Q values
        Qval_steps.append(np.mean(max_next_states_Qvals[max_next_states_Qvals != 0]))
        # Calculate the traget Q values
        targetQs = rewards_batch + args.discount_rate * max_next_states_Qvals

        # Pass to DQN
        states_batch = tf.cast(states_batch, tf.float32)
        targetQs = tf.cast(targetQs, tf.float32)
        DQN.train_step(states_batch, actionQID_batch, targetQs)

        # Update DQN_target every args.update_target_step steps
        if si % args.update_target_step == 0:
            update_save_path = os.path.join(args.checkpoint_dir, 'DQN_Update')
            DQN.save_model(update_save_path)
            DQN_target.load_model(update_save_path)

        duration = time.time() - start_time
        duration_steps.append(duration)

        # Save log
        if si % args.save_log_step == 0:
            avg_training_loss = DQN.get_training_loss()

            logger.info("{Training Step: %d/%d}", si, args.num_steps_train)
            logger.info("Number of Episodes: %d", len(reward_episodes))
            logger.info("Recent Step Exploration Rate: %.5f", epsilon)
            logger.info("Average Per-Episode Reward: %.5f", sum(reward_episodes)/float(len(reward_episodes)))
            logger.info("Average Per-Episode Step: %.3f", sum(step_episodes)/float(len(step_episodes)))
            logger.info("Average Per-Step Maximum Predicted Q Value: %.8f", sum(Qval_steps)/float(len(Qval_steps)))
            logger.info("Average Per-Step Training Loss: %.8f", avg_training_loss)
            logger.info("Average Per-Step Training Time: %.5f second", sum(duration_steps)/float(len(duration_steps)))

            tf.summary.scalar('Episodes', len(reward_episodes), step=si, description='Number of Episodes')
            tf.summary.scalar('epsilon', epsilon, step=si, description='Recent Step Exploration Rate')
            tf.summary.scalar('avgReward', sum(reward_episodes)/float(len(reward_episodes)), step=si, description='Average Per-Episode Reward')
            tf.summary.scalar('avgStep', sum(step_episodes)/float(len(step_episodes)), step=si, description='Average Per-Episode Step Count')
            tf.summary.scalar('avgQval', sum(Qval_steps)/float(len(Qval_steps)), step=si, description='Average Per-Step Maximum Predicted Q Value')
            tf.summary.scalar('avgTrainLoss', avg_training_loss, step=si, description='Average Per-Step Training Loss')
            tf.summary.scalar('avgTrainTime', sum(duration_steps)/float(len(duration_steps)), step=si, description='Average Per-Step Training Time')

            if args.save_tb_trace:
                # Save computation graph
                tf.summary.trace_export(name="model_trace", step=si, profiler_outdir=tf_train_log_dir)

            # Reset the parameters
            reward_episodes = []
            step_episodes = []
            duration_steps = []
            Qval_steps = []

        # Save checkpoint
        if si % args.save_checkpoint_step == 0:
            save_checkpoint_path = os.path.join(args.checkpoint_dir, 
                    'DQN_Train')
            DQN.save_model(save_checkpoint_path, ckpt_number=si)
            # Duplicate the current logfile
            src_log_filepath = os.path.join(args.log_dir, args.log_filename)
            dst_log_filepath = os.path.join(args.checkpoint_dir, 
                    'DQN_Train_{}.log'.format(si))
            shutil.copyfile(src_log_filepath, dst_log_filepath)

    # Training finished
    logger.info("Finished training...")
    # Save trained network
    save_final_network_path = os.path.join(args.checkpoint_dir, 'DQN_Trained')
    DQN.save_model(save_final_network_path, ckpt_number=args.num_steps_train)

if __name__ == '__main__':
    # Change back to repository directory
    os.chdir(os.path.realpath(os.path.join(os.path.abspath(__file__), '../../')))
    train_args = get_train_args()
    log_train_args(train_args)
    train(train_args)