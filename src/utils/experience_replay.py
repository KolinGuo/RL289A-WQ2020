'''
## Experience Replay ##
# ReplayMemory class
# Adapted from: https://github.com/msinto93/DQN_Atari/blob/master/utils/experience_replay.py
'''

import numpy as np
import random
import logging
# Get logger for replay memory
logger = logging.getLogger('replay_mem')

class ReplayMemory:
    def __init__(self, args):
        self.buffer_size = args.replay_mem_size
        self.min_buffer_size = args.initial_replay_mem_size
        self.dims = (args.grid_height, args.grid_width, args.num_surfaces)

        # Preallocate replay memory
        self.actions = np.empty(self.buffer_size, dtype=np.uint8)
        self.rewards = np.empty(self.buffer_size, dtype=np.float64)
        self.grids = np.empty((self.dims[0], self.dims[1], self.dims[2], self.buffer_size), dtype=np.uint8)
        self.terminals = np.empty(self.buffer_size, dtype=np.bool)

        # Replay memory config
        self.grids_per_state = args.grids_per_state
        self.batch_size = args.batch_size
        self.count = 0
        self.current = 0

        # Preallocate transition states for minibatch
        self.states = np.empty((self.batch_size, self.dims[0], self.dims[1], self.dims[2], self.grids_per_state), dtype=np.uint8)
        self.next_states = np.empty((self.batch_size, self.dims[0], self.dims[1], self.dims[2], self.grids_per_state), dtype=np.uint8)

        logger.info('Initializing a ReplayMemory of size %d', self.buffer_size)

    def add(self, action, reward, grid, terminal):
        assert grid.shape == self.dims, "Grids must be of same shape"

        # Note: grid is S_{t+1}, after action and reward
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.grids[..., self.current] = grid
        self.terminals[self.current] = terminal

        # Increment count and current idx
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.buffer_size

        #logger.debug('{Current/Count: %d/%d} {Action/Reward/Terminal: %d/%.1f/%s}', self.current, self.count, action, reward, terminal)

    def getState(self, idx):
        # Returns a state consisting of grid at idx and 3 previous grids
        return self.grids[..., (idx - (self.grids_per_state-1)):(idx+1)]

    def getMinibatch(self):
        # Memory much include at least grids_per_state
        assert self.count > self.grids_per_state, "Replay memory must contain more grids than the desired number of grids per state"
        # Memory should be initially populated with random actions up to 'min_buffer_size'
        assert self.count >= self.min_buffer_size, "Replay memory does not contain enough samples to start learning, take random actions to populate replay memory"

        # Sample random indices
        indices = []
        while len(indices) < self.batch_size:
            # Find random index
            while True:
                # sample one idx (ignore states wraping over)
                idx = random.randint(self.grids_per_state, self.count - 1)
                # if wraps over current pointer, get a new one
                if idx >= self.current and idx - self.grids_per_state < self.current:
                    continue
                # if wraps over episode end, get a new one
                # The last grid (idx) can be terminal
                if self.terminals[(idx - self.grids_per_state):idx].any():
                    continue
                # Index is ok to use
                break
            # Put the sampled state and next_state into minibatch
            self.states[len(indices), ...] = self.getState(idx - 1)
            self.next_states[len(indices), ...] = self.getState(idx)
            indices.append(idx)

        actions = self.actions[indices]
        rewards = self.rewards[indices]
        terminals = self.terminals[indices]

        return self.states, actions, rewards, self.next_states, terminals

if __name__ == '__main__':
    ### For testing ###
    import argparse
    args = argparse.ArgumentParser()

    args.add_argument("--num_surfaces", type=int, default=7, help="Number of room states for one-hot encoding")
    args.add_argument("--grid_height", type=int, default=13, help="Grid height")
    args.add_argument("--grid_width", type=int, default=11, help="Grid width")
    args.add_argument("--grids_per_state", type=int, default=4, help="Sequence of grids which constitutes a single state")
    args.add_argument("--batch_size", type=int, default=5, help="Batch size of state transitions")
    args.add_argument("--replay_mem_size", type=int, default=100, help="Maximum number of steps in replay memory buffer")
    args.add_argument("--initial_replay_mem_size", type=int, default=50, help="Initial number of steps in replay memory (populated by random actions) before learning can start")
    args = args.parse_args()

    mem = ReplayMemory(args)

    for i in range(0, 60):
        grid = np.random.choice([False, True], size=(args.grid_height, args.grid_width, args.num_surfaces))
        action = np.random.randint(4) + 1
        reward = np.random.randint(2)
        terminal = np.random.choice([False, False, False, False, False, False, False, False, True])

        mem.add(action, reward, grid, terminal)

    states, actions, rewards, next_states, terminals = mem.getMinibatch()
    assert states.shape == (args.batch_size, args.grid_height, args.grid_width, args.num_surfaces, args.grids_per_state), 'states shape error'
    assert actions.shape == (args.batch_size,), 'actions shape error'
    assert rewards.shape == (args.batch_size,), 'rewards shape error'
    assert next_states.shape == (args.batch_size, args.grid_height, args.grid_width, args.num_surfaces, args.grids_per_state), 'next_states shape error'
    assert terminals.shape == (args.batch_size,), 'terminals shape error'
    print('All assertion passed!')
