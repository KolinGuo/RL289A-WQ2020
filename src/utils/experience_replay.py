'''
## Experience Replay ##
# Adapted from: https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py
'''

import numpy as np
import random

class ReplayMemory:
    def __init__(self, args):
        self.buffer_size = args.replay_mem_size
        self.min_buffer_size = args.initial_replay_mem_size

        self.actions = np.empty(self.buffer_size, dtype=np.uint8)
        self.rewards = np.empty(self.buffer_size, dtype=np.float64)
        self.steps = np.empty((self.buffer_size, args.grid_height))

