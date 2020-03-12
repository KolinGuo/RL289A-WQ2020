'''
## State Buffer ##
# A state consists of multiple grids. StateBuffer maintains the last 'grids_per_state' in a buffer.
# Adapted from: https://github.com/msinto93/DQN_Atari/blob/master/utils/state_buffer.py
'''
import numpy as np
import logging
# Get logger for state buffer
logger = logging.getLogger('state_buf')

class StateBuffer:
    def __init__(self, args):
        self.grids_per_state = args.grids_per_state
        self.dims = (args.grid_height, args.grid_width, args.num_surfaces)
        self.buffer = np.zeros(self.dims + (self.grids_per_state,), dtype=np.uint8)

        logger.info('Initializing a StateBuffer of size %d', self.grids_per_state)

    def add(self, grid):
        assert grid.shape == self.dims, "Grids must be of same shape"
        self.buffer[..., :-1] = self.buffer[..., 1:]
        self.buffer[..., -1] = grid

    def reset(self):
        self.buffer *= 0

    def get_state(self):
        return self.buffer
