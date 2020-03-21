'''
## Utils ##
@author: Kolin Guo
'''

import numpy as np

# RGB
WALL              = [0, 0, 0]
FLOOR             = [243, 248, 238]
BOX_TARGET        = [254, 126, 125]
BOX_OFF_TARGET    = [142, 121, 56]
BOX_ON_TARGET     = [254, 95, 56]
PLAYER_OFF_TARGET = [160, 212, 56]
PLAYER_ON_TARGET  = [219, 212, 56]
# 7 possible surfaces
SURFACES = [WALL, FLOOR, BOX_TARGET, BOX_OFF_TARGET, BOX_ON_TARGET, PLAYER_OFF_TARGET, PLAYER_ON_TARGET]

# Convert a tiny_rgb_array observation into one-hot encoded grid representation
def preprocess_observation(args, observation):
    grid_dims = (args.grid_height, args.grid_width, args.num_surfaces)
    grid = np.zeros(grid_dims, dtype=np.uint8)

    for si in range(len(SURFACES)):
        grid[..., si] = np.all(observation == SURFACES[si], axis=2).astype(np.uint8)

    return grid

# Reset Gym-Sokoban environment and state buffer, initialize state buffer to first grid
def reset_env_and_state_buffer(env, state_buf, args):
    env.reset()
    state_buf.reset()

    observation = env.render(mode='tiny_rgb_array')
    grid = preprocess_observation(args, observation)

    for _ in range(args.grids_per_state):
        state_buf.add(grid)
