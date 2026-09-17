import numpy as np
import pytest
from environment import ShoverWorldEnv, BOX, BARRIER, LAVA


def test_push_into_lava():
    env = ShoverWorldEnv(grid_size=5)

    # Agent
    env.agent_pos = np.array([2, 1])
    env.agent_dir = 2  # Right

    # Box in front
    env.grid[2, 2] = BOX

    # Lava cell
    env.grid[2, 3] = LAVA

    obs, reward, done, info = env.step(2)  # PUSH

    # Box should be removed from its original place
    assert env.grid[2, 2] == 0

    # Lava must remain intact (-1)
    assert env.grid[2, 3] == LAVA

    # Counters
    assert info["num_destroyed"] == 1
    assert info["lava_destroyed_this_step"] == 1
