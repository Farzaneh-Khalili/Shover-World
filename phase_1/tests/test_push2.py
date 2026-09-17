
import numpy as np
import pytest
from environment import ShoverWorldEnv, BOX, BARRIER, LAVA


def test_push_single_box():
    env = ShoverWorldEnv(grid_size=5)

    # Place agent
    env.agent_pos = np.array([2, 1])
    env.agent_dir = 2  # Right

    # Place a chain box
    env.grid[2, 2] = 10

    obs, reward, done, info = env.step(2)  # PUSH

    # Box should move right
    assert env.grid[2, 3] == 10
    # Agent should move into old box position
    assert (env.agent_pos == np.array([2, 2])).all()
    assert info["num_destroyed"] == 0
    assert info["last_action_valid"] is True
