import numpy as np
import pytest
from environment import ShoverWorldEnv, BOX, BARRIER, LAVA


def test_agent_move():
    env = ShoverWorldEnv(grid_size=5)
    env.agent_pos = np.array([2, 2])
    env.agent_dir = 1  # Up

    obs, reward, done, info = env.step(1)

    assert (env.agent_pos == np.array([1, 2])).all()
    assert reward == 0
    assert done is False
    assert info["last_action_valid"] is True


def test_blocked_by_wall():
    env = ShoverWorldEnv(grid_size=5)
    env.agent_pos = np.array([0, 2])  # Edge
    env.agent_dir = 1  # Up

    obs, reward, done, info = env.step(1)

    assert (env.agent_pos == np.array([0, 2])).all()  # No movement
    assert info["last_action_valid"] is False
