

import numpy as np
import pytest
from environment import ShoverWorldEnv, BOX, BARRIER, LAVA


def test_chain_push_and_blocking():
    # scenario: 1x5 grid, agent at col0, boxes at 1,2,3 and empty at 4
    env = ShoverWorldEnv(render_mode=None, n_rows=1, n_cols=5,
                         initial_stamina=1000, initial_force=5, unit_force=2)
    env.grid = np.array([[0, BOX, BOX, BOX, 0]], dtype=np.int32)
    env.agent_pos = (0, 0)
    env.timestep = 0

    # action: push to the right (z=2).
    obs, r, done, info = env.step([0, 1, 2])
    assert info["last_action_valid"] is True
    assert info["chain_length_k"] == 3
    assert env.agent_pos == (0, 1)
    np.testing.assert_array_equal(env.grid, np.array(
        [[0, 0, BOX, BOX, BOX]], dtype=np.int32))

    # blocking case: tail would hit barrier -> push invalid, grid unchanged
    env2 = ShoverWorldEnv(render_mode=None, n_rows=1,
                          n_cols=5, initial_stamina=1000)
    env2.grid = np.array([[0, BOX, BOX, BOX, BARRIER]], dtype=np.int32)
    env2.agent_pos = (0, 0)
    obs2, r2, done2, info2 = env2.step([0, 1, 2])
    assert info2["last_action_valid"] is False
    np.testing.assert_array_equal(env2.grid, np.array(
        [[0, BOX, BOX, BOX, BARRIER]], dtype=np.int32))
