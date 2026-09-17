
import numpy as np
import pytest
from environment import ShoverWorldEnv, BOX, BARRIER, LAVA


def test_push_cost_and_initial_force_charged():
    env = ShoverWorldEnv(render_mode=None, n_rows=1, n_cols=3,
                         initial_stamina=100.0, initial_force=5.0, unit_force=2.0)
    # grid: [empty, box, empty]
    env.grid = np.array([[0, BOX, 0]], dtype=np.int32)
    env.agent_pos = (0, 0)
    obs, r, done, info = env.step([0, 1, 2])  # push right
    # expected: stamina = 100 - baseline(1) - initial_force(5) - unit_force(2*1)
    expected = 100.0 - 1.0 - 5.0 - 2.0
    assert pytest.approx(env.stamina, rel=1e-6) == expected
    assert info["initial_force_charged"] is True
    assert info["chain_length_k"] == 1


def test_lava_refund():
    env = ShoverWorldEnv(render_mode=None, n_rows=1, n_cols=3,
                         initial_stamina=50.0, initial_force=5.0, unit_force=2.0)
    # grid: [empty, box, lava] -> pushing right destroys box into lava
    env.grid = np.array([[0, BOX, LAVA]], dtype=np.int32)
    env.agent_pos = (0, 0)
    obs, r, done, info = env.step([0, 1, 2])
    # cost: baseline -1, initial_force charged and then refunded (+5),
    # and unit_force*1 charged -> net change = -1 - unit_force
    expected = 50.0 - 1.0 - 2.0
    assert pytest.approx(env.stamina, rel=1e-6) == expected
    assert info["lava_destroyed_this_step"] == 1
