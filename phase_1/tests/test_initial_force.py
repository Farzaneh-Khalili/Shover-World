import numpy as np
import pytest
from environment import ShoverWorldEnv


def test_initial_force_stationary_logic():

    env = ShoverWorldEnv(
        grid_size=5,
        initial_force=10.0,
        unit_force=2.0,
        initial_stamina=200.0
    )

    env.grid.fill(0)

    # Agent
    env.agent_pos = np.array([2, 1])
    env.agent_dir = 2  # right

    # box
    env.grid[2, 2] = 10

    s0 = float(env.stamina)

    #  STEP 1
    _, _, _, info1 = env.step(2)   # PUSH right
    s1 = float(env.stamina)

    assert info1["initial_force_charged"] is True

    assert s1 == pytest.approx(s0 - (1 + 10 + 2))

    #  STEP 2
    _, _, _, info2 = env.step(2)
    s2 = float(env.stamina)

    assert info2["initial_force_charged"] is False

    assert s2 == pytest.approx(s1 - (1 + 2))

    #  STEP 3

    r, c = env.agent_pos
    env.grid[r - 1, c] = 10

    _, _, _, info3 = env.step(1)  # PUSH up
    s3 = float(env.stamina)

    assert info3["initial_force_charged"] is True

    assert s3 == pytest.approx(s2 - (1 + 10 + 2))
