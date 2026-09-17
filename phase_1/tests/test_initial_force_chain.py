import numpy as np
import pytest
from environment import ShoverWorldEnv


def test_chain_push_energy_costs():
    env = ShoverWorldEnv(
        grid_size=7,
        initial_force=10.0,
        unit_force=2.0,
        initial_stamina=500.0
    )

    env.grid.fill(0)

    # Agent looking right
    env.agent_pos = np.array([3, 1])
    env.agent_dir = 2  # right

    # Chain of 3 boxes
    env.grid[3, 2] = 10
    env.grid[3, 3] = 10
    env.grid[3, 4] = 10

    s0 = float(env.stamina)

    # STEP 1
    _, _, _, info1 = env.step(2)  # PUSH right
    s1 = float(env.stamina)

    assert info1["chain_length_k"] == 3
    assert info1["initial_force_charged"] is True

    # baseline(1) + initial_force(10) + unit_force(2)*3
    expected_drop1 = 1 + 10 + 2*3
    assert s1 == pytest.approx(s0 - expected_drop1)

    # STEP 2

    _, _, _, info2 = env.step(2)
    s2 = float(env.stamina)

    assert info2["chain_length_k"] == 3
    assert info2["initial_force_charged"] is False

    expected_drop2 = 1 + 2*3
    assert s2 == pytest.approx(s1 - expected_drop2)

    # STEP 3
    env.agent_dir = 0  # up

    r, c = env.agent_pos
    env.grid[r-1, c] = 10

    _, _, _, info3 = env.step(1)  # PUSH up
    s3 = float(env.stamina)

    assert info3["chain_length_k"] == 1
    assert info3["initial_force_charged"] is True

    expected_drop3 = 1 + 10 + 2*1
    assert s3 == pytest.approx(s2 - expected_drop3)
