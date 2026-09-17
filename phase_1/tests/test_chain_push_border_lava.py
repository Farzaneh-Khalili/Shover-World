import numpy as np
import pytest
from environment import ShoverWorldEnv


def test_chain_push_into_border_lava():
    env = ShoverWorldEnv(
        n_rows=7,
        n_cols=7,
        number_of_boxes=0,
        number_of_barriers=0,
        number_of_lavas=0,
        border_lava=True,
        seed=0
    )

    env.agent_pos = np.array([3, 1])
    env.agent_dir = 2  # Right

    # chain of three boxed
    env.grid[3, 2] = 10
    env.grid[3, 3] = 10
    env.grid[3, 4] = 10

    #  env.grid[*, 6] == LAVA

    initial_stamina = env.stamina

    # STEP 1 (PUSH)
    obs, reward, done, info = env.step(2)  # PUSH RIGHT

    assert info["lava_destroyed_this_step"] == 0
    assert info["num_destroyed"] == 0

    expected_stamina_1 = initial_stamina - env.push_cost
    assert env.stamina == expected_stamina_1

    assert reward == 0

    # STEP 2 (PUSH)
    obs, reward, done, info = env.step(2)

    assert info["lava_destroyed_this_step"] == 1
    assert info["num_destroyed"] == 1

    expected_stamina_2 = expected_stamina_1 - env.push_cost
    assert env.stamina == expected_stamina_2

    assert reward == pytest.approx(
        env.destroy_reward * info["lava_destroyed_this_step"])

    # STEP 3 (PUSH)

    obs, reward, done, info = env.step(2)

    assert info["lava_destroyed_this_step"] == 1
    assert info["num_destroyed"] == 2

    expected_stamina_3 = expected_stamina_2 - env.push_cost
    assert env.stamina == expected_stamina_3

    assert reward == pytest.approx(
        env.destroy_reward * info["lava_destroyed_this_step"])

    #  STEP 4 (PUSH)

    obs, reward, done, info = env.step(2)

    assert info["lava_destroyed_this_step"] == 1
    assert info["num_destroyed"] == 3

    expected_stamina_4 = expected_stamina_3 - env.push_cost
    assert env.stamina == expected_stamina_4

    assert reward == pytest.approx(
        env.destroy_reward * info["lava_destroyed_this_step"])

    # no box left
    assert (env.grid == 10).sum() == 0
