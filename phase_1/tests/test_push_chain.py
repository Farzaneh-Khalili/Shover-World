import numpy as np
from environment import ShoverWorldEnv, BOX, LAVA, EMPTY


def test_chain_push_into_lava():
    env = ShoverWorldEnv(grid_size=7)

    # clear any randomness and set up deterministic grid
    env.grid.fill(EMPTY)

    # Agent at (3,1) looking right
    env.agent_pos = np.array([3, 1])
    env.agent_dir = 2  # right

    # Chain: boxes at (3,2), (3,3), (3,4)
    env.grid[3, 2] = BOX
    env.grid[3, 3] = BOX
    env.grid[3, 4] = BOX

    # Lava at (3,5)
    env.grid[3, 5] = LAVA

    # --- Step 1: first PUSH (should destroy the right-most box immediately) ---
    obs, reward, done, info = env.step(2)  # PUSH right
    assert info["chain_length_k"] == 3
    assert info["lava_destroyed_this_step"] == 1
    assert info["num_destroyed"] == 1
    # after push: boxes should be at cols 3 and 4 (from old 2->3, 3->4), col2 empty
    assert env.grid[3, 2] == EMPTY
    assert env.grid[3, 3] == BOX
    assert env.grid[3, 4] == BOX
    # lava stays lava
    assert env.grid[3, 5] == LAVA
    # agent moved one step forward
    assert (env.agent_pos == np.array([3, 2])).all()

    # --- Step 2: second PUSH (destroy next box) ---
    obs, reward, done, info = env.step(2)
    assert info["chain_length_k"] == 2
    assert info["lava_destroyed_this_step"] == 1
    assert info["num_destroyed"] == 2
    # after push: remaining box should have shifted, check grid
    assert env.grid[3, 3] == EMPTY
    assert env.grid[3, 4] == BOX
    assert env.grid[3, 5] == LAVA
    assert (env.agent_pos == np.array([3, 3])).all()

    # --- Step 3: third PUSH (destroy last box) ---
    obs, reward, done, info = env.step(2)
    assert info["chain_length_k"] == 1
    assert info["lava_destroyed_this_step"] == 1
    assert info["num_destroyed"] == 3
    # now no boxes remain
    assert (env.grid == BOX).sum() == 0
    assert env.grid[3, 5] == LAVA
    assert (env.agent_pos == np.array([3, 4])).all()
