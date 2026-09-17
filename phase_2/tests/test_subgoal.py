import numpy as np

from environment import ShoverWorldEnv
from player_ai import State
from subgoal import SubgoalManager


def make_env():
    """Create a small deterministic environment for subgoal tests."""
    env = ShoverWorldEnv(
        n_rows=6,
        n_cols=6,
        number_of_boxes=0,
        number_of_barriers=0,
        number_of_lavas=0,
        border_lava=False,
        seed=42,
    )

    env.grid.fill(0)
    return env


def make_state(env):
    """Convert the current environment configuration to the AI State."""
    boxes = frozenset(
        (r, c)
        for r in range(env.n_rows)
        for c in range(env.n_cols)
        if 1 <= env.grid[r][c] <= 99
    )

    return State(
        agent_pos=env.agent_pos,
        agent_dir=env.agent_dir,
        boxes=boxes,
    )


def test_push_to_lava_subgoal():
    """A box adjacent to lava should produce a push-to-lava subgoal."""
    env = make_env()

    env.agent_pos = (3, 1)
    env.grid[2, 2] = 10
    env.grid[2, 3] = -100

    manager = SubgoalManager(env)
    state = make_state(env)

    subgoal = manager.select_subgoal(state)

    assert subgoal is not None
    assert subgoal["type"] == "push_to_lava"


def test_use_perfect_square_subgoal():
    """A perfect square of boxes should be detected as a subgoal."""
    env = make_env()

    env.agent_pos = (3, 1)

    # 2x2 perfect square
    env.grid[1, 1] = 10
    env.grid[1, 2] = 10
    env.grid[2, 1] = 10
    env.grid[2, 2] = 10

    manager = SubgoalManager(env)
    state = make_state(env)

    subgoal = manager.select_subgoal(state)

    assert subgoal is not None
    assert subgoal["type"] == "use_perfect_square"


def test_approach_box_subgoal():
    """When no higher-priority structure exists, approach a box."""
    env = make_env()

    env.agent_pos = (4, 1)
    env.grid[1, 4] = 10

    manager = SubgoalManager(env)
    state = make_state(env)

    subgoal = manager.select_subgoal(state)

    assert subgoal is not None
    assert subgoal["type"] == "approach_box"


def test_subgoal_priority():
    """
    A box adjacent to lava has higher priority than simply
    approaching another box.
    """
    env = make_env()

    env.agent_pos = (4, 1)

    # Box A: adjacent to lava
    env.grid[2, 2] = 10
    env.grid[2, 3] = -100

    # Box B: farther away
    env.grid[4, 4] = 10

    manager = SubgoalManager(env)
    state = make_state(env)

    subgoal = manager.select_subgoal(state)

    assert subgoal is not None
    assert subgoal["type"] == "push_to_lava"
