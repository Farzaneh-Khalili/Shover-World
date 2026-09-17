from environment import ShoverWorldEnv
from heuristic_b import EnergyHeuristic
from player_ai import State
from subgoal import SubgoalManager


def make_env():
    """Create a small deterministic environment for heuristic tests."""
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


def test_basic_energy_heuristic():
    """The heuristic should return a numeric value for a valid state."""
    env = make_env()

    env.agent_pos = (3, 1)
    env.grid[2, 2] = 10

    state = make_state(env)
    subgoal_manager = SubgoalManager(env)
    heuristic = EnergyHeuristic(env, subgoal_manager)

    value = heuristic.evaluate(state)

    assert isinstance(value, (int, float))
    assert value == value  # rejects NaN


def test_heuristic_changes_with_box_count():
    """Adding a box should affect the heuristic value."""
    env = make_env()
    env.agent_pos = (3, 1)

    subgoal_manager = SubgoalManager(env)
    heuristic = EnergyHeuristic(env, subgoal_manager)
    state_without_box = make_state(env)
    value_without_box = heuristic.evaluate(state_without_box)

    env.grid[2, 2] = 10

    state_with_box = make_state(env)
    value_with_box = heuristic.evaluate(state_with_box)

    assert value_with_box != value_without_box


def test_heuristic_accounts_for_initial_stamina():
    """The heuristic should reflect the environment's initial stamina."""
    env = make_env()

    env.agent_pos = (3, 1)
    env.grid[2, 2] = 10

    state = make_state(env)

    env.initial_stamina = 1000
    subgoal_manager = SubgoalManager(env)
    heuristic_high = EnergyHeuristic(env, subgoal_manager)
    high_stamina_value = heuristic_high.evaluate(state)

    env.initial_stamina = 100
    subgoal_manager = SubgoalManager(env)
    heuristic_low = EnergyHeuristic(env, subgoal_manager)
    low_stamina_value = heuristic_low.evaluate(state)

    assert high_stamina_value != low_stamina_value


def test_heuristic_with_perfect_square():
    """A perfect square should affect the heuristic."""
    env = make_env()
    env.agent_pos = (3, 1)

    env.grid[1, 1] = 10
    env.grid[1, 2] = 10
    env.grid[2, 1] = 10
    env.grid[2, 2] = 10

    state = make_state(env)
    subgoal_manager = SubgoalManager(env)
    heuristic = EnergyHeuristic(env, subgoal_manager)
    value = heuristic.evaluate(state)

    assert isinstance(value, (int, float))
    assert value == value
