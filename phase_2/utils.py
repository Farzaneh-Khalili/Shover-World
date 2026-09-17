# utils.py

from environment import ShoverWorldEnv


def state_to_env(state, env_template):
    """Reconstruct an environment from a given search state."""
    env = ShoverWorldEnv(
        n_rows=env_template.n_rows,
        n_cols=env_template.n_cols,
        initial_stamina=env_template.initial_stamina,
        initial_force=env_template.initial_force,
        unit_force=env_template.unit_force,
        border_lava=env_template.border_lava,
    )

    env.grid = env_template.grid.copy()

    for r in range(env.n_rows):
        for c in range(env.n_cols):
            if 1 <= env.grid[r][c] <= 10:
                env.grid[r][c] = 0

    for r, c in state.boxes:
        env.grid[r][c] = 10

    env.agent_pos = state.agent_pos
    env.agent_dir = state.agent_dir
    env.stamina = env_template.initial_stamina

    return env
