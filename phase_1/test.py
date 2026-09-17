import numpy as np
from environment import ShoverWorldEnv


def main():
    env = ShoverWorldEnv(
        render_mode=None,
        n_rows=6,
        n_cols=9,
        initial_force=4.0,
        unit_force=1.0,
        seed=0
    )

    obs = env.reset()
    done = False
    total_r = 0.0

    while not done:
        a = env.action_space.sample()
        obs, r, done, info = env.step(a)
        total_r += r

    print("Episode return:", total_r)
    env.close()


if __name__ == "__main__":
    main()
