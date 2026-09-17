# planning/heuristic_b.py

import copy

import numpy as np

from utils import state_to_env


class EnergyHeuristic:
    def __init__(self, env, subgoal_manager):
        self.env = env
        self.subgoals = subgoal_manager

    def evaluate(self, state, subgoal=None):
        """Evaluate the current state using the energy-based heuristic."""
        env = state_to_env(state, self.env)
        grid = env.grid
        agent_r, agent_c = env.agent_pos
        stamina = env.stamina

        num_boxes = int(np.sum((grid > 0) & (grid < 100)))

        lava_positions = np.argwhere(grid < 0)
        if len(lava_positions) == 0:
            min_dist_to_lava = 0
        else:
            min_dist_to_lava = int(
                np.min(
                    np.abs(lava_positions[:, 0] - agent_r)
                    + np.abs(lava_positions[:, 1] - agent_c)
                )
            )

        # Chain length in 4 directions (push potential proxy)
        chain_length = 0
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            r, c = agent_r + dr, agent_c + dc
            k = 0

            while (
                0 <= r < grid.shape[0]
                and 0 <= c < grid.shape[1]
                and 0 < grid[r, c] < 100
            ):
                k += 1
                r += dr
                c += dc

            chain_length = max(chain_length, k)

        # Safe perfect-square count (no mutation of env or grid)
        if getattr(env, "perf_square_manager", None) is not None:
            try:
                psm_copy = copy.deepcopy(env.perf_square_manager)
                grid_copy = grid.copy()
                squares = psm_copy.detect(grid_copy)
                num_squares = len(squares)
            except Exception:
                num_squares = 0
        else:
            num_squares = 0

        # Base heuristic
        h_value = (
            num_boxes
            - 0.01 * stamina
            - 0.2 * chain_length
            - 0.5 * num_squares
            + 0.1 * min_dist_to_lava
        )

        # Subgoal shaping
        if subgoal is not None and isinstance(subgoal, dict):
            stype = subgoal.get("type", None)

            # Encourage getting near a specific box
            if stype == "approach_box":
                br, bc = subgoal["box_pos"]
                dist_agent_box = abs(agent_r - br) + abs(agent_c - bc)
                h_value += 0.25 * dist_agent_box

            # Encourage states where the target box is nearer to lava
            elif stype == "push_to_lava":
                br, bc = subgoal["box_pos"]

                if len(lava_positions) > 0:
                    dist_box_lava = int(
                        np.min(
                            np.abs(lava_positions[:, 0] - br)
                            + np.abs(lava_positions[:, 1] - bc)
                        )
                    )
                    h_value += 0.5 * dist_box_lava

                # Also encourage the agent to get near that box to push it
                dist_agent_box = abs(agent_r - br) + abs(agent_c - bc)
                h_value += 0.1 * dist_agent_box

            # If we are in "use perfect square" mode, reward squares more strongly
            elif stype == "use_perfect_square":
                h_value -= 1.5 * num_squares

        return float(h_value)
