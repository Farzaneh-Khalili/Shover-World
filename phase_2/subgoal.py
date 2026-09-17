# planning/subgoal.py

import numpy as np

from utils import state_to_env


class SubgoalManager:
    """
    Generates and manages subgoals based on the current environment state.
    Designed specifically for ShoverWorldEnv.
    """

    def __init__(self, env):
        """Initialize the subgoal manager with a template environment."""
        self.env = env

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def generate_subgoals(self, state_dict):
        """
        Generate subgoals based on the current state.

        Args:
            state_dict: Environment-like dictionary containing:
                - "grid": np.ndarray
                - "agent": (r, c)

        Returns:
            List of subgoal dictionaries sorted by priority.
        """
        grid = state_dict["grid"]
        agent_pos = tuple(state_dict["agent"])

        subgoals = []

        # 1) Push box into lava (highest priority)
        lava_boxes = self._find_boxes_adjacent_to_lava(grid)

        for box_pos in lava_boxes:
            subgoals.append({
                "type": "push_to_lava",
                "box_pos": box_pos,
                "priority": 0
            })

        # 2) Use perfect square (if available)
        squares = self._detect_perfect_squares(grid)

        if len(squares) > 0:
            subgoals.append({
                "type": "use_perfect_square",
                "squares": squares,
                "priority": 1
            })

        # 3) Approach nearest box (fallback)
        nearest_box = self._find_nearest_box(grid, agent_pos)

        if nearest_box is not None:
            subgoals.append({
                "type": "approach_box",
                "box_pos": nearest_box,
                "priority": 2
            })

        # Sort subgoals by priority (lower is better)
        subgoals.sort(key=lambda g: g.get("priority", 10))

        return subgoals

    def select_subgoal(self, state):
        """
        Return the highest-priority subgoal for a given State.

        The grid is rebuilt from the given State using state_to_env()
        to avoid retaining outdated box positions from the template map.
        """
        env = state_to_env(state, self.env)

        state_dict = {
            "grid": env.grid.copy(),
            "agent": env.agent_pos
        }

        subgoals = self.generate_subgoals(state_dict)

        return subgoals[0] if subgoals else None

    def is_subgoal_reached(self, state, subgoal):
        """
        Check whether a given subgoal has been achieved.

        Accepts either:
            - an environment-like dictionary: {"grid": ..., "agent": ...}
            - a State object with agent_pos and boxes attributes
        """
        if subgoal is None:
            return True

        if isinstance(state, dict):
            grid = state["grid"]
            agent = state["agent"]
        else:
            env = state_to_env(state, self.env)
            grid = env.grid
            agent = env.agent_pos

        if subgoal["type"] == "push_to_lava":
            r, c = subgoal["box_pos"]
            return not self._is_box(grid[r, c])

        if subgoal["type"] == "approach_box":
            agent_r, agent_c = agent
            box_r, box_c = subgoal["box_pos"]

            return abs(agent_r - box_r) + abs(agent_c - box_c) == 1

        if subgoal["type"] == "use_perfect_square":
            squares = self._detect_perfect_squares(grid)
            return len(squares) == 0

        return False

    # --------------------------------------------------
    # Helper functions
    # --------------------------------------------------

    def _is_box(self, val):
        return (val > 0) and (val < 100)

    def _find_boxes_adjacent_to_lava(self, grid):
        rows, cols = grid.shape
        result = []

        for r in range(rows):
            for c in range(cols):
                if self._is_box(grid[r, c]):
                    for dr, dc in [
                        (-1, 0),
                        (1, 0),
                        (0, -1),
                        (0, 1)
                    ]:
                        nr, nc = r + dr, c + dc

                        if 0 <= nr < rows and 0 <= nc < cols:
                            if grid[nr, nc] < 0:
                                result.append((r, c))
                                break

        return result

    def _find_nearest_box(self, grid, agent_pos):
        agent_r, agent_c = agent_pos
        boxes = np.argwhere((grid > 0) & (grid < 100))

        if len(boxes) == 0:
            return None

        distances = [
            (
                abs(agent_r - r) + abs(agent_c - c),
                (int(r), int(c))
            )
            for r, c in boxes
        ]

        distances.sort(key=lambda x: x[0])

        return distances[0][1]

    def _detect_perfect_squares(self, grid):
        if getattr(self.env, "perf_square_manager", None) is None:
            return []

        try:
            squares = self.env.perf_square_manager.detect(grid.copy())
            return [
                (int(n), int(r), int(c))
                for (n, r, c, age) in squares
            ]
        except Exception:
            return []
