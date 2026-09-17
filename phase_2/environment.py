# environment.py
"""
ShoverWorldEnv - Milestone 1 (Environment Specifications)

Fixes applied:
1) Boxes are treated consistently as ANY value 1..99 (i.e., 0 < v < BARRIER).
   - All places that previously used (grid == BOX) now use a consistent box-count.
2) Done condition, info["num_boxes"], HUD render now reflect all boxes (1..99), not only value 10.
3) No other behavior changes (pushing, stamina, rewards, perfect-square logic remain as-is).
"""

import os
import random
import time
from typing import Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

try:
    import pygame
except Exception:
    pygame = None


# Cell encodings
LAVA = -100
EMPTY = 0
BOX = 10
BARRIER = 100


class PerfectSquareManager:
    """Detect, track, and manage perfect squares in the environment."""

    def __init__(self, perf_sq_initial_age=10):
        self.age_dict = {}
        self.perf_sq_initial_age = int(perf_sq_initial_age)

    # -------------------------
    # Box predicate (MATCH ENV)
    # -------------------------
    @staticmethod
    def is_box(v):
        # Any positive < 100 is a box
        return (v > 0) and (v < 100)

    # -------------------------
    # Core checks
    # -------------------------
    @staticmethod
    def check_all_boxes(grid, r, c, n):
        rows = len(grid)
        cols = len(grid[0])

        if r + n > rows or c + n > cols:
            return False

        for i in range(r, r + n):
            for j in range(c, c + n):
                if not PerfectSquareManager.is_box(grid[i][j]):
                    return False

        return True

    @staticmethod
    def check_perimeter_no_boxes(grid, r, c, n):
        """
        Ensure the 1-cell thick border around the n×n square has no boxes.
        """
        rows = len(grid)
        cols = len(grid[0])

        R1 = max(r - 1, 0)
        C1 = max(c - 1, 0)
        R2 = min(r + n, rows - 1)
        C2 = min(c + n, cols - 1)

        for i in range(R1, R2 + 1):
            for j in range(C1, C2 + 1):
                inside = (r <= i < r + n) and (c <= j < c + n)

                if inside:
                    continue

                if PerfectSquareManager.is_box(grid[i][j]):
                    return False

        return True

    # -------------------------
    # Detect perfect squares
    # -------------------------
    @staticmethod
    def detect_perfect_squares(grid, max_size=None):
        rows = len(grid)
        cols = len(grid[0])

        if max_size is None:
            max_size = min(rows, cols)

        found = []

        for r in range(rows):
            for c in range(cols):
                if not PerfectSquareManager.is_box(grid[r][c]):
                    continue

                # Try sizes from 2..max_size
                for n in range(2, max_size + 1):
                    if r + n > rows or c + n > cols:
                        break

                    # MUST be full n×n boxes
                    if not PerfectSquareManager.check_all_boxes(
                        grid, r, c, n
                    ):
                        break

                    # Optional perimeter rule
                    if PerfectSquareManager.check_perimeter_no_boxes(
                        grid, r, c, n
                    ):
                        found.append({
                            "n": n,
                            "top_left": (r, c)
                        })

        return found

    # -------------------------
    # Age Tracking
    # -------------------------
    @staticmethod
    def update_square_ages(prev_ages, current_squares):
        new_ages = {}

        current_keys = [
            (
                (sq["top_left"][0], sq["top_left"][1]),
                sq["n"]
            )
            for sq in current_squares
        ]

        for key in current_keys:
            new_ages[key] = prev_ages.get(key, -1) + 1

        return new_ages

    @staticmethod
    def get_oldest_square(age_dict):
        if not age_dict:
            return None

        return max(
            age_dict.items(),
            key=lambda item: item[1]
        )[0]

    @staticmethod
    def dissolve_old_squares(
        grid,
        age_dict,
        perf_sq_initial_age
    ):
        to_delete = []

        for key, age in age_dict.items():
            if age >= perf_sq_initial_age:
                (r, c), n = key

                for i in range(r, r + n):
                    for j in range(c, c + n):
                        grid[i][j] = 0

                to_delete.append(key)

        for key in to_delete:
            del age_dict[key]

    # -------------------------
    # Public API used by environment
    # -------------------------
    def detect(self, grid):
        """Detect current perfect squares and update their ages."""
        squares = self.detect_perfect_squares(grid)

        # Update ages based on REAL squares
        self.age_dict = self.update_square_ages(
            self.age_dict,
            squares
        )

        # Dissolve old squares
        self.dissolve_old_squares(
            grid,
            self.age_dict,
            self.perf_sq_initial_age
        )

        result = []

        for (r, c), n in self.age_dict.keys():
            age = self.age_dict[((r, c), n)]
            result.append((n, r, c, age))

        return result

    def apply_barrier_maker(self, grid):
        # Safety: only allow if there is at least one REAL perfect square
        # right now.
        _ = self.detect(grid)

        key = self.get_oldest_square(self.age_dict)

        if key is None:
            return False

        (r, c), n = key

        for i in range(r, r + n):
            for j in range(c, c + n):
                grid[i][j] = 100

        return True, n

    def apply_hellify(self, grid):
        _ = self.detect(grid)

        key = self.get_oldest_square(self.age_dict)

        if key is None:
            return False

        (r, c), n = key

        if n <= 2:
            return False

        for i in range(r, r + n):
            for j in range(c, c + n):
                if (
                    r < i < r + n - 1
                    and c < j < c + n - 1
                ):
                    grid[i][j] = -100
                else:
                    grid[i][j] = 0

        return True, n


class ShoverWorldEnv(gymnasium.Env):
    """Shover World grid environment."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        n_rows: int = 10,
        n_cols: int = 10,
        max_timestep: int = 400,
        number_of_boxes: int = 8,
        number_of_barriers: int = 3,
        number_of_lavas: int = 2,
        initial_stamina: float = 1000.0,
        initial_force: float = 40.0,
        unit_force: float = 10.0,
        perf_sq_initial_age: int = 10,
        map_path: Optional[str] = None,
        seed: Optional[int] = None,
        grid_size=10,
        stamina=1000,
        border_lava: bool = True,
        time_penalty_ms=200
    ):
        super().__init__()

        self.border_lava = bool(border_lava)
        self.grid_size = grid_size
        self.stamina_max = stamina

        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.render_mode = render_mode
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.max_timestep = int(max_timestep)

        self.initial_stamina = float(initial_stamina)
        self.initial_force = float(initial_force)
        self.unit_force = float(unit_force)
        self.perf_sq_initial_age = int(perf_sq_initial_age)

        self.map_path = map_path

        self.action_space = spaces.MultiDiscrete(
            [self.n_rows, self.n_cols, 7]
        )

        self.observation_space = spaces.Dict({
            "grid": spaces.Box(
                low=np.int32(-1000),
                high=np.int32(1000),
                shape=(self.n_rows, self.n_cols),
                dtype=np.int32
            ),
            "agent": spaces.Box(
                low=0,
                high=max(self.n_rows, self.n_cols) - 1,
                shape=(2,),
                dtype=np.int32
            ),
            "agent_dir": spaces.Discrete(5),
            "stamina": spaces.Box(
                low=np.float32(-1e9),
                high=np.float32(1e9),
                shape=(),
                dtype=np.float32
            ),
            "previous_selected_position": spaces.Box(
                low=-1,
                high=max(self.n_rows, self.n_cols) - 1,
                shape=(2,),
                dtype=np.int32
            ),
            "previous_action": spaces.Discrete(7)
        })

        self.grid = np.zeros(
            (self.n_rows, self.n_cols),
            dtype=np.int32
        )

        self.agent_pos: Tuple[int, int] = (0, 0)
        self.agent_dir = 0
        self.timestep = 0
        self.stamina = float(self.initial_stamina)
        self.previous_selected_position = (-1, -1)
        self.previous_action = 0
        self.num_destroyed = 0

        self.last_action_cost = 0.0
        self.push_cost = 0.0
        self.destroy_reward = float(self.initial_force)
        self.lava_destroyed_this_step = 0

        self.non_stationary_current = set()
        self.non_stationary_next = set()

        self.perf_square_manager = None

        if self.map_path:
            self._load_map_from_path(self.map_path)
        else:
            self._random_fill(
                number_of_boxes,
                number_of_barriers,
                number_of_lavas
            )

        self.original_grid = np.copy(self.grid)
        self.original_agent_pos = tuple(self.agent_pos)
        self.original_agent_dir = int(self.agent_dir)

        if self.border_lava:
            self._add_border_lava()

        self._init_renderer()

        self.perf_square_manager = PerfectSquareManager(
            perf_sq_initial_age=self.perf_sq_initial_age
        )

        self.time_penalty_ms = time_penalty_ms
        self.last_step_time = None

    # -------------------------
    # Consistent box helpers
    # -------------------------
    def _is_box(self, val: int) -> bool:
        return (val > 0) and (val < BARRIER)

    def _num_boxes(self) -> int:
        """Return the number of boxes currently on the grid."""
        return int(
            np.sum(
                (self.grid > 0)
                & (self.grid < BARRIER)
            )
        )

    # -------------------------
    # Map creation / loading
    # -------------------------
    def _random_fill(
        self,
        n_boxes: int,
        n_barriers: int,
        n_lavas: int
    ):
        self.grid.fill(EMPTY)

        free = [
            (r, c)
            for r in range(self.n_rows)
            for c in range(self.n_cols)
        ]

        random.shuffle(free)

        ar, ac = free.pop()
        self.agent_pos = (ar, ac)

        for _ in range(min(n_boxes, len(free))):
            r, c = free.pop()
            self.grid[r, c] = BOX

        for _ in range(min(n_barriers, len(free))):
            r, c = free.pop()
            self.grid[r, c] = BARRIER

        for _ in range(min(n_lavas, len(free))):
            r, c = free.pop()
            self.grid[r, c] = LAVA

    def _load_map_from_path(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Map file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            lines = [
                ln.rstrip("\n")
                for ln in f
                if ln.strip()
            ]

        if len(lines) == 0:
            return

        first_tokens = lines[0].split()
        is_int_format = True

        for tok in first_tokens:
            try:
                int(tok)
            except Exception:
                is_int_format = False
                break

        if is_int_format:
            grid_rows = []

            for ln in lines:
                row = [int(t) for t in ln.split()]
                grid_rows.append(row)

            arr = np.array(grid_rows, dtype=np.int32)
            h, w = arr.shape

            self.n_rows = h
            self.n_cols = w
            self.grid = np.zeros(
                (self.n_rows, self.n_cols),
                dtype=np.int32
            )
            self.grid[:h, :w] = arr

            # If agent is out of bounds or on a box, relocate.
            ar, ac = self.agent_pos

            if (
                not (
                    0 <= ar < self.n_rows
                    and 0 <= ac < self.n_cols
                )
                or self._is_box(self.grid[ar, ac])
            ):
                free = list(
                    zip(*np.where(self.grid == EMPTY))
                )
                self.agent_pos = (
                    tuple(random.choice(free))
                    if free
                    else (0, 0)
                )

        else:
            # Symbolic format
            sym_map = []
            found_agent = False
            agent_loc = None

            for r, ln in enumerate(lines):
                toks = ln.split()
                row = []

                for c, ch in enumerate(toks):
                    if ch == '.':
                        row.append(EMPTY)
                    elif ch == 'B':
                        row.append(BOX)
                    elif ch == '#':
                        row.append(BARRIER)
                    elif ch == 'L':
                        row.append(LAVA)
                    elif ch == 'A':
                        row.append(EMPTY)
                        found_agent = True
                        agent_loc = (r, c)
                    else:
                        try:
                            row.append(int(ch))
                        except Exception:
                            row.append(EMPTY)

                sym_map.append(row)

            arr = np.array(sym_map, dtype=np.int32)
            h, w = arr.shape

            self.n_rows = h
            self.n_cols = w
            self.grid = np.zeros(
                (self.n_rows, self.n_cols),
                dtype=np.int32
            )
            self.grid[:h, :w] = arr

            if found_agent and agent_loc is not None:
                self.agent_pos = agent_loc
            else:
                free = list(
                    zip(*np.where(self.grid == EMPTY))
                )
                self.agent_pos = (
                    tuple(random.choice(free))
                    if free
                    else (0, 0)
                )

    def _add_border_lava(self):
        self.grid[0, :] = LAVA
        self.grid[self.n_rows - 1, :] = LAVA
        self.grid[:, 0] = LAVA
        self.grid[:, self.n_cols - 1] = LAVA

        ar, ac = self.agent_pos

        if (
            not (
                0 <= ar < self.n_rows
                and 0 <= ac < self.n_cols
            )
            or self.grid[ar, ac] < 0
        ):
            inner = [
                (r, c)
                for r in range(1, self.n_rows - 1)
                for c in range(1, self.n_cols - 1)
                if self.grid[r, c] == EMPTY
            ]

            if inner:
                self.agent_pos = tuple(random.choice(inner))
            else:
                free = list(
                    zip(*np.where(self.grid == EMPTY))
                )

                if free:
                    self.agent_pos = tuple(random.choice(free))
                else:
                    cr, cc = (
                        self.n_rows // 2,
                        self.n_cols // 2
                    )
                    self.grid[cr, cc] = EMPTY
                    self.agent_pos = (cr, cc)

    # -------------------------
    # Gym API
    # -------------------------
    def reset(self):
        """Reset the environment to its original state."""
        self.grid = np.copy(self.original_grid)
        self.agent_pos = tuple(self.original_agent_pos)
        self.agent_dir = int(self.original_agent_dir)

        self.timestep = 0
        self.stamina = float(self.initial_stamina)
        self.previous_selected_position = (-1, -1)
        self.previous_action = 0
        self.num_destroyed = 0
        self.non_stationary_current = set()
        self.non_stationary_next = set()
        self.last_action_cost = 0.0
        self.push_cost = 0.0
        self.lava_destroyed_this_step = 0

        if self.border_lava:
            self._add_border_lava()

        self.last_step_time = time.time()

        return self._get_obs()

    def _get_obs(self):
        return {
            "grid": self.grid.copy(),
            "agent": np.array(
                self.agent_pos,
                dtype=np.int32
            ),
            "agent_dir": int(self.agent_dir),
            "stamina": np.float32(self.stamina),
            "previous_selected_position": np.array(
                self.previous_selected_position,
                dtype=np.int32
            ),
            "previous_action": int(self.previous_action),
        }

    def step(self, action):
        """Apply an action and return the resulting environment state."""
        now = time.time()

        if self.last_step_time is None:
            elapsed_ms = 0
        else:
            elapsed_ms = (
                now - self.last_step_time
            ) * 1000

        self.last_step_time = now

        time_units = int(
            elapsed_ms // self.time_penalty_ms
        )

        if time_units > 0:
            self.stamina -= time_units

        if isinstance(action, int):
            r_sel, c_sel, z = -1, -1, action
        else:
            r_sel = int(action[0])
            c_sel = int(action[1])
            z = int(action[2])

        self.timestep += 1
        self.previous_selected_position = (r_sel, c_sel)
        self.previous_action = int(z)

        info = {
            "timestep": self.timestep,
            "stamina": self.stamina,
            "num_boxes": self._num_boxes(),
            "num_destroyed": int(self.num_destroyed),
            "last_action_valid": False,
            "chain_length_k": 0,
            "initial_force_charged": False,
            "lava_destroyed_this_step": 0,
            "perfect_squares_available": [],
        }

        reward = 0.0
        baseline_cost = 1.0
        self.stamina -= baseline_cost
        reward_squares = 0

        self.push_cost = 0.0
        self.last_action_cost = 0.0
        self.lava_destroyed_this_step = 0

        if self.perf_square_manager is not None:
            try:
                ps = self.perf_square_manager.detect(self.grid)

                info["perfect_squares_available"] = [
                    (int(n), int(r), int(c))
                    for (n, r, c, age) in ps
                ]
            except Exception:
                info["perfect_squares_available"] = []

        # Movement actions (1..4)
        if z in (1, 2, 3, 4):
            dr, dc = {
                1: (-1, 0),
                2: (0, 1),
                3: (1, 0),
                4: (0, -1)
            }[z]

            ar, ac = self.agent_pos
            expected_r, expected_c = (
                ar + dr,
                ac + dc
            )

            if r_sel != -1 and c_sel != -1:
                if (
                    not (
                        0 <= r_sel < self.n_rows
                        and 0 <= c_sel < self.n_cols
                    )
                    or (
                        r_sel != expected_r
                        or c_sel != expected_c
                    )
                ):
                    done = self._check_done()
                    info["last_action_valid"] = False
                    info["stamina"] = float(self.stamina)

                    return (
                        self._get_obs(),
                        reward,
                        done,
                        info
                    )

            ok, k, lava_destroyed, front_box_stationary = (
                self._handle_movement(dr, dc)
            )

            dir_map = {
                (-1, 0): 1,
                (0, 1): 2,
                (1, 0): 3,
                (0, -1): 4
            }

            self.agent_dir = dir_map.get(
                (dr, dc),
                self.agent_dir
            )

            net = self._apply_stamina_cost(
                z,
                k,
                front_box_stationary,
                lava_destroyed
            )

            if k > 0:
                total_cost = baseline_cost + net
                self.push_cost = total_cost
                self.last_action_cost = net
                self.lava_destroyed_this_step = lava_destroyed
            else:
                self.push_cost = 0.0

            if lava_destroyed > 0:
                reward += (
                    self.destroy_reward
                    * lava_destroyed
                )

            info["last_action_valid"] = ok
            info["chain_length_k"] = k
            info["lava_destroyed_this_step"] = lava_destroyed
            info["initial_force_charged"] = (
                front_box_stationary
            )

        else:
            reward_squares = 0

            # Special actions 5=BarrierMaker, 6=Hellify
            if self.perf_square_manager is None:
                info["last_action_valid"] = False
            else:
                if z == 5:
                    res = (
                        self.perf_square_manager
                        .apply_barrier_maker(self.grid)
                    )

                    if res:
                        info["last_action_valid"] = True

                        if (
                            isinstance(res, tuple)
                            and len(res) >= 2
                        ):
                            try:
                                n = int(res[1])
                                self.stamina += n * n
                                reward_squares += 10 * (n ** 2)
                            except Exception:
                                pass

                elif z == 6:
                    res = (
                        self.perf_square_manager
                        .apply_hellify(self.grid)
                    )

                    if res:
                        n = int(res[1])
                        reward_squares += 10 * (n ** 2)
                        info["last_action_valid"] = True

        self.non_stationary_current = set(
            self.non_stationary_next
        )
        self.non_stationary_next = set()

        info["stamina"] = float(self.stamina)
        info["num_boxes"] = self._num_boxes()
        info["num_destroyed"] = int(self.num_destroyed)

        reward += reward_squares

        done = self._check_done()

        return self._get_obs(), reward, done, info

    # -------------------------
    # Movement & pushing
    # -------------------------
    def _handle_movement(self, dr: int, dc: int):
        """Handle movement and box-pushing in the given direction."""
        ar, ac = self.agent_pos
        nr, nc = ar + dr, ac + dc

        if not (
            0 <= nr < self.n_rows
            and 0 <= nc < self.n_cols
        ):
            return False, 0, 0, False

        if self.grid[nr, nc] == BARRIER:
            return False, 0, 0, False

        if self.grid[nr, nc] == EMPTY:
            self.agent_pos = (nr, nc)
            return True, 0, 0, False

        if self.grid[nr, nc] < 0:
            return False, 0, 0, False

        if self._is_box(self.grid[nr, nc]):
            chain = []
            r, c = nr, nc

            while (
                0 <= r < self.n_rows
                and 0 <= c < self.n_cols
                and self._is_box(self.grid[r, c])
            ):
                chain.append((r, c))
                r += dr
                c += dc

            beyond_r, beyond_c = r, c

            if not (
                0 <= beyond_r < self.n_rows
                and 0 <= beyond_c < self.n_cols
            ):
                return False, len(chain), 0, False

            if (
                self.grid[beyond_r, beyond_c] == BARRIER
                or self._is_box(self.grid[beyond_r, beyond_c])
            ):
                return False, len(chain), 0, False

            lava_destroyed = 0
            moved_new_positions = []

            for cr, cc in reversed(chain):
                new_r, new_c = cr + dr, cc + dc

                if self.grid[new_r, new_c] < 0:
                    self.grid[cr, cc] = EMPTY
                    lava_destroyed += 1
                    self.num_destroyed += 1
                else:
                    self.grid[new_r, new_c] = self.grid[cr, cc]
                    self.grid[cr, cc] = EMPTY
                    moved_new_positions.append(
                        (new_r, new_c)
                    )

            self.agent_pos = (nr, nc)
            k = len(chain)

            head_r, head_c = chain[0]
            dir_key = (
                (head_r, head_c),
                (dr, dc)
            )

            was_non_stationary = (
                dir_key in self.non_stationary_current
            )
            initial_force_charged = not was_non_stationary

            for pos in moved_new_positions:
                self.non_stationary_next.add(
                    (pos, (dr, dc))
                )

            return (
                True,
                k,
                lava_destroyed,
                initial_force_charged
            )

        return False, 0, 0, False

    # -------------------------
    # Stamina cost helper
    # -------------------------
    def _apply_stamina_cost(
        self,
        action_type,
        chain_length,
        front_box_stationary,
        lava_destroyed=0
    ) -> float:
        cost = 0.0

        if (
            action_type in [1, 2, 3, 4]
            and chain_length > 0
        ):
            if front_box_stationary:
                cost += (
                    self.initial_force
                    + self.unit_force * chain_length
                )
            else:
                cost += (
                    self.unit_force
                    * chain_length
                )

        refund = (
            lava_destroyed
            * self.initial_force
        )

        net = cost - refund

        self.stamina -= net
        self.last_action_cost = net
        self.lava_destroyed_this_step = lava_destroyed

        return net

    # -------------------------
    # Termination
    # -------------------------
    def _check_done(self):
        """Check whether the current episode should terminate."""
        if self.timestep >= self.max_timestep:
            return True

        if self.stamina <= 0:
            return True

        if self._num_boxes() == 0:
            return True

        return False

    # -------------------------
    # Rendering (pygame)
    # -------------------------
    def _init_renderer(self):
        if (
            self.render_mode == "human"
            and pygame is not None
        ):
            pygame.init()

            self.cell_size = 36
            width = self.n_cols * self.cell_size
            height = (
                self.n_rows * self.cell_size
                + 40
            )

            self.screen = pygame.display.set_mode(
                (width, height)
            )
            pygame.display.set_caption("Shover-World")

            self.font = pygame.font.SysFont(
                None,
                20
            )
            self.clock = pygame.time.Clock()
        else:
            self.screen = None

    def render(self):
        if self.screen is None:
            return

        self.screen.fill((220, 220, 220))

        for r in range(self.n_rows):
            for c in range(self.n_cols):
                rect = pygame.Rect(
                    c * self.cell_size,
                    r * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )

                val = self.grid[r, c]

                if val == EMPTY:
                    color = (245, 245, 245)
                elif self._is_box(val):
                    color = (170, 110, 60)
                elif val == BARRIER:
                    color = (50, 50, 50)
                elif val < 0:
                    color = (200, 50, 50)
                else:
                    color = (120, 120, 120)

                pygame.draw.rect(
                    self.screen,
                    color,
                    rect
                )
                pygame.draw.rect(
                    self.screen,
                    (180, 180, 180),
                    rect,
                    1
                )

        ar, ac = self.agent_pos

        center = (
            int(
                ac * self.cell_size
                + self.cell_size / 2
            ),
            int(
                ar * self.cell_size
                + self.cell_size / 2
            )
        )

        pygame.draw.circle(
            self.screen,
            (40, 120, 200),
            center,
            int(self.cell_size * 0.35)
        )

        # Box count uses _num_boxes()
        hud = self.font.render(
            (
                f"t={self.timestep}  "
                f"stamina={int(self.stamina)}  "
                f"boxes={self._num_boxes()}"
            ),
            True,
            (0, 0, 0)
        )

        self.screen.blit(
            hud,
            (5, self.n_rows * self.cell_size + 6)
        )

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if pygame is not None:
            pygame.quit()


if __name__ == "__main__":
    env = ShoverWorldEnv(
        n_rows=5,
        n_cols=5,
        number_of_boxes=0,
        number_of_barriers=0,
        number_of_lavas=0,
        seed=0,
        border_lava=False
    )

    env.grid.fill(0)
    env.agent_pos = (2, 1)
    env.grid[2, 2] = 1
    env.grid[2, 3] = -100

    print("Before step:")
    print(env.grid)

    obs, r, done, info = env.step(2)

    print("After step:")
    print(env.grid)
    print("Info:", info)
    print("push_cost (exposed):", env.push_cost)
