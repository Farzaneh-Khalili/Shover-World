from utils import state_to_env
from subgoal import SubgoalManager
from heuristic_b import EnergyHeuristic
from environment import ShoverWorldEnv
import numpy as np
from typing import Any, FrozenSet, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import heapq
import copy
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


@dataclass(frozen=True)
class State:
    agent_pos: Tuple[int, int]
    agent_dir: int
    boxes: FrozenSet[Tuple[int, int]]


@dataclass(order=True)
class Node:
    f: float
    state: State = field(compare=False)
    g: float = field(compare=False)
    h: float = field(compare=False)
    parent: Optional["Node"] = field(compare=False, default=None)
    action: Optional[Any] = field(compare=False, default=None)


def env_to_state(env) -> State:
    """Convert the environment state into the search state representation."""
    boxes = set()
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            if 1 <= env.grid[r][c] <= 10:
                boxes.add((r, c))

    return State(
        agent_pos=env.agent_pos,
        agent_dir=env.agent_dir,
        boxes=frozenset(boxes),
    )


def get_successors(state: State, env_template: ShoverWorldEnv):
    """Generate valid successor states for all available actions."""
    successors = []

    for action in range(7):
        env = state_to_env(state, env_template)
        obs, reward, done, info = env.step(action)

        if not info.get("last_action_valid", True):
            continue

        next_state = env_to_state(env)
        cost = -reward

        successors.append((next_state, action, cost))

    return successors


def is_corner(box, env):
    r, c = box
    walls = 0

    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc

        if not (0 <= nr < env.n_rows and 0 <= nc < env.n_cols):
            walls += 1
        elif env.grid[nr][nc] in (-100, 100):
            walls += 1

    return walls >= 2


def is_goal(state):
    return len(state.boxes) == 0


def astar(env: ShoverWorldEnv, start_state: State):
    """Run A* search from the given initial state."""
    open_list = []
    closed = set()
    expanded_nodes = 0
    max_open_size = 0

    subgoal_manager = SubgoalManager(env)
    energy_heuristic = EnergyHeuristic(env, subgoal_manager)

    h0 = energy_heuristic.evaluate(
        start_state,
        subgoal_manager.select_subgoal(start_state)
    )

    heapq.heappush(
        open_list,
        Node(
            f=h0,
            state=start_state,
            g=0,
            h=h0,
            parent=None,
            action=None
        )
    )

    while open_list:
        current = heapq.heappop(open_list)
        expanded_nodes += 1

        if is_goal(current.state):
            print("Goal reached!")
            print("Expanded nodes:", expanded_nodes)
            print("Max open list size:", max_open_size)
            return current

        if current.state in closed:
            continue

        closed.add(current.state)

        for next_state, action, cost in get_successors(current.state, env):
            if next_state in closed:
                continue

            g = current.g + cost
            subgoal = subgoal_manager.select_subgoal(next_state)
            h = energy_heuristic.evaluate(next_state, subgoal)
            f = g + h

            heapq.heappush(
                open_list,
                Node(
                    f=f,
                    state=next_state,
                    g=g,
                    h=h,
                    parent=current,
                    action=action
                )
            )

        max_open_size = max(max_open_size, len(open_list))

    print("Search failed")
    print("Expanded nodes:", expanded_nodes)
    print("Max open list size:", max_open_size)
    return None


def extract_plan(goal_node: Node):
    """Extract the action sequence from the goal node."""
    actions = []
    node = goal_node

    while node.parent is not None:
        actions.append(node.action)
        node = node.parent

    actions.reverse()
    return actions


if __name__ == "__main__":
    MAP_PATH = Path(__file__).resolve().parent / "maps" / "map8.txt"

    START_POS = (7, 1)
    START_DIR = 0

    env = ShoverWorldEnv(
        map_path=MAP_PATH
    )

    print(
        "Initial box count:",
        np.sum((env.grid > 0) & (env.grid < 100))
    )

    print(
        "Boxes positions:",
        list(zip(*np.where((env.grid > 0) & (env.grid < 100))))
    )

    print(
        "Perfect squares:",
        env.perf_square_manager.detect(env.grid.copy())
    )

    env.agent_pos = START_POS
    env.agent_dir = START_DIR

    start_state = env_to_state(env)

    print(
        "State boxes:",
        len(start_state.boxes),
        sorted(start_state.boxes)[:20]
    )

    print("Agent:", env.agent_pos)

    valid = []

    for a in range(7):
        e2 = copy.deepcopy(env)
        _, _, _, info = e2.step(a)

        if info.get("last_action_valid", True):
            valid.append(a)

    print("Valid actions from start:", valid)

    def any_push_to_lava_possible(env):
        g = env.grid
        R, C = g.shape

        for r in range(R):
            for c in range(C):
                if 0 < g[r, c] < 100:
                    for dr, dc in [
                        (-1, 0),
                        (1, 0),
                        (0, -1),
                        (0, 1)
                    ]:
                        rr, cc = r, c

                        while (
                            0 <= rr < R
                            and 0 <= cc < C
                            and (0 < g[rr, cc] < 100)
                        ):
                            rr += dr
                            cc += dc

                        if (
                            0 <= rr < R
                            and 0 <= cc < C
                            and g[rr, cc] < 0
                        ):
                            return True

        return False

    print(
        "Any push-to-lava possible (ignoring reachability)?",
        any_push_to_lava_possible(env)
    )

    goal_node = astar(env, start_state)

    if goal_node:
        plan = extract_plan(goal_node)

        print("Plan length:", len(plan))
        print("Plan:", plan)

        print("\n--- Replaying plan ---")

        replay_env = ShoverWorldEnv(
            map_path=MAP_PATH,
            border_lava=True
        )

        replay_env.agent_pos = START_POS
        replay_env.agent_dir = START_DIR

        total_reward = 0

        for step_idx, action in enumerate(plan):
            obs, r, done, info = replay_env.step(action)
            total_reward += r

            if not info.get("last_action_valid", True):
                print(f"Invalid action at step {step_idx}: {action}")
                print(info)
                break

            if done:
                break

        boxes_left = int(
            np.sum(
                (replay_env.grid > 0)
                & (replay_env.grid < 100)
            )
        )

        print("Replay finished")
        print("Boxes left:", boxes_left)
        print("Destroyed boxes:", replay_env.num_destroyed)
        print("Total reward:", total_reward)

    else:
        print("No solution found")
