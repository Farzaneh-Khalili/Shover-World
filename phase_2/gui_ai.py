from player_ai import env_to_state, astar, extract_plan
from environment import ShoverWorldEnv
import os
import sys
import pygame

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def run_gui(env, plan_provider=None, ai_step_delay_ms=120):
    """
    Run the Shover World GUI with manual and AI controls.

    Args:
        env: ShoverWorldEnv instance.
        plan_provider: Function with signature plan_provider(env) -> list[int].
        ai_step_delay_ms: Delay between AI actions in milliseconds.
    """
    pygame.init()

    WIDTH = 800
    HEIGHT = 600
    CELL_SIZE = 40

    rows = env.n_rows
    cols = env.n_cols
    total_reward = 0.0

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Shover World (Manual + AI)")

    font = pygame.font.SysFont(None, 24)
    clock = pygame.time.Clock()

    # ---------- AI control state ----------
    ai_mode = False
    plan = []
    plan_idx = 0
    last_ai_step_time = 0
    # -------------------------------------

    running = True

    while running:
        now_ms = pygame.time.get_ticks()

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                # -------- Toggle AI mode --------
                if event.key == pygame.K_p:
                    if plan_provider is None:
                        print("No plan_provider passed. AI mode unavailable.")
                    else:
                        print("Computing plan...")
                        plan = plan_provider(env) or []
                        plan_idx = 0
                        ai_mode = len(plan) > 0
                        last_ai_step_time = now_ms

                        print(
                            f"Plan computed. Length={len(plan)}. "
                            f"AI mode={ai_mode}"
                        )

                if event.key == pygame.K_o:
                    if len(plan) > 0:
                        ai_mode = not ai_mode
                        last_ai_step_time = now_ms
                        print("AI mode:", ai_mode)

                if event.key == pygame.K_ESCAPE:
                    running = False

                # -------- Manual controls (only if AI is off) --------
                if not ai_mode:
                    if event.key == pygame.K_w:  # up
                        obs, reward, done, info = env.step(1)
                        total_reward += reward
                        if done:
                            running = False

                    elif event.key == pygame.K_d:  # right
                        obs, reward, done, info = env.step(2)
                        total_reward += reward
                        if done:
                            running = False

                    elif event.key == pygame.K_s:  # down
                        obs, reward, done, info = env.step(3)
                        total_reward += reward
                        if done:
                            running = False

                    elif event.key == pygame.K_a:  # left
                        obs, reward, done, info = env.step(4)
                        total_reward += reward
                        if done:
                            running = False

                    elif event.key == pygame.K_b:  # Barrier Maker
                        obs, reward, done, info = env.step(5)
                        total_reward += reward
                        if done:
                            running = False

                    elif event.key == pygame.K_h:  # Hellify
                        obs, reward, done, info = env.step(6)
                        total_reward += reward
                        if done:
                            running = False

                    elif event.key == pygame.K_r:  # reset
                        env.reset()
                        total_reward = 0.0
                        ai_mode = False
                        plan = []
                        plan_idx = 0

                    elif event.key == pygame.K_q:  # quit
                        running = False

            # Optional mouse teleport for debugging
            if event.type == pygame.MOUSEBUTTONDOWN and not ai_mode:
                mouse_x, mouse_y = event.pos
                col = mouse_x // CELL_SIZE
                row = mouse_y // CELL_SIZE

                if 0 <= row < rows and 0 <= col < cols:
                    if env.grid[row][col] == 0:
                        env.agent_pos = (row, col)

        # ---------- AI stepping ----------
        if ai_mode and plan_idx < len(plan):
            if now_ms - last_ai_step_time >= ai_step_delay_ms:
                action = plan[plan_idx]
                obs, reward, done, info = env.step(action)
                total_reward += reward
                plan_idx += 1
                last_ai_step_time = now_ms

                if not info.get("last_action_valid", True):
                    print(
                        f"AI produced invalid action at "
                        f"idx={plan_idx - 1}: {action}"
                    )
                    ai_mode = False

                if done:
                    ai_mode = False

        # --- Drawing ---
        screen.fill((255, 255, 255))

        for r in range(rows):
            for c in range(cols):
                cell_val = env.grid[r][c]

                if cell_val == 0:
                    color = (220, 220, 220)
                elif 1 <= cell_val <= 10:
                    color = (139, 69, 19)
                elif cell_val == 100:
                    color = (0, 0, 0)
                elif cell_val == -100:
                    color = (255, 0, 0)
                else:
                    color = (180, 180, 180)

                pygame.draw.rect(
                    screen,
                    color,
                    (
                        c * CELL_SIZE,
                        r * CELL_SIZE,
                        CELL_SIZE,
                        CELL_SIZE
                    )
                )

                pygame.draw.rect(
                    screen,
                    (50, 50, 50),
                    (
                        c * CELL_SIZE,
                        r * CELL_SIZE,
                        CELL_SIZE,
                        CELL_SIZE
                    ),
                    1
                )

        # Perfect squares highlight
        ps_render = []

        if (
            hasattr(env, "perf_square_manager")
            and env.perf_square_manager is not None
        ):
            try:
                raw = env.perf_square_manager.detect_perfect_squares(env.grid)
                ps_render = [
                    (
                        sq["n"],
                        sq["top_left"][0],
                        sq["top_left"][1]
                    )
                    for sq in raw
                ]
            except Exception:
                ps_render = []

        for n, r, c in ps_render:
            pygame.draw.rect(
                screen,
                (0, 255, 0),
                (
                    c * CELL_SIZE,
                    r * CELL_SIZE,
                    n * CELL_SIZE,
                    n * CELL_SIZE
                ),
                3
            )

        # Agent
        ax, ay = env.agent_pos

        pygame.draw.circle(
            screen,
            (0, 0, 255),
            (
                ay * CELL_SIZE + CELL_SIZE // 2,
                ax * CELL_SIZE + CELL_SIZE // 2
            ),
            CELL_SIZE // 3
        )

        # Info text
        x = 5
        y = (CELL_SIZE * rows) + 10

        lines = [
            f"Timestep: {env.timestep}",
            f"Stamina: {int(env.stamina)}",
            f"Agent Position: {env.agent_pos}",
            f"Total reward: {int(total_reward)}",
            f"AI mode: {ai_mode} (P=plan&run, O=pause/resume)",
            f"Plan progress: {plan_idx}/{len(plan)}",
        ]

        for i, line in enumerate(lines):
            screen.blit(
                font.render(line, True, (0, 0, 0)),
                (x, y + i * 20)
            )

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


MAP_PATH = os.path.join(
    os.path.dirname(__file__),
    "maps",
    "map4.txt"
)


def plan_provider(env):
    """Generate an AI plan from the environment's current state."""
    start_state = env_to_state(env)
    goal_node = astar(env, start_state)

    if goal_node is None:
        return []

    return extract_plan(goal_node)


if __name__ == "__main__":
    env = ShoverWorldEnv(
        map_path=MAP_PATH,
        border_lava=True
    )

    run_gui(
        env,
        plan_provider=plan_provider,
        ai_step_delay_ms=260
    )
