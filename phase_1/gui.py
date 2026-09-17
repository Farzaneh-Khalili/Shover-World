import pygame


def run_gui(env):
    pygame.init()

    WIDTH = 800
    HEIGHT = 600

    CELL_SIZE = 40
    rows = env.n_rows
    cols = env.n_cols

    perfect_squares = []

    total_reward = 0.0

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Shover World")

    font = pygame.font.SysFont(None, 24)
    clock = pygame.time.Clock()

    running = True
    while running:
        # --- event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:  # up
                    obs, reward, done, info = env.step(1)
                    total_reward += reward
                    perfect_squares = info.get("perfect_squares_available", [])
                    if done:
                        running = False

                elif event.key == pygame.K_d:  # right
                    obs, reward, done, info = env.step(2)
                    total_reward += reward
                    perfect_squares = info.get("perfect_squares_available", [])
                    if done:
                        running = False

                elif event.key == pygame.K_s:  # down
                    obs, reward, done, info = env.step(3)
                    total_reward += reward
                    perfect_squares = info.get("perfect_squares_available", [])
                    if done:
                        running = False

                elif event.key == pygame.K_a:  # left
                    obs, reward, done, info = env.step(4)
                    total_reward += reward
                    perfect_squares = info.get("perfect_squares_available", [])
                    if done:
                        running = False

                elif event.key == pygame.K_b:  # Barrier Maker
                    obs, reward, done, info = env.step(5)
                    total_reward += reward
                    perfect_squares = info.get("perfect_squares_available", [])
                    if done:
                        running = False

                elif event.key == pygame.K_h:  # Hellify
                    obs, reward, done, info = env.step(6)
                    total_reward += reward
                    perfect_squares = info.get("perfect_squares_available", [])
                    if done:
                        running = False

                elif event.key == pygame.K_r:  # reset
                    env.reset()
                    total_reward = 0.0
                    continue

                elif event.key == pygame.K_q:  # quit
                    running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                col = mouse_x // CELL_SIZE
                row = mouse_y // CELL_SIZE

                if 0 <= row < rows and 0 <= col < cols:
                    cell_val = env.grid[row][col]

                    if cell_val == 0:
                        env.agent_pos = (row, col)

        # --- drawing ---
        screen.fill((255, 255, 255))

        for r in range(rows):
            for c in range(cols):

                cell_val = env.grid[r][c]

                # choose color
                if cell_val == 0:
                    color = (220, 220, 220)
                elif 1 <= cell_val <= 10:
                    color = (139, 69, 19)  # brown for boxes
                elif cell_val == 100:
                    color = (0, 0, 0)      # barrier
                elif cell_val == -100:
                    color = (255, 0, 0)    # lava
                else:
                    color = (180, 180, 180)

                pygame.draw.rect(
                    screen,
                    color,
                    (c*CELL_SIZE, r*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                )

                # cell borders
                pygame.draw.rect(
                    screen,
                    (50, 50, 50),
                    (c*CELL_SIZE, r*CELL_SIZE, CELL_SIZE, CELL_SIZE),
                    1
                )
        # compute perfect squares for rendering
        ps_render = []
        if hasattr(env, "perf_square_manager") and env.perf_square_manager is not None:
            try:
                raw = env.perf_square_manager.detect_perfect_squares(env.grid)
                # raw is a list of dicts: {"n": n, "top_left": (r,c)}
                ps_render = [(sq["n"], sq["top_left"][0],
                              sq["top_left"][1]) for sq in raw]
            except Exception as e:
                ps_render = []

        # Highlight perfect squares
        for (n, r, c) in ps_render:
            pygame.draw.rect(
                screen,
                (0, 255, 0),  # green outline
                (c * CELL_SIZE, r * CELL_SIZE, n * CELL_SIZE, n * CELL_SIZE),
                3
            )

        # agent
        ax, ay = env.agent_pos
        pygame.draw.circle(
            screen,
            (0, 0, 255),
            (ay*CELL_SIZE + CELL_SIZE//2, ax*CELL_SIZE + CELL_SIZE//2),
            CELL_SIZE//3
        )

        # info
        x = 5
        y = (CELL_SIZE * rows) + 10

        timestep = env.timestep
        stamina = env.stamina
        agent_pos = env.agent_pos
        # mode = env.special_mode

        lines = [
            f"Timestep: {timestep}",
            f"Stamina: {int(stamina)}",
            f"Agent Position: {agent_pos}",
            f"Total reward: {int(total_reward)}"
        ]

        for i, line in enumerate(lines):
            text_surface = font.render(line, True, (0, 0, 0))
            screen.blit(text_surface, (x, y + i*20))

        pygame.display.flip()
        clock.tick(30)    # limit FPS to make CPU happy

    pygame.quit()


if __name__ == "__main__":
    from environment import ShoverWorldEnv

    env = ShoverWorldEnv(
        map_path=r"D:\uni\7. ترم هفتم\هوش مصنوعی\shower_world\maps\map1.txt")
    run_gui(env)
