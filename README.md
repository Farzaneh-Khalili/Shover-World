# Shover-World

A discrete grid-world environment with **box-pushing mechanics** and an **A\*-based Player AI**.

## Overview

The project is developed in two phases:

- **Phase 1:** Grid-world environment with agent movement, box pushing, lava, barriers, stamina, and special mechanics.
- **Phase 2:** Search-based Player AI using **A\***, heuristic evaluation, and subgoal selection.

A **Pygame GUI** is included for both manual and AI-controlled interaction.

## Main Features

- Grid-based movement and box pushing
- Chain pushing
- Lava and box destruction
- Stamina and pushing-force mechanics
- Barriers and special actions
- Random and file-based maps
- A\* search-based Player AI
- Heuristic evaluation and subgoal selection
- Pygame interface

## Project Structure

```text
Shover-World/
├── phase_1/
│   ├── environment.py
│   ├── gui.py
│   ├── test.py
│   ├── maps/
│   └── tests/
├── phase_2/
│   ├── environment.py
│   ├── player_ai.py
│   ├── heuristic_b.py
│   ├── subgoal.py
│   ├── utils.py
│   ├── gui_ai.py
│   ├── maps/
│   └── tests/
├── reports/
├── .gitignore
├── requirements.txt
└── README.md
```

## Running

### Phase 1

```bash
cd phase_1
python test.py
```

Run the GUI:

```bash
python gui.py
```

Run tests:

```bash
pytest tests -v
```

### Phase 2

```bash
cd phase_2
python player_ai.py
```

Run the AI GUI:

```bash
python gui_ai.py
```

## Reports

Detailed project specifications, implementation details, experiments, and evaluations are available in the `reports/` directory.

## Authors

**Farzaneh Khalili**
**Hasti Javadzadeh**

Computer Science — Shahid Beheshti University
