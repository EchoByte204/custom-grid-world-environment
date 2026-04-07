"""
tasks.py
========
Three warehouse navigation task definitions.

EASY   → 5×5 grid, 1 package, 2 obstacles
MEDIUM → 8×8 grid, 2 packages, 6 obstacles
HARD   → 12×12 grid, 3 packages, 15 obstacles + time pressure

Each task dict contains:
  grid_size       : (rows, cols)
  agent_start     : (row, col) starting position
  packages        : list of package dicts
  delivery_bays   : list of bay dicts
  obstacles       : list of (row, col) blocked cells
  max_steps       : episode step limit
  description     : human-readable task description

HOW IT FITS IN:
  env.py calls load_task(task_id) in reset()
  The task dict seeds the entire episode state.
"""

from __future__ import annotations


# ══════════════════════════════════════════════════════════════
# TASK 1 — EASY  (5×5 grid)
# ══════════════════════════════════════════════════════════════
# Grid layout (R=robot, P=package, B=bay, X=obstacle, .=empty):
#
#   . . X . .
#   . . . P .
#   . X . . .
#   . . . . .
#   . . . . B
#
# Robot starts at (0,0), picks P at (1,3), delivers to B at (4,4)
# Optimal path is ~10 steps — max 20 gives breathing room.
# ══════════════════════════════════════════════════════════════
def get_easy_task() -> dict:
    return {
        "task_id": "easy",
        "description": (
            "Navigate a 5×5 warehouse grid. Pick up 1 package and deliver it "
            "to the shipping bay. Avoid 2 obstacles."
        ),
        "grid_size": (5, 5),
        "agent_start": (0, 0),
        "packages": [
            {"id": "P1", "pos": [1, 3], "picked_up": False, "delivered": False},
        ],
        "delivery_bays": [
            {"id": "B1", "pos": [4, 4]},
        ],
        "obstacles": [
            [0, 2],
            [2, 1],
        ],
        "max_steps": 20,
        "reward_weights": {
            "pick_up":      0.25,
            "deliver":      0.50,
            "all_done":     0.20,
            "step_penalty": 0.005,
            "wall_penalty": 0.05,
            "approach_bonus": 0.02,
        },
    }


# ══════════════════════════════════════════════════════════════
# TASK 2 — MEDIUM  (8×8 grid)
# ══════════════════════════════════════════════════════════════
# Two packages, one bay, more obstacles. Agent must plan which
# package to fetch first (P1 is closer to start).
# ══════════════════════════════════════════════════════════════
def get_medium_task() -> dict:
    return {
        "task_id": "medium",
        "description": (
            "Navigate an 8×8 warehouse grid. Pick up 2 packages and deliver "
            "both to the shipping bay. Avoid 6 obstacles."
        ),
        "grid_size": (8, 8),
        "agent_start": (0, 0),
        "packages": [
            {"id": "P1", "pos": [2, 3], "picked_up": False, "delivered": False},
            {"id": "P2", "pos": [5, 6], "picked_up": False, "delivered": False},
        ],
        "delivery_bays": [
            {"id": "B1", "pos": [7, 7]},
        ],
        "obstacles": [
            [1, 2], [1, 5],
            [3, 0], [3, 4],
            [5, 2], [6, 5],
        ],
        "max_steps": 40,
        "reward_weights": {
            "pick_up":      0.20,
            "deliver":      0.30,
            "all_done":     0.20,
            "step_penalty": 0.003,
            "wall_penalty": 0.05,
            "approach_bonus": 0.015,
        },
    }


# ══════════════════════════════════════════════════════════════
# TASK 3 — HARD  (12×12 grid)
# ══════════════════════════════════════════════════════════════
# Three packages, two bays (any bay accepts any package).
# Dense obstacle field — requires genuine pathfinding.
# Each package must be picked up and delivered separately
# (agent can only carry one at a time).
# ══════════════════════════════════════════════════════════════
def get_hard_task() -> dict:
    return {
        "task_id": "hard",
        "description": (
            "Navigate a 12×12 warehouse grid. Pick up 3 packages and deliver "
            "them to shipping bays. Avoid 15 obstacles. Time pressure applies."
        ),
        "grid_size": (12, 12),
        "agent_start": (0, 0),
        "packages": [
            {"id": "P1", "pos": [2, 4],  "picked_up": False, "delivered": False},
            {"id": "P2", "pos": [6, 2],  "picked_up": False, "delivered": False},
            {"id": "P3", "pos": [9, 8],  "picked_up": False, "delivered": False},
        ],
        "delivery_bays": [
            {"id": "B1", "pos": [11, 11]},
            {"id": "B2", "pos": [11, 0]},
        ],
        "obstacles": [
            [1, 3], [1, 7], [1, 10],
            [3, 1], [3, 5], [3, 9],
            [5, 3], [5, 7], [5, 11],
            [7, 0], [7, 4], [7, 8],
            [9, 2], [9, 6], [10, 9],
        ],
        "max_steps": 80,
        "reward_weights": {
            "pick_up":      0.15,
            "deliver":      0.20,
            "all_done":     0.25,
            "step_penalty": 0.002,
            "wall_penalty": 0.05,
            "approach_bonus": 0.01,
        },
    }


# ══════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════

TASK_REGISTRY: dict[str, callable] = {
    "easy":   get_easy_task,
    "medium": get_medium_task,
    "hard":   get_hard_task,
}


def load_task(task_id: str) -> dict:
    """
    Load a task definition by ID.
    Returns a fresh copy each time (so resets are truly independent).
    """
    if task_id not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task_id '{task_id}'. "
            f"Valid options: {list(TASK_REGISTRY.keys())}"
        )
    import copy
    return copy.deepcopy(TASK_REGISTRY[task_id]())


def list_tasks() -> list[dict]:
    """Return summary info for all tasks (used by /tasks endpoint)."""
    summaries = []
    for tid, fn in TASK_REGISTRY.items():
        t = fn()
        summaries.append({
            "id":            t["task_id"],
            "description":   t["description"],
            "grid_size":     list(t["grid_size"]),
            "num_packages":  len(t["packages"]),
            "num_obstacles": len(t["obstacles"]),
            "max_steps":     t["max_steps"],
        })
    return summaries
