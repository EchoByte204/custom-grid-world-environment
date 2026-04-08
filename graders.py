"""
graders.py
==========
Deterministic grading functions for each task.

HOW GRADING WORKS:
  Graders look at the current episode state and compute a score 0.0–1.0.
  Score = weighted sum of sub-objectives completed.

  This is NOT binary (0 or 1). Partial progress always scores partial credit.
  Example (easy task):
    Package picked up but not delivered → 0.30
    Package delivered                   → 0.85
    Done with steps remaining           → 1.00

HOW IT FITS IN:
  env.py calls grade() after every step() to get the new score.
  The difference (new_score - old_score) = step_reward.
"""

from __future__ import annotations


def grade(task_id: str, state: dict) -> float:
    """
    Master grader — dispatches to task-specific grader.
    Always returns float clamped to [0.0, 1.0].

    Args:
        task_id : "easy" | "medium" | "hard"
        state   : the env's internal state dict (from env.py)

    Returns:
        float: score between 0.0 and 1.0
    """
    graders = {
        "easy":   _grade_easy,
        "medium": _grade_medium,
        "hard":   _grade_hard,
    }
    if task_id not in graders:
        raise ValueError(f"No grader for task_id='{task_id}'")

    raw = graders[task_id](state)
    # Strictly between 0 and 1 (exclusive boundaries)
    return round(max(0.0001, min(0.9999, raw)), 4)


# ══════════════════════════════════════════════════════════════
# EASY GRADER
# Max score components:
#   pick_up   : 0.30  (agent has picked up P1)
#   deliver   : 0.55  (agent has delivered P1)
#   efficiency: 0.15  (steps remaining / max_steps bonus)
# ══════════════════════════════════════════════════════════════
def _grade_easy(state: dict) -> float:
    packages  = state["packages"]     # list of package dicts
    step      = state["step_count"]
    max_steps = state["max_steps"]

    score = 0.0
    p = packages[0]  # only one package in easy

    # Component 1: picked up
    if p["picked_up"] or p["delivered"]:
        score += 0.30

    # Component 2: delivered
    if p["delivered"]:
        score += 0.55
        # Efficiency bonus: more steps left = higher bonus (up to 0.15)
        steps_used = step / max_steps
        efficiency = max(0.0, 0.15 * (1.0 - steps_used))
        score += efficiency

    return score


# ══════════════════════════════════════════════════════════════
# MEDIUM GRADER
# Two packages — each worth equal weight.
# pick_up each  : 0.15 each = 0.30 total
# deliver each  : 0.28 each = 0.56 total
# efficiency    : 0.14 bonus
# ══════════════════════════════════════════════════════════════
def _grade_medium(state: dict) -> float:
    packages  = state["packages"]
    step      = state["step_count"]
    max_steps = state["max_steps"]

    score = 0.0
    picked_count    = sum(1 for p in packages if p["picked_up"] or p["delivered"])
    delivered_count = sum(1 for p in packages if p["delivered"])

    score += picked_count    * 0.15
    score += delivered_count * 0.28

    # Efficiency bonus only if all delivered
    if delivered_count == len(packages):
        steps_used = step / max_steps
        score += max(0.0, 0.14 * (1.0 - steps_used))

    return score


# ══════════════════════════════════════════════════════════════
# HARD GRADER
# Three packages — each worth equal weight.
# pick_up each  : 0.10 each = 0.30 total
# deliver each  : 0.18 each = 0.54 total
# all_done      : 0.10 bonus
# efficiency    : 0.06 bonus
# ══════════════════════════════════════════════════════════════
def _grade_hard(state: dict) -> float:
    packages  = state["packages"]
    step      = state["step_count"]
    max_steps = state["max_steps"]

    picked_count    = sum(1 for p in packages if p["picked_up"] or p["delivered"])
    delivered_count = sum(1 for p in packages if p["delivered"])

    score  = picked_count    * 0.10
    score += delivered_count * 0.18

    # All-done bonus
    if delivered_count == len(packages):
        score += 0.10
        # Efficiency bonus
        steps_used = step / max_steps
        score += max(0.0, 0.06 * (1.0 - steps_used))

    return score


# ══════════════════════════════════════════════════════════════
# BREAKDOWN — for info/debug output
# ══════════════════════════════════════════════════════════════
def get_score_breakdown(task_id: str, state: dict) -> dict:
    """
    Returns a human-readable breakdown of score components.
    Used in StepResponse.info for the agent to understand its progress.
    """
    packages        = state["packages"]
    picked_count    = sum(1 for p in packages if p["picked_up"] or p["delivered"])
    delivered_count = sum(1 for p in packages if p["delivered"])
    total           = len(packages)

    return {
        "packages_picked":    f"{picked_count}/{total}",
        "packages_delivered": f"{delivered_count}/{total}",
        "step_progress":      f"{state['step_count']}/{state['max_steps']}",
        "total_score":        grade(task_id, state),
    }
