"""
models.py
=========
Typed Pydantic models for the Warehouse Robot Navigation Environment.

These define the exact data structures that flow between the agent and
the environment. Think of these as the "protocol" — both sides must
speak the same language.

HOW IT FITS IN:
  Agent sends  → Action
  Env returns  → Observation + Reward (wrapped in StepResponse)
  Env state    → StateSnapshot (for debugging/inspection)
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Optional


# ══════════════════════════════════════════════════════════════
# GRID PRIMITIVES
# ══════════════════════════════════════════════════════════════

class Position(BaseModel):
    """A (row, col) coordinate on the grid."""
    row: int = Field(..., description="Row index (0 = top)")
    col: int = Field(..., description="Column index (0 = left)")

    def to_list(self) -> list[int]:
        return [self.row, self.col]

    def distance_to(self, other: "Position") -> int:
        """Manhattan distance to another position."""
        return abs(self.row - other.row) + abs(self.col - other.col)


class Package(BaseModel):
    """A package somewhere on the grid waiting to be picked up."""
    id: str = Field(..., description="Unique package ID e.g. 'P1'")
    pos: list[int] = Field(..., description="[row, col] location of package")
    picked_up: bool = Field(default=False, description="Has the agent picked this up?")
    delivered: bool = Field(default=False, description="Has this been delivered?")


class DeliveryBay(BaseModel):
    """A delivery bay — the destination for packages."""
    id: str = Field(..., description="Unique bay ID e.g. 'B1'")
    pos: list[int] = Field(..., description="[row, col] location of bay")


# ══════════════════════════════════════════════════════════════
# OBSERVATION — what the agent sees each step
# ══════════════════════════════════════════════════════════════

class Observation(BaseModel):
    """
    Full observation returned to the agent after every reset() and step().

    The agent uses this to decide its next action.
    Key fields:
      - agent_pos     : where the robot is right now
      - packages      : all packages (picked up or not)
      - obstacles     : walls/blocked cells the agent cannot enter
      - carrying      : which package ID the agent is currently holding (or null)
      - remaining_issues : high-level task checklist still outstanding
    """
    task_id: str = Field(..., description="Active task: easy | medium | hard")
    step: int = Field(..., description="Current step number (starts at 0)")
    max_steps: int = Field(..., description="Episode ends when step >= max_steps")

    grid_size: list[int] = Field(..., description="[rows, cols] of the grid")
    agent_pos: list[int] = Field(..., description="Agent's current [row, col]")

    packages: list[Package] = Field(..., description="All packages and their status")
    delivery_bays: list[DeliveryBay] = Field(..., description="All delivery bay locations")
    obstacles: list[list[int]] = Field(..., description="List of [row, col] blocked cells")

    carrying: Optional[str] = Field(
        default=None,
        description="Package ID the agent is holding, or null if hands are empty"
    )
    delivered_count: int = Field(default=0, description="Packages delivered so far")
    total_packages: int = Field(..., description="Total packages to deliver this episode")

    current_score: float = Field(..., description="Running score 0.0–1.0")
    remaining_issues: list[str] = Field(
        default_factory=list,
        description="High-level objectives still pending e.g. ['pick_P1', 'deliver_P1']"
    )
    message: str = Field(default="", description="Feedback from the last action taken")

    # ASCII grid for easy debugging
    ascii_grid: str = Field(
        default="",
        description="Text visualization of the current grid state"
    )


# ══════════════════════════════════════════════════════════════
# ACTION — what the agent sends to the environment
# ══════════════════════════════════════════════════════════════

class Action(BaseModel):
    """
    One action the agent wants to take.

    Valid action_type values:
      move_up    — move north (row - 1)
      move_down  — move south (row + 1)
      move_left  — move west  (col - 1)
      move_right — move east  (col + 1)
      pick_up    — pick up package at current cell
      drop_off   — deliver held package at current cell (must be a bay)
      wait       — stay in place (small time penalty)

    No parameters needed for navigation actions.
    """
    action_type: str = Field(
        ...,
        description=(
            "One of: move_up | move_down | move_left | move_right | "
            "pick_up | drop_off | wait"
        )
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional extra parameters (reserved for future use)"
    )


# ══════════════════════════════════════════════════════════════
# REWARD — what the environment sends back after each step
# ══════════════════════════════════════════════════════════════

class Reward(BaseModel):
    """
    Reward signal returned after each step().

    score       : cumulative normalized score 0.0–1.0
    step_reward : delta reward earned by this specific action (can be negative)
    done        : whether the episode has ended
    success     : True if the agent completed the full task (score >= 0.95)
    feedback    : human-readable explanation of what happened
    """
    score: float = Field(..., description="Cumulative score 0.0–1.0")
    step_reward: float = Field(..., description="Reward from this step (can be negative)")
    done: bool = Field(..., description="True if episode is over")
    success: bool = Field(default=False, description="True if task fully completed")
    feedback: str = Field(..., description="What happened as a result of this action")


# ══════════════════════════════════════════════════════════════
# STEP RESPONSE — full return value from step()
# ══════════════════════════════════════════════════════════════

class StepResponse(BaseModel):
    """Complete response from a single step() call."""
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra diagnostic information (step count, raw scores, etc.)"
    )


# ══════════════════════════════════════════════════════════════
# RESET REQUEST — input to reset()
# ══════════════════════════════════════════════════════════════

class ResetRequest(BaseModel):
    """Request body for POST /reset"""
    task_id: str = Field(
        default="easy",
        description="Which task to load: easy | medium | hard"
    )
