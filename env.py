"""
env.py
======
Core Warehouse Robot Navigation Environment.

THIS IS THE MOST IMPORTANT FILE.
All game logic, state management, and OpenEnv interface lives here.

OpenEnv interface implemented:
  reset(task_id)  → Observation       (start fresh episode)
  step(action)    → StepResponse      (agent acts, env responds)
  state()         → dict              (raw internal state for inspection)

Grid coordinate system:
  (0,0) = top-left corner
  row increases going DOWN  (south)
  col increases going RIGHT (east)

Cell contents (internal):
  'A' = Agent
  'P' = Package (not yet picked)
  'B' = Delivery Bay
  'X' = Obstacle
  '.' = Empty floor

HOW IT FITS IN:
  app.py creates ONE instance of this class and routes HTTP calls to it.
  inference.py calls the HTTP endpoints, NOT this class directly.
"""

from __future__ import annotations
import math
from typing import Optional

from models import (
    Action, Observation, Reward, StepResponse,
    Package, DeliveryBay
)
from tasks import load_task
from graders import grade, get_score_breakdown


# Direction vectors — (row_delta, col_delta)
MOVE_DELTAS: dict[str, tuple[int, int]] = {
    "move_up":    (-1,  0),
    "move_down":  ( 1,  0),
    "move_left":  ( 0, -1),
    "move_right": ( 0,  1),
}


class WarehouseNavEnv:
    """
    Warehouse Robot Navigation Environment.

    The agent controls a robot on a 2D grid.
    The robot must navigate to packages, pick them up,
    navigate to delivery bays, and drop them off.
    """

    def __init__(self):
        # These are set by reset()
        self.task: Optional[dict]        = None
        self.agent_pos: list[int]        = [0, 0]
        self.packages: list[dict]        = []
        self.delivery_bays: list[dict]   = []
        self.obstacles: set[tuple]       = set()
        self.carrying: Optional[str]     = None   # Package ID being held
        self.grid_size: tuple[int, int]  = (5, 5)
        self.step_count: int             = 0
        self.max_steps: int              = 20
        self.episode_score: float        = 0.0
        self.done: bool                  = False
        self._last_message: str          = ""

    # ──────────────────────────────────────────────────────────
    # reset()
    # ──────────────────────────────────────────────────────────
    def reset(self, task_id: str = "easy") -> Observation:
        """
        Start a fresh episode.
        Loads the task, initializes all state, returns first observation.
        """
        self.task        = load_task(task_id)
        self.grid_size   = self.task["grid_size"]
        self.agent_pos   = list(self.task["agent_start"])
        self.packages    = [dict(p) for p in self.task["packages"]]
        self.delivery_bays = [dict(b) for b in self.task["delivery_bays"]]
        self.obstacles   = {tuple(o) for o in self.task["obstacles"]}
        self.max_steps   = self.task["max_steps"]
        self.carrying    = None
        self.step_count  = 0
        self.episode_score = grade(self.task["task_id"], self._raw_state())
        self.done        = False
        self._last_message = (
            f"Episode started. Task: {task_id}. "
            f"Grid: {self.grid_size[0]}×{self.grid_size[1]}. "
            f"Packages to deliver: {len(self.packages)}."
        )
        return self._make_observation()

    # ──────────────────────────────────────────────────────────
    # step()
    # ──────────────────────────────────────────────────────────
    def step(self, action: Action) -> StepResponse:
        """
        Process one agent action.

        Flow:
          1. Validate episode is active
          2. Apply the action to the grid state
          3. Re-grade the full episode state
          4. Check termination conditions
          5. Return StepResponse
        """
        # Guard: episode already done
        if self.done:
            return StepResponse(
                observation=self._make_observation(),
                reward=Reward(
                    score=self.episode_score,
                    step_reward=0.0,
                    done=True,
                    success=False,
                    feedback="Episode is over. Call /reset to start a new one."
                ),
                done=True,
                info={"warning": "Episode already finished."}
            )

        self.step_count += 1
        score_before = self.episode_score
        atype = action.action_type.lower().strip()

        # Apply action and get immediate feedback
        step_delta, feedback = self._apply_action(atype)

        # Re-grade full state
        new_score = grade(self.task["task_id"], self._raw_state())
        self.episode_score = new_score
        actual_delta = round(new_score - score_before, 4)

        # Check termination
        all_delivered = all(p["delivered"] for p in self.packages)
        out_of_steps  = self.step_count >= self.max_steps

        if all_delivered:
            self.done = True
            feedback += f" ✅ All packages delivered! Final score: {new_score:.3f}"
        elif out_of_steps:
            self.done = True
            feedback += f" ⏰ Out of steps. Final score: {new_score:.3f}"

        self._last_message = feedback

        reward = Reward(
            score=new_score,
            step_reward=actual_delta,
            done=self.done,
            success=all_delivered,
            feedback=feedback,
        )

        return StepResponse(
            observation=self._make_observation(),
            reward=reward,
            done=self.done,
            info={
                **get_score_breakdown(self.task["task_id"], self._raw_state()),
                "agent_pos": self.agent_pos,
                "carrying":  self.carrying,
            }
        )

    # ──────────────────────────────────────────────────────────
    # state()
    # ──────────────────────────────────────────────────────────
    def state(self) -> dict:
        """Return full raw state. Used by GET /state for debugging."""
        if self.task is None:
            return {"status": "not_initialized", "message": "Call /reset first."}
        return {
            "task_id":       self.task["task_id"],
            "step_count":    self.step_count,
            "max_steps":     self.max_steps,
            "done":          self.done,
            "episode_score": self.episode_score,
            "agent_pos":     self.agent_pos,
            "carrying":      self.carrying,
            "packages":      self.packages,
            "delivery_bays": self.delivery_bays,
            "obstacles":     [list(o) for o in self.obstacles],
            "grid_size":     list(self.grid_size),
        }

    # ──────────────────────────────────────────────────────────
    # _apply_action()  — the heart of the environment
    # ──────────────────────────────────────────────────────────
    def _apply_action(self, atype: str) -> tuple[float, str]:
        """
        Apply one action to the environment state.
        Returns (step_delta_reward, feedback_string).
        """
        weights = self.task.get("reward_weights", {})

        # ── MOVEMENT ─────────────────────────────────────────
        if atype in MOVE_DELTAS:
            dr, dc = MOVE_DELTAS[atype]
            new_row = self.agent_pos[0] + dr
            new_col = self.agent_pos[1] + dc
            rows, cols = self.grid_size

            # Wall collision
            if not (0 <= new_row < rows and 0 <= new_col < cols):
                return (
                    -weights.get("wall_penalty", 0.05),
                    f"Bumped into wall moving {atype.replace('move_','')}. Stay within grid."
                )

            # Obstacle collision
            if (new_row, new_col) in self.obstacles:
                return (
                    -weights.get("wall_penalty", 0.05),
                    f"Hit an obstacle at [{new_row},{new_col}]. Choose a different path."
                )

            # Compute approach bonus before moving
            approach_bonus = self._compute_approach_bonus(
                (self.agent_pos[0], self.agent_pos[1]),
                (new_row, new_col),
                weights.get("approach_bonus", 0.02)
            )

            # Valid move
            self.agent_pos = [new_row, new_col]
            step_pen = -weights.get("step_penalty", 0.005)
            total = step_pen + approach_bonus

            return (
                total,
                f"Moved {atype.replace('move_','')} to [{new_row},{new_col}]."
                + (f" Getting closer to target." if approach_bonus > 0 else "")
            )

        # ── PICK UP ──────────────────────────────────────────
        elif atype == "pick_up":
            if self.carrying is not None:
                return (-0.02, f"Already carrying package {self.carrying}. Deliver it first.")

            # Find package at agent position
            pkg = self._package_at(self.agent_pos)
            if pkg is None:
                return (-0.02, f"No package at {self.agent_pos}. Move to a package first.")
            if pkg["picked_up"]:
                return (-0.02, f"Package {pkg['id']} already picked up.")

            pkg["picked_up"] = True
            self.carrying = pkg["id"]
            return (
                weights.get("pick_up", 0.20),
                f"Picked up package {pkg['id']} at {self.agent_pos}. Now deliver it to a bay."
            )

        # ── DROP OFF ─────────────────────────────────────────
        elif atype == "drop_off":
            if self.carrying is None:
                return (-0.02, "Not carrying any package. Pick one up first.")

            # Check if agent is on a delivery bay
            bay = self._bay_at(self.agent_pos)
            if bay is None:
                return (
                    -0.02,
                    f"Not at a delivery bay. Current position {self.agent_pos}. "
                    f"Bays are at: {[b['pos'] for b in self.delivery_bays]}."
                )

            # Deliver!
            pkg_id = self.carrying
            for p in self.packages:
                if p["id"] == pkg_id:
                    p["delivered"] = True
                    break

            self.carrying = None
            return (
                weights.get("deliver", 0.35),
                f"Delivered package {pkg_id} to bay {bay['id']}! "
                f"{sum(1 for p in self.packages if p['delivered'])}/{len(self.packages)} packages done."
            )

        # ── WAIT ─────────────────────────────────────────────
        elif atype == "wait":
            return (
                -weights.get("step_penalty", 0.005) * 2,
                "Waited one step. Time costs efficiency — keep moving."
            )

        # ── UNKNOWN ──────────────────────────────────────────
        else:
            return (
                -0.05,
                f"Unknown action '{atype}'. Valid actions: "
                "move_up, move_down, move_left, move_right, pick_up, drop_off, wait."
            )

    # ──────────────────────────────────────────────────────────
    # Helper: compute approach bonus
    # ──────────────────────────────────────────────────────────
    def _compute_approach_bonus(
        self,
        old_pos: tuple[int, int],
        new_pos: tuple[int, int],
        bonus_val: float
    ) -> float:
        """
        Give a small reward if the agent moved closer to its current target.
        Target = nearest undelivered package (if not carrying) OR nearest bay (if carrying).
        """
        target = self._get_current_target()
        if target is None:
            return 0.0

        old_dist = abs(old_pos[0] - target[0]) + abs(old_pos[1] - target[1])
        new_dist = abs(new_pos[0] - target[0]) + abs(new_pos[1] - target[1])

        if new_dist < old_dist:
            return bonus_val
        elif new_dist > old_dist:
            return -bonus_val * 0.5
        return 0.0

    def _get_current_target(self) -> Optional[tuple[int, int]]:
        """Return (row, col) of what the agent should be moving toward."""
        if self.carrying:
            # Moving toward nearest bay
            bays = self.delivery_bays
            if not bays:
                return None
            agent = self.agent_pos
            nearest = min(bays, key=lambda b: abs(b["pos"][0] - agent[0]) + abs(b["pos"][1] - agent[1]))
            return tuple(nearest["pos"])
        else:
            # Moving toward nearest undelivered, not-yet-picked package
            pending = [p for p in self.packages if not p["picked_up"] and not p["delivered"]]
            if not pending:
                return None
            agent = self.agent_pos
            nearest = min(pending, key=lambda p: abs(p["pos"][0] - agent[0]) + abs(p["pos"][1] - agent[1]))
            return tuple(nearest["pos"])

    def _package_at(self, pos: list[int]) -> Optional[dict]:
        """Return package dict at given position (not yet delivered), or None."""
        for p in self.packages:
            if p["pos"] == pos and not p["delivered"]:
                return p
        return None

    def _bay_at(self, pos: list[int]) -> Optional[dict]:
        """Return delivery bay dict at given position, or None."""
        for b in self.delivery_bays:
            if b["pos"] == pos:
                return b
        return None

    # ──────────────────────────────────────────────────────────
    # _raw_state() — internal state for graders
    # ──────────────────────────────────────────────────────────
    def _raw_state(self) -> dict:
        return {
            "packages":   self.packages,
            "step_count": self.step_count,
            "max_steps":  self.max_steps,
        }

    # ──────────────────────────────────────────────────────────
    # _make_observation() — build Observation model
    # ──────────────────────────────────────────────────────────
    def _make_observation(self) -> Observation:
        # Build remaining_issues checklist
        remaining = []
        for p in self.packages:
            if not p["picked_up"] and not p["delivered"]:
                remaining.append(f"pick_{p['id']}")
            if p["picked_up"] and not p["delivered"]:
                remaining.append(f"deliver_{p['id']}")

        return Observation(
            task_id         = self.task["task_id"] if self.task else "none",
            step            = self.step_count,
            max_steps       = self.max_steps,
            grid_size       = list(self.grid_size),
            agent_pos       = list(self.agent_pos),
            packages        = [Package(**p) for p in self.packages],
            delivery_bays   = [DeliveryBay(**b) for b in self.delivery_bays],
            obstacles       = [list(o) for o in self.obstacles],
            carrying        = self.carrying,
            delivered_count = sum(1 for p in self.packages if p["delivered"]),
            total_packages  = len(self.packages),
            current_score   = round(self.episode_score, 4),
            remaining_issues= remaining,
            message         = self._last_message,
            ascii_grid      = self._render_ascii(),
        )

    # ──────────────────────────────────────────────────────────
    # _render_ascii() — text visualization of the grid
    # ──────────────────────────────────────────────────────────
    def _render_ascii(self) -> str:
        """
        Renders a text grid. Useful for debugging and for the LLM agent
        to understand the spatial layout without needing pygame.

        Legend:
          A = Agent (robot)
          C = Agent carrying a package
          P = Package waiting to be picked up
          B = Delivery Bay
          X = Obstacle
          . = Empty floor
        """
        if self.task is None:
            return ""

        rows, cols = self.grid_size
        grid = [['.' for _ in range(cols)] for _ in range(rows)]

        # Place obstacles
        for (r, c) in self.obstacles:
            grid[r][c] = 'X'

        # Place delivery bays
        for bay in self.delivery_bays:
            r, c = bay["pos"]
            grid[r][c] = 'B'

        # Place packages (only undelivered ones)
        for pkg in self.packages:
            if not pkg["delivered"] and not pkg["picked_up"]:
                r, c = pkg["pos"]
                grid[r][c] = pkg["id"][0]  # 'P'

        # Place agent (C if carrying, A if not)
        ar, ac = self.agent_pos
        grid[ar][ac] = 'C' if self.carrying else 'A'

        # Build string with column header
        col_header = "   " + " ".join(str(c) for c in range(cols))
        lines = [col_header]
        for r in range(rows):
            row_str = f"{r:2d} " + " ".join(grid[r])
            lines.append(row_str)

        legend = "\nLegend: A=Agent C=Carrying P=Package B=Bay X=Obstacle .=Floor"
        return "\n".join(lines) + legend
