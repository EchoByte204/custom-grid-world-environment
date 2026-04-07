"""
app.py
======
FastAPI HTTP server that wraps the WarehouseNavEnv.

WHY THIS EXISTS:
  OpenEnv requires the environment to be accessible via HTTP so that
  the competition validators, HF Spaces, and inference.py can all
  talk to it without importing Python modules directly.

ENDPOINTS:
  GET  /          → health check (competition validator pings this)
  POST /reset     → start new episode
  POST /step      → take one action
  GET  /state     → inspect raw state (debugging)
  GET  /tasks     → list all tasks
  GET  /render    → get ASCII grid of current state

HOW IT FITS IN:
  inference.py  → calls these endpoints via HTTP
  Dockerfile    → runs: uvicorn app:app --host 0.0.0.0 --port 7860
  HF Spaces     → exposes port 7860 publicly
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from models import Action, ResetRequest, StepResponse, Observation
from env import WarehouseNavEnv
from tasks import list_tasks

# ── Load .env file if present ────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional in production container

# ── App setup ────────────────────────────────────────────────
app = FastAPI(
    title="Warehouse Robot Navigation Environment",
    description=(
        "OpenEnv-compliant environment where AI agents learn to navigate "
        "warehouse floors, pick up packages, and deliver them to shipping bays."
    ),
    version="1.0.0",
    docs_url="/docs",      # Swagger UI at /docs
    redoc_url="/redoc",    # ReDoc UI at /redoc
)

# Allow all origins (needed for HF Spaces iframe + validators)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared environment instance (stateful)
env = WarehouseNavEnv()


# ════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════

@app.get("/")
@app.get("/health")
def health_check():
    """
    Health check endpoint.
    Competition validator pings this — must return HTTP 200.
    """
    return {
        "status":      "ok",
        "environment": "Warehouse Robot Navigation Environment",
        "version":     "1.0.0",
        "tasks":       ["easy", "medium", "hard"],
        "endpoints":   ["/reset", "/step", "/state", "/tasks", "/render", "/docs", "/health"],
    }


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest):
    """
    Start a new episode.

    Body: {"task_id": "easy" | "medium" | "hard"}
    Returns: Initial Observation
    """
    try:
        obs = env.reset(task_id=request.task_id)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(action: Action):
    """
    Take one action in the current episode.

    Body: {"action_type": "move_right", "parameters": {}}
    Returns: StepResponse with new Observation + Reward
    """
    if env.task is None:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call POST /reset first."
        )
    return env.step(action)


@app.get("/state")
def state():
    """
    Return raw internal environment state.
    Useful for debugging — not required by agents.
    """
    return env.state()


@app.get("/tasks")
def tasks():
    """List all available tasks with metadata."""
    return {"tasks": list_tasks()}


@app.get("/render", response_class=PlainTextResponse)
def render():
    """
    Return ASCII text visualization of the current grid.
    Useful for debugging without pygame.

    Example output:
       0 1 2 3 4
     0 A . X . .
     1 . . . P .
     2 . X . . .
     3 . . . . .
     4 . . . . B
    """
    if env.task is None:
        return "No active episode. Call POST /reset first."
    return env._render_ascii()
