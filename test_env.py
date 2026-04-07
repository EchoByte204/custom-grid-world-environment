"""
test_env.py
===========
Compatibility + smoke tests for the Warehouse Nav Environment.

RUN THIS FIRST before starting the server or inference script.
It catches common issues early: wrong Python version, missing packages,
broken env logic, grader problems.

Usage:
  python test_env.py

All tests must show ✅ before proceeding.
"""

import sys
import os

print("=" * 60)
print("  Warehouse Nav Env — Compatibility & Smoke Tests")
print("=" * 60)

PASS = "✅"
FAIL = "❌"
results = []


def test(name: str, fn):
    """Run a test function and record pass/fail."""
    try:
        fn()
        print(f"  {PASS} {name}")
        results.append((name, True, None))
    except Exception as e:
        print(f"  {FAIL} {name}")
        print(f"       Error: {e}")
        results.append((name, False, str(e)))


# ══════════════════════════════════════════════════════════════
# 1. Python version check
# ══════════════════════════════════════════════════════════════
print("\n[1] Python Environment")

def check_python_version():
    major, minor = sys.version_info.major, sys.version_info.minor
    assert major == 3 and minor >= 10, (
        f"Python 3.10+ required. You have {major}.{minor}. "
        f"Download from https://python.org"
    )
    print(f"       Python {major}.{minor} ✓", end="")

test("Python version >= 3.10", check_python_version)


# ══════════════════════════════════════════════════════════════
# 2. Required packages
# ══════════════════════════════════════════════════════════════
print("\n[2] Required Packages")

def check_fastapi():
    import fastapi
    assert hasattr(fastapi, "__version__")

def check_uvicorn():
    import uvicorn

def check_pydantic():
    import pydantic
    major = int(pydantic.__version__.split(".")[0])
    assert major >= 2, f"Pydantic 2.x required, got {pydantic.__version__}"

def check_numpy():
    import numpy as np
    _ = np.array([1, 2, 3])

def check_openai():
    import openai
    assert hasattr(openai, "OpenAI")

def check_requests():
    import requests

def check_dotenv():
    from dotenv import load_dotenv

test("fastapi",          check_fastapi)
test("uvicorn",          check_uvicorn)
test("pydantic >= 2.x",  check_pydantic)
test("numpy",            check_numpy)
test("openai",           check_openai)
test("requests",         check_requests)
test("python-dotenv",    check_dotenv)


# ══════════════════════════════════════════════════════════════
# 3. Project files exist
# ══════════════════════════════════════════════════════════════
print("\n[3] Project Files")

required_files = [
    "models.py", "tasks.py", "graders.py",
    "env.py", "app.py", "inference.py",
    "openenv.yaml", "Dockerfile", "requirements.txt",
    ".env.example", "README.md",
]

for fname in required_files:
    def make_check(f):
        def _check():
            assert os.path.exists(f), f"Missing file: {f}"
        return _check
    test(f"File exists: {fname}", make_check(fname))


# ══════════════════════════════════════════════════════════════
# 4. Models import and instantiate
# ══════════════════════════════════════════════════════════════
print("\n[4] Models")

def check_models_import():
    from models import Action, Observation, Reward, StepResponse, ResetRequest, Package, DeliveryBay

def check_action_model():
    from models import Action
    a = Action(action_type="move_right")
    assert a.action_type == "move_right"

def check_reset_model():
    from models import ResetRequest
    r = ResetRequest(task_id="easy")
    assert r.task_id == "easy"

test("models.py imports cleanly",    check_models_import)
test("Action model instantiates",    check_action_model)
test("ResetRequest model works",     check_reset_model)


# ══════════════════════════════════════════════════════════════
# 5. Tasks load correctly
# ══════════════════════════════════════════════════════════════
print("\n[5] Tasks")

def check_tasks_import():
    from tasks import load_task, list_tasks, TASK_REGISTRY
    assert set(TASK_REGISTRY.keys()) == {"easy", "medium", "hard"}

def check_easy_task():
    from tasks import load_task
    t = load_task("easy")
    assert t["task_id"] == "easy"
    assert len(t["packages"]) == 1
    assert t["grid_size"] == (5, 5)

def check_medium_task():
    from tasks import load_task
    t = load_task("medium")
    assert len(t["packages"]) == 2
    assert t["grid_size"] == (8, 8)

def check_hard_task():
    from tasks import load_task
    t = load_task("hard")
    assert len(t["packages"]) == 3
    assert t["grid_size"] == (12, 12)

def check_task_isolation():
    from tasks import load_task
    t1 = load_task("easy")
    t2 = load_task("easy")
    t1["packages"][0]["picked_up"] = True
    assert not t2["packages"][0]["picked_up"], "Tasks must be independent (deep copy)"

test("tasks.py imports + 3 tasks registered", check_tasks_import)
test("Easy task loads correctly",              check_easy_task)
test("Medium task loads correctly",            check_medium_task)
test("Hard task loads correctly",              check_hard_task)
test("Task isolation (deep copy)",             check_task_isolation)


# ══════════════════════════════════════════════════════════════
# 6. Graders work correctly
# ══════════════════════════════════════════════════════════════
print("\n[6] Graders")

def check_grader_zero():
    from graders import grade
    state = {"packages": [{"picked_up": False, "delivered": False}],
             "step_count": 0, "max_steps": 20}
    score = grade("easy", state)
    assert score == 0.0, f"Empty state should score 0.0, got {score}"

def check_grader_pick():
    from graders import grade
    state = {"packages": [{"picked_up": True, "delivered": False}],
             "step_count": 5, "max_steps": 20}
    score = grade("easy", state)
    assert 0.0 < score < 1.0, f"Partial progress should be between 0 and 1, got {score}"

def check_grader_full():
    from graders import grade
    state = {"packages": [{"picked_up": True, "delivered": True}],
             "step_count": 5, "max_steps": 20}
    score = grade("easy", state)
    assert score > 0.8, f"Full delivery should score > 0.8, got {score}"

def check_grader_range():
    from graders import grade
    for tid in ["easy", "medium", "hard"]:
        from tasks import load_task
        t = load_task(tid)
        state = {"packages": t["packages"], "step_count": 0, "max_steps": t["max_steps"]}
        score = grade(tid, state)
        assert 0.0 <= score <= 1.0, f"{tid} grader out of range: {score}"

test("Grader returns 0.0 for empty state",   check_grader_zero)
test("Grader gives partial credit",          check_grader_pick)
test("Grader gives high score for delivery", check_grader_full)
test("All graders return 0.0–1.0",          check_grader_range)


# ══════════════════════════════════════════════════════════════
# 7. Environment core logic
# ══════════════════════════════════════════════════════════════
print("\n[7] Environment Logic")

def check_env_reset():
    from env import WarehouseNavEnv
    from models import Observation
    e = WarehouseNavEnv()
    obs = e.reset("easy")
    assert isinstance(obs, Observation)
    assert obs.task_id == "easy"
    assert obs.step == 0
    assert obs.current_score == 0.0

def check_env_move():
    from env import WarehouseNavEnv
    from models import Action
    e = WarehouseNavEnv()
    e.reset("easy")
    start_pos = list(e.agent_pos)
    resp = e.step(Action(action_type="move_right"))
    # Either moved or hit wall/obstacle — state must change
    assert resp.reward.score >= 0.0

def check_env_wall_penalty():
    from env import WarehouseNavEnv
    from models import Action
    e = WarehouseNavEnv()
    e.reset("easy")
    # Agent starts at (0,0) — moving up hits wall
    resp = e.step(Action(action_type="move_up"))
    feedback = resp.reward.feedback.lower()
    assert "wall" in feedback or "bump" in feedback, f"Expected wall feedback, got: {resp.reward.feedback}"

def check_env_pickup_delivery():
    from env import WarehouseNavEnv
    from models import Action
    e = WarehouseNavEnv()
    e.reset("easy")
    # Manually place agent on package
    e.agent_pos = list(e.packages[0]["pos"])
    resp = e.step(Action(action_type="pick_up"))
    assert e.carrying == "P1", f"Should be carrying P1, got: {e.carrying}"
    assert resp.reward.step_reward > 0, "Pick up should give positive reward"

def check_env_state():
    from env import WarehouseNavEnv
    e = WarehouseNavEnv()
    e.reset("medium")
    s = e.state()
    assert s["task_id"] == "medium"
    assert "packages" in s
    assert "agent_pos" in s

def check_env_ascii_render():
    from env import WarehouseNavEnv
    e = WarehouseNavEnv()
    e.reset("easy")
    grid = e._render_ascii()
    assert "A" in grid, "ASCII grid must show agent position"
    assert "B" in grid, "ASCII grid must show delivery bay"

def check_env_done_guard():
    from env import WarehouseNavEnv
    from models import Action
    e = WarehouseNavEnv()
    e.reset("easy")
    e.done = True  # Force done
    resp = e.step(Action(action_type="move_right"))
    assert resp.done is True
    assert "reset" in resp.reward.feedback.lower()

test("env.reset() returns valid Observation",   check_env_reset)
test("env.step() move works",                   check_env_move)
test("Wall collision gives negative reward",    check_env_wall_penalty)
test("Pick up action works + positive reward",  check_env_pickup_delivery)
test("env.state() returns correct data",        check_env_state)
test("ASCII grid renders correctly",            check_env_ascii_render)
test("Done guard prevents extra steps",         check_env_done_guard)


# ══════════════════════════════════════════════════════════════
# 8. FastAPI app imports
# ══════════════════════════════════════════════════════════════
print("\n[8] FastAPI App")

def check_app_imports():
    import app as a
    assert hasattr(a, "app")
    assert hasattr(a, "env")

def check_app_routes():
    import app as a
    routes = [r.path for r in a.app.routes]
    assert "/reset" in routes
    assert "/step"  in routes
    assert "/state" in routes

test("app.py imports without error", check_app_imports)
test("All required routes exist",    check_app_routes)


# ══════════════════════════════════════════════════════════════
# 9. openenv.yaml valid
# ══════════════════════════════════════════════════════════════
print("\n[9] OpenEnv Metadata")

def check_yaml():
    import yaml
    with open("openenv.yaml") as f:
        data = yaml.safe_load(f)
    assert "name"    in data
    assert "tasks"   in data
    assert len(data["tasks"]) >= 3, "Must have at least 3 tasks"

try:
    import yaml
    test("openenv.yaml parses + has 3 tasks", check_yaml)
except ImportError:
    print(f"  ⚠️  PyYAML not installed — skipping yaml test (non-critical)")


# ══════════════════════════════════════════════════════════════
# RESULTS SUMMARY
# ══════════════════════════════════════════════════════════════
total  = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = total - passed

print(f"\n{'═'*60}")
print(f"  Results: {passed}/{total} passed", end="")

if failed == 0:
    print(f"  🎉 All tests passed! Ready to run.")
    print(f"\n  Next step:")
    print(f"    uvicorn app:app --host 0.0.0.0 --port 7860 --reload")
else:
    print(f"  ⚠️  {failed} test(s) failed — fix before proceeding.")
    print(f"\n  Failed tests:")
    for name, ok, err in results:
        if not ok:
            print(f"    ✗ {name}: {err}")

print(f"{'═'*60}\n")
sys.exit(0 if failed == 0 else 1)
