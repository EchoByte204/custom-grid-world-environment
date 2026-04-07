# 🏭 Warehouse Robot Navigation Environment

> **OpenEnv-compliant** real-world grid-world environment where AI agents learn
> to navigate a warehouse floor — picking up packages and delivering them to
> shipping bays while avoiding obstacles and other workers.

---

## 📌 Why This Is Real-World (Not a Toy)

Warehouse robot navigation is a **billion-dollar industry problem**.
Companies like Amazon, DHL, and Flipkart deploy thousands of autonomous robots
to navigate warehouse floors. This environment models the exact decisions those
robots must make: pathfinding, package pickup, delivery routing, obstacle
avoidance, and multi-step planning.

---

## 🗂️ Project Structure

```
warehouse-nav-env/
│
├── 📄 README.md              ← You are here. Full documentation.
├── 📄 .env.example           ← Copy this to .env and fill in your keys
├── 📄 requirements.txt       ← All Python dependencies
├── 📄 openenv.yaml           ← OpenEnv competition metadata
├── 📄 Dockerfile             ← Container for HF Spaces deployment
│
├── 🐍 models.py              ← Pydantic typed models (Observation/Action/Reward)
├── 🐍 tasks.py               ← 3 task definitions (easy → medium → hard)
├── 🐍 graders.py             ← Deterministic scoring functions (0.0 → 1.0)
├── 🐍 env.py                 ← CORE: Grid world logic, reset/step/state
├── 🐍 app.py                 ← FastAPI HTTP server (exposes env over REST)
├── 🐍 inference.py           ← Baseline LLM agent (runs all 3 tasks, prints scores)
│
├── 🐍 renderer.py            ← Optional: pygame visual renderer (local dev only)
└── 🧪 test_env.py            ← Compatibility + smoke tests (run this first!)
```

---

## 🎯 Tasks

| # | Task ID | Difficulty | Goal | Max Steps |
|---|---------|-----------|------|-----------|
| 1 | `easy` | 🟢 Easy | Navigate 5×5 grid, pick 1 package, deliver it | 20 |
| 2 | `medium` | 🟡 Medium | Navigate 8×8 grid, pick 2 packages, avoid 6 obstacles | 40 |
| 3 | `hard` | 🔴 Hard | Navigate 12×12 grid, pick 3 packages, avoid 15 obstacles, time penalty | 80 |

---

## 🔧 Action Space

| Action | Code | Description |
|--------|------|-------------|
| Move Up | `move_up` | Move agent one cell north |
| Move Down | `move_down` | Move agent one cell south |
| Move Left | `move_left` | Move agent one cell west |
| Move Right | `move_right` | Move agent one cell east |
| Pick Up | `pick_up` | Pick up package at current cell |
| Drop Off | `drop_off` | Deliver package at shipping bay |
| Wait | `wait` | Stay in place (costs -0.01) |

---

## 👁️ Observation Space

```json
{
  "task_id": "easy",
  "step": 3,
  "max_steps": 20,
  "grid_size": [5, 5],
  "agent_pos": [2, 2],
  "packages": [{"id": "P1", "pos": [1, 3], "picked_up": false}],
  "delivery_bays": [{"id": "B1", "pos": [4, 4]}],
  "obstacles": [[0, 2], [3, 1]],
  "carrying": null,
  "delivered_count": 0,
  "total_packages": 1,
  "current_score": 0.15,
  "remaining_issues": ["pick_P1", "deliver_P1"],
  "message": "Moved right. No obstacle."
}
```

---

## 🏆 Reward Function

| Event | Reward |
|-------|--------|
| Successfully picks up package | +0.20 |
| Successfully delivers package | +0.40 |
| Moves closer to target | +0.02 |
| Moves away from target | -0.01 |
| Hits obstacle / wall | -0.05 |
| Wait action | -0.01 |
| All packages delivered | +0.20 bonus |
| Each step taken | -0.005 (efficiency penalty) |

Score is always normalized to **0.0 → 1.0**.

---

## 🚀 Setup Instructions

### Prerequisites

- Python 3.10, 3.11, or 3.12 (recommended: 3.11)
- pip
- Git
- Docker (for deployment)
- Antigravity / VS Code (any IDE)
- Hugging Face account (for deployment)

### Step 1 — Clone & Enter Project

```bash
cd warehouse-nav-env
```

### Step 2 — Create Virtual Environment

```bash
python -m venv venv

# Windows (Powershell)
venv\Scripts\Activate.ps1

# Mac/Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure Environment Variables

```bash
# Copy the example env file
copy .env.example .env       # Windows
cp .env.example .env         # Mac/Linux

# Edit .env and fill in your keys
```

### Step 5 — Run Compatibility Tests FIRST

```bash
python test_env.py
```

All tests must pass before proceeding.

### Step 6 — Start the Server

```bash
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

Visit: http://localhost:7860

### Step 7 — Run Baseline Agent

```bash
python inference.py
```

---

## 🐳 Docker

```bash
docker build -t warehouse-nav-env .
docker run -p 7860:7860 --env-file .env warehouse-nav-env
```

---

## 📊 Expected Baseline Scores

| Task | Expected Score | Notes |
|------|---------------|-------|
| easy | ~0.85–0.95 | LLM handles well |
| medium | ~0.55–0.70 | Pathfinding gets tricky |
| hard | ~0.30–0.50 | Challenges frontier models |

---

## 📡 API Reference

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| GET | `/` | — | Health check |
| POST | `/reset` | `{"task_id": "easy"}` | Start new episode |
| POST | `/step` | `{"action_type": "move_right"}` | Take action |
| GET | `/state` | — | Inspect current state |
| GET | `/tasks` | — | List all tasks |
| GET | `/render` | — | ASCII grid render |

---

## 🔑 Key Files Explained

| File | Why It Matters |
|------|---------------|
| `env.py` | **THE BRAIN** — all grid logic lives here |
| `models.py` | Defines the language agent ↔ env speak |
| `graders.py` | Judges score the agent fairly & deterministically |
| `tasks.py` | Three carefully designed scenarios |
| `inference.py` | **REQUIRED by competition** — must be named exactly this |
| `openenv.yaml` | Competition metadata — validators read this |
