"""
inference.py
============
Baseline inference script for the Warehouse Robot Navigation Environment.

REQUIRED BY COMPETITION — must be named exactly "inference.py" and placed
in the root of the project.

WHAT THIS DOES:
  1. Connects to the running environment server (ENV_URL)
  2. For each of the 3 tasks (easy, medium, hard):
     a. Resets the environment
     b. Feeds the observation to an LLM (via OpenAI-compatible API)
     c. LLM decides the next action
     d. Sends action to environment
     e. Repeats until done or max_steps reached
  3. Prints a score table at the end

ENVIRONMENT VARIABLES (set in .env or system):
  API_BASE_URL  — LLM API endpoint base URL
  MODEL_NAME    — Model identifier (e.g. gpt-4o-mini)
  HF_TOKEN      — API key / Hugging Face token
  ENV_URL       — URL of the running env server

RUNTIME: Should complete in < 20 minutes on 2 vCPU / 8GB RAM.
"""

import os
import sys
import json
import time
import requests
from openai import OpenAI

# ── Load .env if available ────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Configuration ─────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")

# Validate config
if not HF_TOKEN:
    print("❌ ERROR: HF_TOKEN not set.")
    print("   Set it in your .env file or as an environment variable.")
    sys.exit(1)

# ── LLM Client ────────────────────────────────────────────────
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── System Prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an autonomous warehouse robot agent. Navigate a 2D grid, pick up packages, and deliver them to bays.

Respond with ONLY a raw JSON object — no text, no markdown, no explanation.

CRITICAL DECISION RULES (follow in exact order):

1. CHECK IF YOU SHOULD DROP OFF:
   - If carrying is NOT null AND your agent_pos matches any delivery_bay pos → respond: {"action_type": "drop_off", "parameters": {}}

2. CHECK IF YOU SHOULD PICK UP:
   - If carrying is null AND your agent_pos matches any package pos where picked=false → respond: {"action_type": "pick_up", "parameters": {}}

3. NAVIGATE:
   - If carrying is null: move toward the nearest package that has picked=false
   - If carrying is NOT null: move toward any delivery_bay
   - Plan a path around obstacles (X cells). If a direction is blocked, try another.
   - NEVER repeat the same blocked move twice in a row — try a perpendicular direction.

MOVEMENT ACTIONS:
{"action_type": "move_up", "parameters": {}}     ← row decreases by 1
{"action_type": "move_down", "parameters": {}}   ← row increases by 1
{"action_type": "move_left", "parameters": {}}   ← col decreases by 1
{"action_type": "move_right", "parameters": {}}  ← col increases by 1
{"action_type": "pick_up", "parameters": {}}
{"action_type": "drop_off", "parameters": {}}

GRID: A=you, C=carrying, P=package, B=bay, X=obstacle, .=empty
Your position is [row, col]. Rows increase downward, cols increase rightward.

Always output exactly one JSON object."""


def parse_action(raw_response: str) -> dict:
    """Parse LLM response into a valid action dict."""
    text = raw_response.strip()

    # Strip markdown code fences
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break

    # Find first JSON object
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON found in: {text[:100]}")

    action = json.loads(text[start:end])

    # Ensure parameters key exists
    if "parameters" not in action:
        action["parameters"] = {}

    return action


def get_llm_action(observation: dict, last_action: str = None) -> dict:
    """Ask the LLM what action to take given the current observation."""
    agent_pos = observation["agent_pos"]
    carrying  = observation["carrying"]
    packages  = observation["packages"]
    bays      = observation["delivery_bays"]

    # Build explicit position-match hints so small LLMs don't miss them
    hints = []
    if carrying:
        # Are we on a bay right now?
        on_bay = any(b["pos"] == agent_pos for b in bays)
        if on_bay:
            hints.append(f"⚠️  YOU ARE ON A BAY AND CARRYING {carrying}. You MUST use drop_off NOW.")
        else:
            nearest_bay = min(bays, key=lambda b: abs(b["pos"][0]-agent_pos[0])+abs(b["pos"][1]-agent_pos[1]))
            dr = nearest_bay["pos"][0] - agent_pos[0]
            dc = nearest_bay["pos"][1] - agent_pos[1]
            hints.append(f"You are carrying {carrying}. Nearest bay at {nearest_bay['pos']}. "
                         f"Need to go {'DOWN' if dr>0 else 'UP'} {abs(dr)} rows, "
                         f"{'RIGHT' if dc>0 else 'LEFT'} {abs(dc)} cols.")
    else:
        # Are we on a package right now?
        on_pkg = next((p for p in packages if p["pos"] == agent_pos and not p["picked_up"] and not p["delivered"]), None)
        if on_pkg:
            hints.append(f"⚠️  YOU ARE ON PACKAGE {on_pkg['id']} AT {agent_pos}. You MUST use pick_up NOW.")
        else:
            pending = [p for p in packages if not p["picked_up"] and not p["delivered"]]
            if pending:
                nearest = min(pending, key=lambda p: abs(p["pos"][0]-agent_pos[0])+abs(p["pos"][1]-agent_pos[1]))
                dr = nearest["pos"][0] - agent_pos[0]
                dc = nearest["pos"][1] - agent_pos[1]
                hints.append(f"Not carrying. Target package {nearest['id']} at {nearest['pos']}. "
                             f"Need to go {'DOWN' if dr>0 else 'UP'} {abs(dr)} rows, "
                             f"{'RIGHT' if dc>0 else 'LEFT'} {abs(dc)} cols.")

    hint_str = "\n".join(hints) if hints else ""

    user_msg = f"""
Current warehouse state:
- Task: {observation['task_id']} | Step: {observation['step']}/{observation['max_steps']}
- Score: {observation['current_score']:.3f}
- Agent position [row, col]: {agent_pos}
- Carrying: {carrying or 'nothing (null)'}
- Packages: {[{'id': p['id'], 'pos': p['pos'], 'picked_up': p['picked_up'], 'delivered': p['delivered']} for p in packages]}
- Delivery bays: {[{'id': b['id'], 'pos': b['pos']} for b in bays]}
- Remaining goals: {observation['remaining_issues']}
- Last message: {observation['message']}
- Last action taken: {last_action or 'none (first step)'}

{hint_str}
{"⚠️  AVOID repeating: " + last_action + " — it just failed or got you nowhere!" if last_action and "Hit" in observation.get("message","") else ""}

Grid (A=you, C=carrying, P=package, B=bay, X=obstacle):
{observation.get('ascii_grid', 'N/A')}

Apply CRITICAL DECISION RULES from system prompt. Respond with ONE JSON action object only.
"""

    # Adding retry logic specifically for Groq 429 Rate Limits
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=100,
                temperature=0.1,   # Near-deterministic for reproducibility
            )
            return parse_action(response.choices[0].message.content)
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                print(f"  [Wait] Rate limit hit. Retrying in 8s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(8.0)
            else:
                raise e


def run_task(task_id: str) -> float:
    """Run the agent on one full task episode. Returns final score."""
    print(f"\n{'═'*55}")
    print(f"  TASK: {task_id.upper()}")
    print(f"{'═'*55}")

    # --- REQUIRED OPENENV START FORMAT ---
    print(f"[START] task={task_id} env=warehouse_robot model={MODEL_NAME}")

    # ── Reset environment ─────────────────────────────────────
    resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    obs = resp.json()

    final_score  = 0.0001
    done         = False
    step_num     = 0
    error_count  = 0
    last_action  = None   # Track to detect stuck loops
    rewards_history = []
    MAX_ERRORS   = 5  # Stop if LLM keeps producing invalid actions

    while not done and step_num < obs["max_steps"]:
        step_num += 1

        # ── Deterministic override: pick_up / drop_off ────────
        # Python-side safety net: if we're on a package or bay,
        # don't trust the LLM — just do the right action directly.
        agent_pos = obs["agent_pos"]
        carrying  = obs.get("carrying")
        packages  = obs.get("packages", [])
        bays      = obs.get("delivery_bays", [])

        override_action = None
        if carrying:
            # Check if standing on a bay
            if any(b["pos"] == agent_pos for b in bays):
                override_action = {"action_type": "drop_off", "parameters": {}}
        else:
            # Check if standing on an unpicked package
            for p in packages:
                if p["pos"] == agent_pos and not p["picked_up"] and not p["delivered"]:
                    override_action = {"action_type": "pick_up", "parameters": {}}
                    break

        # ── Get action from LLM (or use override) ────────────
        err_msg = "null"
        try:
            action = override_action if override_action else get_llm_action(obs, last_action)
        except Exception as e:
            error_count += 1
            err_msg = str(e).replace('"', "'")
            if error_count >= MAX_ERRORS:
                action = {"action_type": "wait", "parameters": {}}
                break
            action = {"action_type": "wait", "parameters": {}}

        # ── Send action to environment ────────────────────────
        try:
            step_resp = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
            step_resp.raise_for_status()
            result = step_resp.json()
        except Exception as e:
            err_msg = str(e).replace('"', "'")
            result = None
            break

        if result:
            obs         = result["observation"]
            reward      = result["reward"]
            done        = result["done"]
            final_score = reward["score"]
            step_reward = reward["step_reward"]
        else:
            step_reward = 0.0

        rewards_history.append(f"{step_reward:.2f}")
        last_action = action["action_type"]

        # --- REQUIRED OPENENV STEP FORMAT ---
        action_str = action['action_type']
        print(f"[STEP] step={step_num} action={action_str} reward={step_reward:.2f} done={str(done).lower()} error={err_msg}")

        # Small sleep to avoid rate limiting
        time.sleep(0.3)

    # --- REQUIRED OPENENV END FORMAT ---
    success_bool = obs.get("delivered_count", 0) == obs.get("total_packages", -1)
    rewards_joined = ",".join(rewards_history) if rewards_history else "0.00"
    print(f"[END] success={str(success_bool).lower()} steps={step_num} score={final_score:.4f} rewards={rewards_joined}")

    return final_score


def main():
    """Run all 3 tasks and print summary."""
    print("\n🤖 Warehouse Robot Navigation — Baseline Inference")
    print(f"   Model    : {MODEL_NAME}")
    print(f"   Endpoint : {API_BASE_URL}")
    print(f"   Env URL  : {ENV_URL}")

    # ── Verify environment is running ─────────────────────────
    try:
        health = requests.get(f"{ENV_URL}/", timeout=10)
        health.raise_for_status()
        print(f"   Env status : ✅ Running\n")
    except Exception as e:
        print(f"\n❌ Environment not reachable at {ENV_URL}")
        print(f"   Error: {e}")
        print("\n   Fix: Run this in another terminal:")
        print("   uvicorn app:app --host 0.0.0.0 --port 7860")
        sys.exit(1)

    # ── Run all tasks ─────────────────────────────────────────
    scores = {}
    for task_id in ["easy", "medium", "hard"]:
        try:
            scores[task_id] = run_task(task_id)
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user.")
            break
        except Exception as e:
            print(f"  ❌ Task '{task_id}' crashed: {e}")
            scores[task_id] = 0.0

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print("  BASELINE SCORES SUMMARY")
    print(f"{'═'*55}")
    for tid, score in scores.items():
        filled = int(score * 25)
        bar    = "█" * filled + "░" * (25 - filled)
        print(f"  {tid:8s} │ {bar} │ {score:.2f}")

    if scores:
        avg = sum(scores.values()) / len(scores)
        print(f"  {'AVERAGE':8s} │ {'─'*25} │ {avg:.2f}")

    print(f"{'═'*55}\n")


if __name__ == "__main__":
    main()
