---
title: Robot Task Agent
emoji: 🤖
colorFrom: yellow
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
---

# Robot Task Agent

**InnovaHack Chapter-1 — Domain 4: Agentic AI**
*Autonomous Personal Assistant Agent for Multi-Step Real-World Tasks*

An agentic system that takes one natural-language instruction, decomposes it into
ordered sub-tasks, executes each on a 6-DOF robot arm in **real MuJoCo physics**,
verifies the outcome against simulated ground truth, diagnoses failures, retries,
and explains every decision it made.

![Pick and place in MuJoCo](outputs/demo_strip.png)

*Left to right: approach, descend onto the red block, transfer, released in the bin.*

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python run_agent.py "put the red block in the bin"
```

```bash
python dashboard/app.py
```

The dashboard serves on <http://localhost:7860> (it walks forward to 7869 if the
port is busy).

## Demo instructions that work

```bash
python run_agent.py "put the red block in the bin, then stack the blue block on the green block"
```

```bash
python run_agent.py "put the block in the bin"
```

The first runs two sub-tasks end-to-end. The second is deliberately ambiguous —
the agent asks which block rather than guessing, then re-plans on your answer.

## How it works

1. **Plan** — AWS Bedrock (GLM-5) decomposes the instruction into ordered
   sub-tasks. A heuristic keyword parser takes over if Bedrock is unreachable,
   and the run is clearly labelled when that happens.
2. **Clarify** — if the instruction is genuinely ambiguous ("the block", when
   four exist), the agent asks instead of guessing. The bar is deliberately
   high: plural phrasing and any colour word resolve it without a question.
3. **Execute** — an IK-driven waypoint controller drives the arm through
   approach → align → descend → grasp → lift → transfer → descend-over →
   lower → release → retreat, in MuJoCo.
4. **Verify** — success is read from the object's final position in `mjData`,
   never from the policy's own claim. A block still flat on the table cannot
   pass as stacked.
5. **Retry** — failures are diagnosed by kind (grasp slip, placement accuracy,
   release height) and retried with matched adjustments, up to a limit.
6. **Report** — a JSONL trace log plus a summary of what happened and why.

## What is real, and what is not

This matters more than a demo that overclaims, so it is stated plainly:

- **Real:** the MuJoCo physics simulation, the arm's kinematics and IK, contact
  and collision, gravity, the release, and where objects finally come to rest.
  Verification reads actual simulation state.
- **Real:** task decomposition by the LLM, failure diagnosis, and the retry loop.
- **Not learned:** the controller is a scripted waypoint sequence, not a trained
  policy. It does not claim to be — but unlike a mock, its outcome is *not*
  predetermined: a mis-aimed approach genuinely misses and genuinely fails.
- **Assisted:** friction-based grasping of a 40mm cube proved too solver-sensitive
  to tune reliably, so while the gripper is closed the block is held to the grip
  site. It is handed back to the physics engine on release and falls under real
  gravity. `Executor.describe_policy()` reports this at runtime, and the
  dashboard shows it as a banner.

`--mock` swaps in a fully scripted policy with no physics, for a fast
dependency-free run. It is labelled as scripted everywhere it appears.

## Layout

| Path | Role |
| --- | --- |
| `run_agent.py` | CLI front-end |
| `dashboard/app.py` | Gradio front-end |
| `agent/task_agent.py` | The plan → execute → verify → retry loop, shared by both front-ends |
| `agent/planner.py` | Bedrock decomposition, fallback parser, ambiguity check |
| `agent/executor.py` | Waypoint controller and rollouts |
| `agent/verifier.py` | Ground-truth success checks |
| `agent/retry_controller.py` | Failure diagnosis and retry adjustments |
| `agent/trace_logger.py` | JSONL trace log |
| `env/so101_env.py` | MuJoCo environment |
| `env/scene.xml` | Arm, table, four blocks, bin |

## Configuration

`configs/settings.py` holds object positions, target regions, retry limits and
tolerances. Bedrock model and region come from `BEDROCK_MODEL_ID` and
`AWS_DEFAULT_REGION`.
