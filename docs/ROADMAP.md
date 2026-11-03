# Roadmap

Wedge: a closed-loop agent that verifies physical outcomes from **perception** and retries from the **measured error**, with a measured success-rate gain from the verifier.
Proof: the ablation in [../BENCHMARKS.md](../BENCHMARKS.md). Why each choice: [DECISIONS.md](DECISIONS.md).

Last updated: 2026-11-03.

## State

| Area | State |
|---|---|
| Perception from pixels | Done. Overhead RGB-D + colour segmentation (1-2 mm clean). Qwen3-VL point grounding is an alternative (see BENCHMARKS.md). |
| Ground truth as oracle only | Done. Read by `bench/` and the "oracle verifier" arm; never by the default agent. |
| Evaluation harness + ablations | Done. 4 fault profiles x 3 tasks x 5 arms x 30 seeds, paired exact tests. |
| Local quantised planners | Done. Qwen3-VL 2B/4B/8B (instruct and thinking) via Ollama; accuracy, latency and end-to-end success compared. |
| Standard benchmark | Done for ManiSkill3 PickCube-v1 in Docker (30 seeds, 3 profiles, 3 arms). The bias result replicates; verification loses a few episodes to occlusion. LIBERO not attempted. |
| Typed tools | Done. JSON-schema registry + stdio MCP server, tested. |
| ROS 2 / LeRobot | ROS 2 bridge written, untested. LeRobot not attempted. |
| Tests / CI | `pytest` (fast + physics), `ruff`; GitHub Actions green on `main`. |
| Replay viewer | Static site in `web/`, deployed on Vercel (`vercel.json` + `.vercelignore` publish only `web/`). |
| Docker | Root image for the MuJoCo benchmark; separate image for ManiSkill3. |

## Next (in order of value)

1. **Learned skills.** Replace the scripted controller with a policy (for example a LeRobot SmolVLA or Pi0.5 checkpoint) and re-run the same arms. This is the experiment that would make the numbers comparable to VLA work.
2. **A better retry estimator.** The corrected retry loses under random noise. Estimate the systematic component across attempts (running mean, or a consistency check between consecutive errors) instead of feeding back one noisy error.
3. **Harder perception.** Clutter, occlusion, lighting changes, and a wrist camera; report verifier false-accept rates there.
4. **Remove the grasp assist** (contact-based grasping), which narrows what can fail.
5. **LeRobot dataset export** of recorded episodes, verified against the current LeRobot release.
6. **LIBERO-Plus** with a VLA + verifier wrapper, on Linux/WSL2.

## Known issues

- The Gradio dashboard was exercised end to end through the browser DOM, but its visual layout has not been reviewed by eye.
