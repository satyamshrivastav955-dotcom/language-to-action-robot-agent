# Decisions

Newest first. Each entry: context, decision, consequence.

## Report the negative results, and keep the arms that produce them

**Context.** Corrected retry is not uniformly better: under random noise it chases noise (52% vs 66% for blind retry, p=0.008), and after a mid-carry slip it can "correct" for an error that is not a calibration error.
**Decision.** Keep `verifier_corrected` as the headline arm, add a damped variant (alpha 0.4) as a sensitivity check, and publish both. Parameters were fixed before the full run; none were tuned on the benchmark seeds afterwards.
**Consequence.** The claim is narrow and defensible: verification pays under every fault profile with a failure to detect; error-corrected retry pays when the error is systematic.

## Faults are injected, and the arm differs from the evaluator

**Context.** The scripted controller is accurate and the simulator deterministic, so with no faults every task succeeds on attempt 1 and a verifier has nothing to catch.
**Decision.** `env/faults.py` perturbs actuation (grasp/release position noise, a constant release bias, mid-carry slip) and never the camera or the evaluator. Faults come from a generator seeded by (episode seed, subtask, attempt), so all arms see identical faults on attempt 1 and diverge only through their own decisions. Comparisons are paired (exact McNemar).
**Consequence.** Gains are conditional on the fault sizes I chose; the profiles are published next to every table.

## Perception from pixels; ground truth only scores

**Decision.** The agent's only sensor is an overhead RGB-D camera with noise (`env.capture_rgbd`). Colour segmentation + depth is the default estimator (1-2 mm error in the clean scene); a local VLM (Qwen3-VL via Ollama) is the alternative 2-D grounder, lifted to 3-D by the same depth code. Simulator state is read only by `bench/` after an episode, and by the "oracle verifier" arm, which exists as an upper bound.
**Consequence.** `execution_result["final_pos"]` and the per-attempt `oracle_success` are logged for evaluation but never read by the agent.

## The arm is parked, instantly, before every look

**Decision.** `park_arm()` teleports the arm to an upright stow pose so the overhead camera sees the table. This stands in for "retreat out of view"; its time cost is not modelled.
**Consequence.** Disclosed in the viewer and README. A real system would need a retreat motion or a second camera.

## Retries do not rewind the world

**Context.** The first version restored the pre-subtask state before each retry, which erases the failure the robot is supposed to react to.
**Decision.** `rewind_on_retry=False` by default; a retry starts from the world the failed attempt left, re-perceived. The block that dropped is re-grasped where it landed.

## Fixed two latent bugs before measuring anything

- **Inverse kinematics from the home pose failed** for most targets (wrist pinned at its limit). Earlier "successes" depended on the arm drifting into a good branch. `solve_ik` now falls back to a canonical hover seed, then yaw-aligned random restarts, choosing the converged solution nearest the live pose. Nominal success went from 2/8 to 8/8 on stacks.
- **Wrist joints chattered at about 30 rad/s** (armature 0.01 too low for kp/kv), flinging blocks on release. Armature is 0.05. Gripper travel was also burying the pads 6.5 mm in the cube; it now closes to the cube face.
- An injected slip was instantly undone because the gripper re-latched the falling block; a slipped grasp now cannot re-latch until the gripper opens.

## Standard benchmark: ManiSkill3 in Docker (not LIBERO)

**Context.** LIBERO is a VLA-policy benchmark that plain VLAs score about 97% on (Bifrost, Oct 2026), and its robosuite 1.4.0 pin breaks on native Windows (lerobot issue 4857). ManiSkill 3.0.1 on Windows has CPU sim only.
**Decision.** ManiSkill3 tasks in a Linux Docker image (CPU sim, software Vulkan rendering), reported separately from the in-repo MuJoCo tables. See `maniskill_bench/` and BENCHMARKS.md. Two environment issues cost time and are worth knowing: SAPIEN rejects ManiSkill's "cpu" render device name (the run script patches `RenderSystem` to use the default llvmpipe device), and mplib 0.1.1 segfaults under NumPy 2 (the image pins `numpy<2`). The camera is rendered on demand (`obs_mode=state` for stepping) because rendering every step on a software Vulkan device is far too slow.

## Local planner default: Qwen3-VL 4B

**Decision.** `qwen3-vl:4b-instruct` (3.3 GB at Q4) is the default Ollama model: it fits a 6 GB GPU, one model family serves planning and grounding, and the non-thinking variant answers in about 3 s. The thinking variants (`qwen3-vl:4b`) answered in 24 s and returned unusable output in 11% of cases; the old `qwen2.5:14b` default did not fit the GPU.
**Consequence.** `num_predict` is 3072 so a thinking model can still finish. All sizes are compared in BENCHMARKS.md.

## Packaging

`pyproject.toml` with an editable install replaces the `sys.path.insert` hacks. Dashboard: `python -m dashboard.app`.

## Static viewer, not a served app

**Decision.** `web/` is plain HTML/CSS/JS plus recorded JSON/MP4 (H.264). `vercel.json` publishes `web/` as the output directory. `.gitattributes` exempts `web/**` from Git LFS, because Vercel does not fetch LFS objects and would serve pointer files.
**Consequence.** It cannot 404 or run out of credits on model calls; it only ever serves files that exist in the repo.

## Not done, on purpose

- **LeRobot dataset compatibility:** not attempted. A faithful v3 dataset export needs verification against the current LeRobot release, which I did not do.
- **ROS 2 bridge:** `adapters/ros2_bridge.py` is a short, untested starting point (no ROS 2 install available).
- **Cloud/Bedrock paths removed.** The Bedrock planner and the Bedrock VLM second-opinion verifier were dead or decorative (the verdict never affected pass/fail) and contradicted the local-only goal; the camera verifier replaced them.
