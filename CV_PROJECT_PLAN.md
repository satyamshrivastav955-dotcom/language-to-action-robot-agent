# CV_PROJECT_PLAN.md — Robot Task Agent → Resume-Ready

Working build plan. Execute top-to-bottom over multiple sessions. Each step has an
acceptance test — do not tick a step until the test passes.

Last audited: 2026-07-26 (full code read + live runs against Bedrock and MuJoCo).

---

## 1. Current honest state (condensed audit)

| Component | Status | Ground truth |
| --- | --- | --- |
| Planner (Bedrock GLM-5) | ✅ WORKING | Live test 2026-07-26: `SOURCE: REAL_LLM (Bedrock)`, `used_fallback=False`. LLM itself asked the clarification question for "put the block in the bin". Heuristic `_detect_ambiguity` only runs in the fallback path. |
| Executor (MuJoCo + IK) | ✅ WORKING* | Real MuJoCo, real damped-least-squares IK, 10-phase waypoint controller. *Grasp is a proximity-snap **attach assist** (`env/so101_env.py:369-407`), not friction contact. Docstrings at `agent/executor.py:84-92, 353-358` overclaim "contact, friction". |
| SmolVLA policy | ❌ DEAD + BROKEN | `agent/executor.py:221-291` — only reachable if MuJoCo is NOT installed; treats SmolVLA as a text-gen model (could never work). Sole reason `torch` is a dependency. |
| Verifier Tier A | ✅ WORKING | Reads mjData positions; 5 cm tolerance + bin-height check. `push` verification just delegates to pick_and_place (`agent/verifier.py:167`). |
| Verifier Tier B (VLM) | ⚠️ DECORATIVE | Fully implemented (`_vlm_verify`, Bedrock converse w/ image) but the verdict never affects pass/fail (`agent/verifier.py:82-84`). Default VLM model = GLM-5, likely not multimodal. |
| Retry controller | ⚠️ PARTIAL | Diagnosis uses real physics deltas and genuinely feeds the next attempt. But: `position_offset` is a fixed diagonal nudge, `speed_factor` ≈ no-op in physics mode, `approach_adjustment` consumed by nothing, `grasp_slip` unreachable, `RetryDiagnostic` class dead. |
| State manager | ❌ THEATER | `env.reset()` at the start of EVERY subtask (`agent/executor.py:399`) teleports all blocks home. Multi-step tasks never physically compose. `object_memory` is written but never read. **Biggest integrity gap.** |
| Trace logger | ✅ WORKING | Clean JSONL, honest `plan_source`/`outcome_source` labels. Minor: crashes on bare filename, two clock bases, 41 stale local logs + `outputs/logs/archive_prefix/` junk dir. |
| Clarification | ✅ CLI / ⚠️ dashboard | CLI: tested end-to-end (ask → answer → re-plan → stop if still ambiguous). Dashboard: banner says "edit the instruction and resubmit" — works, not conversational. |
| Dashboard | ⚠️ MOSTLY | Streams plan/log/video. Module-level shared agent = concurrency hazard on Spaces. `gradio>=4.0.0` pin vs Gradio-6-only `launch(css=, theme=)` call. Single video slot (later subtasks overwrite earlier). |
| Deployment | ⚠️ BLOCKED | No hardcoded Windows paths. `MUJOCO_GL=egl` set only in root `app.py`. HF Space pushed but PAUSED (account cpu-basic quota = 0). `vercel.json` and `_pack.py` are junk. `outputs/demo_strip.png` is an LFS pointer (README hero breaks without LFS). |
| requirements.txt | ❌ BLOATED | torch/transformers/accelerate/einops/safetensors/huggingface-hub/pillow are unused (~5 GB). `imageio-ffmpeg` is needed but missing. |
| Fresh bugs | — | `RobotTaskAgent(use_mock_policy=True)` default = coin-flip mock; contradictory comments `executor.py:190-197`; write-only `is_physics_backed_render` typo `so101_env.py:535`; `max_retries=0` impossible (`or` fallthrough, `task_agent.py:127`); `sys.path.insert` hack in all 7 modules. |

---

## 2. Target end state ("done, resume-ready")

- **Planner**: Bedrock LLM decomposition with honest labelled fallback (already true). Clarification handled conversationally in both CLI and dashboard.
- **Executor**: multi-step instructions physically compose — state persists across subtasks within one instruction. Grasping is either (a) real contact-based, or (b) attach-assist that is accurately described everywhere (no "friction" claims). No dead policy code.
- **Verifier**: Tier A ground truth drives retries; Tier B VLM (multimodal model) produces a verdict that is recorded AND cross-checked against Tier A, with disagreements logged — a genuinely defensible "two-tier verification" claim.
- **Retry**: adjustments derived from the actual miss vector, demonstrably changing outcomes (a seeded failure that succeeds on retry).
- **Dashboard**: end-to-end in a browser: plan populates live, per-subtask videos, clarification answerable in-place, no crash under one concurrent user; honest banners retained.
- **Deployment**: one public URL that works (HF Space running, or clearly-labelled local-only + demo video). `git clone && pip install -r requirements.txt && python run_agent.py "..."` works on a clean Linux machine in < 5 min of installs.
- **Documentation**: README with real GIF, architecture diagram, no false claims; every sentence you'd say in an interview is verifiable in the code; tests + CI green badge.

---

## 3. Build steps (dependency order)

### Phase A — Integrity (do first; everything later builds on a clean core)

**Step 1. Delete junk & dead code** — ~1.5 h
Remove: `_pack.py`, `vercel.json`, `SmolVLAPolicy` + top-level `torch` import + both pynvml warning filters, `RetryDiagnostic`, `approach_adjustment`, `Executor.capture_frame`, unused `Tuple` imports, `outputs/logs/archive_prefix/`, stale local logs/videos. Fix `is_physics_backed_render` typo and contradictory comments at `executor.py:190-197`.
✅ *Accept:* `grep -rn "torch\|SmolVLAPolicy\|RetryDiagnostic" agent/ run_agent.py` returns nothing; `python run_agent.py "put the red block in the bin" --no-videos` still succeeds.

**Step 2. Trim requirements.txt** — ~0.5 h (needs Step 1)
Keep: boto3, numpy, opencv-python, imageio, **imageio-ffmpeg (add)**, gradio (pin `>=6.0`), mujoco. Delete the rest.
✅ *Accept:* fresh venv → `pip install -r requirements.txt` → CLI run succeeds and a video file is written.

**Step 3. Multi-step state persistence — THE key fix** — ~3–4 h
Remove `env.reset()` from `Executor.run_subtask` (`executor.py:399`); reset once per *instruction* in `RobotTaskAgent.run` instead. Keep reset-per-attempt semantics ONLY for retrying the same subtask — decide and document: either restore a snapshot of the pre-subtask state (use `mj_getState`/`mj_setState` or qpos/qvel copy) or re-run from current state. Delete the post-run `env.reset()` at `task_agent.py:218`. Make `ScriptedMuJoCoPolicy` read block positions from the *live* env at subtask start (it already re-reads poses — verify). Wire `state_manager.object_memory` reads into logging so it reflects reality.
✅ *Accept:* `python run_agent.py "put the red block in the bin, then stack the blue block on the green block" --no-videos` — final trace shows red_block still at bin (~[0.5, 0, 0.06]) AND blue_block on green_block. A snapshot-restore retry test also passes.

**Step 4. Honesty pass on claims** — ~1 h (needs Step 3)
Fix executor/env docstrings claiming "contact, friction and gravity" during carry; make `describe_policy()` note the attach assist explicitly; default `use_mock_policy=False` in `RobotTaskAgent` and `create_agent`; fix `max_retries=0` fallthrough (`is None` check); fix trace_logger bare-filename crash.
✅ *Accept:* `grep -n "friction" agent/executor.py` shows no overclaims; `RobotTaskAgent()` with no args runs physics; `TraceLogger("t.jsonl")` doesn't crash.

**Step 5. Commit + tag baseline** — ~0.5 h (needs 1–4)
✅ *Accept:* `git tag v1.0-integrity` pushed to GitHub; clean `git status`.

### Phase B — Impressive (order matters within; B1 → B2 → B3; B4/B5 parallel)

**Step 6 (B1). Miss-vector-driven retries** — ~4 h (needs Step 3)
On placement failure compute the actual XY error vector (final − target, already available in verifier details) and pass `position_offset = -error` capped at 2 cm; make grasp failures adjust grasp XY toward the block's true position; remove `speed_factor` or make it actually slow the waypoint phases. Add a demo mode that injects a deliberate 2 cm grasp offset on attempt 1 so retry success is showable.
✅ *Accept:* seeded-failure run: attempt 1 FAILS with diagnosed kind, attempt 2 SUCCEEDS, trace shows the computed offset. This is your best interview artifact.

**Step 7 (B2). Real Tier-B VLM verification** — ~4–6 h (needs Step 5)
Point `VLM_MODEL_ID` at a multimodal Bedrock model (e.g. `anthropic.claude-haiku-4-5` if enabled in your account, else any multimodal model available in eu-north-1). Make the VLM verdict count: log agreement/disagreement with Tier A; on disagreement, mark result "disputed" in the trace and dashboard. Enable via `--vlm-verify` and dashboard toggle.
✅ *Accept:* run with `--vlm-verify` shows a real VLM sentence describing the final frame in the trace, and a forced-failure run shows the VLM agreeing with Tier A FAIL.

**Step 8 (B3). Contact-based grasping attempt (timeboxed)** — ~8–12 h, timebox it (needs Step 3)
Try: heavier solver iterations, `solimp/solref` tuning on finger–block contacts, smaller block or padded finger geoms, slower close + squeeze force. If real friction grasping works — huge win, delete the assist. If after the timebox it doesn't: keep the assist, write up the attempt (what you tried, why 40 mm cube + position actuators is solver-hostile) as a documented engineering trade-off — that writeup is itself interview material.
✅ *Accept:* EITHER assist code deleted and pick-place passes 8/10 seeded runs, OR `docs/grasping-notes.md` exists describing the attempts with numbers.

**Step 9 (B4). Dashboard upgrade** — ~5–7 h (needs Step 5; parallel with 6–8)
Conversational clarification: keep pending state, show an answer textbox, merge answer and continue. Per-subtask video gallery instead of one overwritten slot. Fix Gradio pin (`gradio>=6`) and move css/theme so they apply. Guard against concurrent runs (a simple lock + "busy" message).
✅ *Accept:* in a real browser: ambiguous instruction → question appears → answer in the answer box → run completes; both subtask videos visible; second submit during a run gets "busy", not a crash.

**Step 10 (B5). Working public deployment** — ~3–5 h (needs Steps 2, 9)
Unblock HF Space: verify email at huggingface.co/settings, restart Space (code is already pushed; requirements trim from Step 2 will cut build from ~10 min to ~2 min). Confirm EGL rendering works on Spaces; if render fails, add `MUJOCO_GL` fallback chain (egl → osmesa) in `env/so101_env.py`, not just root `app.py`. If Spaces stays blocked: record a 60–90 s screen capture demo video and link it prominently; label README "run locally".
✅ *Accept:* public URL executes "put the red block in the bin" end-to-end with video, OR README links a demo video and says local-only, with zero false "live demo" claims.

### Phase C — Polish (any order, after Phase B)

**Step 11. Tests + CI** — ~4–6 h
pytest: planner fallback parser (5+ instructions), `_detect_ambiguity` cases, verifier geometry (in-bin/on-table/stacked fixtures), retry classifier kinds, IK convergence on 3 reachable targets. GitHub Actions: lint + tests on push (mock env only — no Bedrock creds in CI; skip physics tests if EGL absent or use osmesa).
✅ *Accept:* green Actions badge on GitHub README.

**Step 12. Package properly** — ~2 h
`pyproject.toml`, remove all 7 `sys.path.insert` hacks, `pip install -e .` works, console entry point `robot-agent`.
✅ *Accept:* fresh venv → `pip install -e .` → `robot-agent "stack blue on green"` runs.

**Step 13. README + visuals** — ~4–6 h
Fix LFS/demo_strip.png so the hero image actually renders on GitHub. Add: GIF of a full run, architecture diagram (plan→execute→verify→retry loop), retry-recovery GIF from Step 6, "engineering decisions" section (grasp trade-off, verification design, honest-disclosure design). Remove/update stale claims; add MUJOCO_GL note and state_manager to layout table.
✅ *Accept:* GitHub page renders all images; every README claim maps to code you can point at.

**Step 14. Case-study writeup** — ~3–4 h (optional but high leverage)
One-page "Engineering challenges" doc or blog post: LLM plan grounding + validation, ambiguity thresholding, physics-based verification vs self-reported success, retry diagnosis, the grasping trade-off. Link from README and resume.
✅ *Accept:* doc exists, linked, readable in 5 minutes.

---

## 4. Definition of done — final checklist

- [ ] `python run_agent.py "put the red block in the bin, then stack the blue block on the green block"` composes physically (red stays in bin) — verified in trace
- [ ] A seeded-failure run shows diagnose → adjusted retry → success
- [ ] No dead code: no torch, no SmolVLAPolicy, no RetryDiagnostic, no `_pack.py`/`vercel.json`
- [ ] Fresh Linux/venv install from requirements.txt works, videos render
- [ ] Dashboard works end-to-end in a browser incl. clarification and per-subtask videos
- [ ] Live demo URL works OR README clearly says local-only + demo video linked
- [ ] README has zero false claims; grasp assist and scripted controller disclosed; hero image renders
- [ ] Tests pass in CI with a green badge
- [ ] `pip install -e .` works; no sys.path hacks
- [ ] Every claim I'd make in an interview ("LLM planning", "physics-verified", "diagnostic retries", "two-tier verification") is demonstrable in < 2 minutes on my machine
- [ ] HF token used during deployment has been revoked and rotated

---

## Time budget summary

| Phase | Hours (honest) |
| --- | --- |
| A — Integrity (Steps 1–5) | 7–9 h |
| B — Impressive (Steps 6–10) | 24–34 h |
| C — Polish (Steps 11–14) | 13–18 h |
| **Total** | **~45–60 h** |

If you only have one session: do **Step 3** (state persistence). It's the difference
between "multi-step agent" being true and being a demo trick.
