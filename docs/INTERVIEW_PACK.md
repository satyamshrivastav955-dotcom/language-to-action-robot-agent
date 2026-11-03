# Interview pack

Every number below is in [BENCHMARKS.md](../BENCHMARKS.md) and reproducible from this repo. Success is always scored by simulator ground truth that the agent never reads.

## 1. The two-minute pitch

Most "LLM robot" demos report success from the model's own text. I wanted to know what actually changes when a robot checks its own work from what it can see, so I built the smallest system that can answer that and measured it.

A local Qwen3-VL model plans an instruction into skills. A scripted controller executes them in MuJoCo. Then a verifier that has only an overhead RGB-D camera, with sensor noise and no simulator state, decides whether the block ended up where it should, and if not, measures the error vector and retries.

The hard part was the experiment, not the loop. A perfect scripted arm never fails, so there is nothing to verify; I inject faults (random grasp and release error, a constant release miscalibration, mid-carry slips) into the arm, never into the camera, and I score every episode against ground truth. Arms see identical faults on the first attempt, so comparisons are paired.

Results over 1,800 episodes: under random error, a camera verifier lifts success from 34% to 66%; after slips, from 52% to 78%. Under a systematic 5.4 cm release bias, feeding the measured error into the retry takes success from 33% to 98%, where blind retry stays at 33%. The same protocol on ManiSkill3's PickCube gives 0% to 83% for the biased case. And the honest part: under random noise the error-corrected retry does worse than a blind one, and on ManiSkill the verifier loses a few episodes to occlusion. The camera verifier agreed with ground truth on 2,314 of 2,318 verdicts.

Before measuring anything I also found my inverse kinematics failed from the home pose and the wrist joints chattered at 30 rad/s; fixing those took stack success from 2 of 8 to 8 of 8 with no faults. That is why I trust the baseline.

## 2. Whiteboard architecture

Draw left to right, one loop:

```
Instruction -> Planner (local LLM) -> subtasks
                                        |
                                        v
        +------------------------> Skill (waypoints + IK) -- acts on -->  Simulator
        |                                                                   |
        |  retry with correction = -alpha * measured error                  | (ground truth,
        |                                                                   |  never read by agent)
        |                                                                   v
        +-- not done -- Verifier <-- estimates block pose <-- RGB-D camera (noisy)
                           | done
                           v
                     next subtask

   outside the loop, dashed:  Oracle ---> scores the final world state only
```

Say these four things while drawing:
1. **Sensing boundary.** The agent's only observation is `capture_rgbd`; poses come from colour mask + depth back-projection (1.5 mm median). Ground truth sits outside the loop.
2. **Where faults go.** Into actuation (where the arm actually grasps and releases) and the carry (slip), seeded by (episode, subtask, attempt) so arms are paired.
3. **Two retries.** Blind: repeat, re-perceiving the block. Corrected: shift the next release by `-alpha x error`, only when the gripper held the block and it landed within 12 cm of the target.
4. **Arms.** none / camera+blind / camera+corrected / camera+damped / oracle verifier. The oracle arm bounds how much perception quality matters.

## 3. Fifteen hard questions

**1. What is the claim, in one sentence?**
Verifying outcomes from camera pixels raises task success whenever execution can fail silently, and feeding the measured error back into the retry adds a large gain only when the error is systematic; both are measured against ground truth.

**2. Your arm is scripted and you inject the faults. Isn't this circular?**
Partly, by design, and the README says so. A perfect scripted arm has nothing to verify, so some fault model is unavoidable. What keeps it honest: faults hit the arm and never the camera or scorer; their sizes are fixed and published; all arms see identical attempt-1 faults; and I report where the method loses. The conclusion is conditional: verification helps if faults exist and are detectable, and correction helps if they are systematic. It is not a claim about real-robot failure rates.

**3. What failed?**
Three things. (a) Corrected retry loses to blind retry under random noise, 52% vs 66% (p = 0.008): it chases noise. A damped variant (alpha 0.4) recovers some (58%) but I did not tune further. (b) After a mid-carry slip, the corrected retry treats a drop as a calibration error and is slightly worse (72% vs 78%, p = 0.06). (c) On ManiSkill PickCube, with no faults, the verifier arms lose 2-3 of 30 episodes because the gripper occludes the held cube and the verifier says "not visible". Also, early on, the baseline itself was broken: IK stalled from the home pose and the wrist chattered, so my first "nominal" stack success was 2 of 8.

**4. How do you know the verifier isn't just reading ground truth?**
It cannot: the agent object holds a `PerceptionSystem` whose only input is `env.capture_rgbd` (RGB, depth, calibration). Ground truth is read by `bench/` after the episode and by the explicitly named oracle-verifier arm. A test moves a block and checks that the estimate moves; another checks segmentation error under 5 mm. The `final_pos` and per-attempt `oracle_success` fields are logged for scoring and not consumed by the agent.

**5. Why is the oracle verifier not better than the camera one?**
Because perception here is easy by design (saturated colours, overhead camera, 1-2 mm error) and both share one geometric definition of success. 52% vs 54% noisy and 98% vs 97% biased, neither significant. That says perception is not the bottleneck in this scene; it says nothing about clutter or occlusion, which I did not test.

**6. What are the limits?**
Simulation only; a toy 6-DOF arm; scripted skills with a kinematic grasp assist during carry (arm motion, collisions, drop and rest are physics); faults I chose; fixed overhead camera; the arm is teleported out of view before each look. Not comparable to VLA benchmark numbers.

**7. Why not use a VLA like pi0.5 or SmolVLA?**
Different question. I wanted a controlled study of verification, which needs failures I can attribute. A learned policy's failures are entangled and unseeded. The next step in the roadmap is to swap the scripted skill for a LeRobot policy and run the same arms; that would make the numbers comparable to VLA work.

**8. Why ManiSkill3 and not LIBERO?**
Plain LIBERO is saturated for VLAs (about 97%), its robosuite 1.4 pin breaks on native Windows, and it is a policy benchmark. ManiSkill3 runs in a Linux container with CPU simulation and software rendering, so it is reproducible on any machine. I report it separately: 30 seeds on PickCube-v1 with its official success metric.

**9. Your statistics?**
Paired by seed, so I use the exact McNemar test on discordant pairs, plus 95% Wilson intervals per cell. 90 paired episodes per MuJoCo cell, 30 on ManiSkill. Several ManiSkill noisy-profile differences are not significant and I say so.

**10. Why is "no verifier" so bad on biased stacking, 0%?**
The robot places 5.4 cm off, a stacked cube then topples or sits beside the target, and with no check it reports success anyway: 60 of 90 biased episodes were silent failures. Place-in-bin tolerates the bias (the bin is 14 cm wide), which is why pick-and-place is 100% on that profile. The task geometry matters, and the per-task table shows it.

**11. How does the corrected retry decide not to "correct" a dropped block?**
It applies a correction only if the gripper held the block and it landed within 12 cm of the target, because a drop mid-carry says nothing about release calibration. It is a heuristic and imperfect: under slips it still sometimes misfires, which is the 72% vs 78% result.

**12. What would you change?**
Replace the single noisy error with an estimate of the systematic component across attempts (running mean or a consistency check) so correction stops hurting under noise; add occlusion and clutter to the scene; remove the grasp assist; put a wrist camera on the arm instead of teleporting it away; swap in a learned skill.

**13. Why local models and why Qwen3-VL?**
No cloud dependency or credits, and one model family serves planning and, optionally, point grounding. On this 6 GB GPU: 4B-instruct plans 94% of 18 hand-labelled instructions correctly in 3.2 s; the 8B gets the same 94% at 9.4 s; the 2B gets 72%; the regex fallback 67%. The thinking variants are slower (24-31 s) and return unusable output more often (11% of cases for 4B, 61% for 2B). End to end, all planners tie, because my benchmark instructions are regular enough for the regex parser; I say so rather than imply a model effect.

**14. How reproducible is this?**
Seeds fix scenes, faults, sensor noise and arm physics; `python -m bench.run --episodes 30` regenerates the tables in a few minutes on a laptop CPU; replays are re-run and checked against the benchmark outcome. There is a Docker image for MuJoCo and one for ManiSkill. 60 tests run in CI.

**15. Why trust your simulator results at all?**
I tried not to. I debugged the baseline before measuring (IK, wrist chatter, an injected slip that was instantly undone by the gripper re-latching), reported a per-seed paired design, kept an oracle upper bound, published the negative results, and replicated the headline (bias, corrected retry) in an independent simulator. What I would not claim is anything about real hardware.

## 4. Three resume bullets

- Built a closed-loop language-to-action manipulation agent (local Qwen3-VL planner, MuJoCo, RGB-D perception, camera-only verifier) and a paired ablation over 1,800 simulated episodes: camera verification raised success from 34% to 66% under random execution error and from 52% to 78% after mid-carry slips (exact McNemar p < 1e-6); feeding the measured error into retries raised success from 33% to 98% under a systematic release bias.
- Replaced ground-truth state with RGB-D perception (colour segmentation + depth, 1.5 mm median error; Qwen3-VL point grounding 6 mm) and kept ground truth as the scoring oracle only; the camera verifier matched ground truth on 2,314 of 2,318 attempt-level verdicts. Replicated the bias result on ManiSkill3 PickCube in a Docker image (0% to 83%, p = 6e-8) and reported where it lost (random-noise correction, occlusion).
- Found and fixed two latent control bugs before measuring (IK stalls from home, wrist chatter), taking nominal stack success from 2/8 to 8/8; shipped 60 tests with CI, Docker images, typed JSON-schema tools with a stdio MCP server, and a static replay viewer.
