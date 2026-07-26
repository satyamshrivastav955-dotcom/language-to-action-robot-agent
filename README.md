# language-to-action-robot-agent
Agentic AI system that decomposes natural-language instructions into robot  pick-and-place sub-tasks, executes them via IK-driven control in a real MuJoCo  physics simulation, verifies outcomes, retries on failure, and explains its  reasoning — built for InnovaHack Chapter-1 (Agentic AI track).

# Robot Task Agent

**InnovaHack Chapter-1 — Domain 4: Agentic AI**
*Autonomous Personal Assistant Agent for Multi-Step Real-World Tasks*

An agentic AI system that takes a single high-level natural-language instruction, 
autonomously decomposes it into sub-tasks, executes each one via a real robot-arm 
controller in MuJoCo physics simulation, verifies outcomes against ground-truth 
state, retries intelligently on failure with diagnosed causes, and reports a 
clear, transparent summary of what it did and why.

## What it does

Give it an instruction like:
> "Put the red block in the bin, then stack the blue block on the green block."

The agent will:
1. **Plan** — decompose the instruction into ordered sub-tasks (AWS Bedrock / GLM-5, 
   with a heuristic fallback parser if the LLM is unavailable)
2. **Execute** — drive a 6-DOF robot arm via inverse kinematics in a real MuJoCo 
   physics simulation to perform each pick-and-place sub-task
3. **Verify** — check success against the object's real physics-simulated position, 
   not assumptions
4. **Retry** — on failure, diagnose the likely cause (grasp slip, timing, placement 
   accuracy) and retry with adjustments, up to a configurable limit
5. **Report** — produce a human-readable trace log and final summary explaining 
   every decision made

## Architecture
