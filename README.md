---
title: NEXUS-1 Robot Task Agent
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

<div align="center">

# 🤖 NEXUS-1: Language-to-Action Robot Agent

### *Closed-Loop Hierarchical Embodied AI with Local LLMs & MuJoCo Physics*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://python.org)
[![Physics](https://img.shields.io/badge/Physics-MuJoCo%203.0%2B-00d4ff.svg?logo=openai&logoColor=white)](https://mujoco.org)
[![Local LLM](https://img.shields.io/badge/Local%20LLM-Ollama%20(Qwen2.5%20%7C%20Llama3.1)-00ff88.svg?logo=ollama&logoColor=white)](https://ollama.com)
[![Dashboard](https://img.shields.io/badge/Interface-Gradio%206-ff7a00.svg?logo=gradio&logoColor=white)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

<br/>

**[Quickstart](#-quickstart)** • **[Architecture](#-system-architecture)** • **[Key Features](#-core-capabilities)** • **[Mission Control UI](#-mission-control-dashboard)** • **[Engineering Disclosure](#-engineering-truth--disclosure)**

<br/>

<!-- Hero Animation -->
<img src="outputs/demo.gif" width="750" alt="MuJoCo 6-DOF Robot Arm Manipulation" style="border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 8px 30px rgba(0,0,0,0.5);"/>

*Real-time 6-DOF manipulator executing pick-and-place with live in-frame HUD telemetry.*

</div>

---

## ⚡ Overview

**NEXUS-1** is an autonomous language-to-action robotic agent that translates high-level natural language instructions into precise, multi-step physical manipulation in **MuJoCo physics**.

Unlike open-loop prompt wrappers, NEXUS-1 implements a **closed-loop Perception-Reasoning-Execution-Verification-Recovery cycle**:
1. **Decomposes** complex instructions into ordered subtasks using an **offline local LLM** (Ollama `qwen2.5:14b` or `llama3.1:8b`).
2. **Executes** smooth end-effector trajectories with a 6-DOF DLS (Damped Least-Squares) kinematic controller.
3. **Verifies** task outcomes directly from raw simulation physics state (`mjData`), not self-reported LLM claims.
4. **Diagnoses & Recovers**: When a grasp slips or a drop drifts, the system computes the exact 3D metric miss vector $\Delta \mathbf{p}$ and dynamically adapts trajectory waypoints to succeed on retry.
5. **Preserves State**: Multi-step instructions physically compose across subtasks without teleporting objects.

---

## 🎛️ Mission Control Dashboard

A cinematic, dark-themed command center built with **Gradio 6**, featuring live telemetry, task progress pulses, disambiguation dialogs, and video streams.

<div align="center">
  <img src="outputs/dashboard.png" width="850" alt="NEXUS-1 Mission Control Dashboard" style="border-radius: 8px; border: 1px solid rgba(0, 212, 255, 0.2); box-shadow: 0 10px 40px rgba(0, 212, 255, 0.1);"/>
</div>

---

## 🔄 Manipulation Sequence

<div align="center">
  <img src="outputs/demo_strip.png" width="850" alt="Approach, Grasp, Transfer, and Release phases" style="border-radius: 6px;"/>
  <p><em>Left to Right: <b>APPROACH</b> &rarr; <b>GRASP</b> &rarr; <b>TRANSFER</b> &rarr; <b>RELEASE IN BIN</b></em></p>
</div>

---

## 🧠 System Architecture

```mermaid
flowchart TD
    User["User Instruction<br/><i>'Put green in bin, then stack red on yellow'</i>"] --> Planner
    
    subgraph "Reasoning Layer (Offline LLM)"
        Planner{"Ollama / Qwen2.5<br/>Task Planner"}
        Planner -->|Ambiguous?| Clarify["Interactive Disambiguation Dialog"]
        Planner -->|Valid Plan| Subtasks["Ordered Subtasks [T-01, T-02]"]
    end
    
    subgraph "Execution & Physics Layer"
        Subtasks --> Controller["DLS Inverse Kinematics Controller"]
        Controller --> MuJoCo["MuJoCo 3.0 Simulation<br/>SO-101 6-DOF Arm"]
    end
    
    subgraph "Verification & Recovery Loop"
        MuJoCo --> Verifier{"Tier-A Physics Verifier<br/>(mjData Ground Truth)"}
        Verifier -->|Success| Persist["Update World State & Advance"]
        Verifier -->|Failure| Diagnostic["Error Vector Diagnostic<br/>Δp = p_target - p_actual"]
        Diagnostic -->|Compensate| Controller
    end
    
    Persist --> Complete["Task Sequence Complete"]
```

---

## 🌟 Core Capabilities

| Feature | Description | Why It Matters |
|:---|:---|:---|
| 🧠 **100% Offline LLM** | Powered by local Ollama (`qwen2.5:14b` or `llama3.1:8b`) with native JSON formatting. | Zero API costs, runs on air-gapped workstations, no cloud dependencies. |
| 🔄 **State Persistence** | World state persists across sequential subtasks with snapshot rollback. | Subtask 2 executes in the physical world left by Subtask 1 (e.g. red stays in bin while blue is stacked). |
| 🎯 **Adaptive Error Recovery** | Measures coordinate miss vectors $\Delta \mathbf{p}$ to adjust retry waypoints. | Automatically converts attempt-1 perturbations into attempt-2 successes. |
| 💬 **Active Disambiguation** | Detects semantic ambiguities (*"move the block"*) and asks questions rather than guessing. | Eliminates silent execution errors in partially specified goals. |
| 📹 **In-Frame HUD Overlays** | Renders live kinematic phases (`APPROACH`, `GRASP`, `LIFT`, `RELEASE`) onto video frames. | Complete visual transparency for debugging and presentation. |

---

## 🚀 Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/satyamshrivastav955-dotcom/language-to-action-robot-agent.git
cd language-to-action-robot-agent

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Local LLM (Ollama)

```bash
# Pull the model (one-time)
ollama pull qwen2.5:14b
# or: ollama pull llama3.1:8b
```

### 3. Run via CLI

```bash
# Single subtask
python run_agent.py "put the green block in the bin"

# Multi-step sequential task
python run_agent.py "put the green block in the bin, then stack red on yellow"

# Ambiguous instruction (agent will proactively clarify)
python run_agent.py "put the block in the bin"
```

### 4. Launch Mission Control Dashboard

```bash
python dashboard/app.py
```
Open **`http://localhost:7860`** in your browser.

---

## 📋 Verified Instruction Matrix

| Prompt | Type | Decomposed Actions | Expected Outcome |
|:---|:---|:---|:---|
| `"put the green block in the bin"` | Single-Step | `pick_and_place(green, bin)` | Green block placed inside bin (sub-cm accuracy) |
| `"put red in bin, then stack blue on green"` | Multi-Step | `pick_and_place(red, bin)` &rarr; `stack(blue, green)` | Red stays in bin; Blue stacked on Green |
| `"put green in bin, then stack red on yellow"` | Multi-Step + Recovery | `pick_and_place(green, bin)` &rarr; `stack(red, yellow)` | Autonomous vector-corrected retry on perturbation |
| `"move the block to the bin"` | Ambiguous | *Clarification Triggered* | Prompts: *"There are 4 blocks on the table. Which one?"* |

---

## 🔍 Engineering Truth & Disclosure

Honest engineering builds trust. Here is what is physics-simulated, and what is scripted:

- **Physics Simulation (Real):** MuJoCo physics engine computes all forward dynamics, arm inertia, collisions, gravity, drop physics, and final resting poses. Success is verified directly from `mjData.qpos` and bounding-box coordinates.
- **Hierarchical Planning (Real):** Natural language decomposition, ambiguity detection, and parameter compensation are handled live by the local LLM.
- **Controller (Deterministic):** The 6-DOF arm is driven by a damped least-squares (DLS) inverse kinematics controller with 11 discrete phases, not an end-to-end learned policy.
- **Grasp Assist (Disclosed):** Pure friction grasping of small rigid cubes in simulation is notoriously sensitive to solver contact damping. When the gripper closes around a block, proximity attach-assist stabilizes the carry phase, and hands the object back to physics for the release and drop.

---

## 📂 Repository Structure

```text
├── agent/
│   ├── planner.py           # Multi-backend LLM planner (Ollama & Bedrock)
│   ├── executor.py          # Waypoint kinematic controller & rollouts
│   ├── verifier.py          # Ground-truth physics verifier
│   ├── retry_controller.py  # Miss-vector diagnostic & adaptive parameter updates
│   ├── state_manager.py     # Multi-step world state & memory tracking
│   ├── task_agent.py        # Central orchestrator loop
│   └── trace_logger.py      # Structured JSONL execution logger
├── configs/
│   └── settings.py          # Workspace coordinates, model settings, & thresholds
├── dashboard/
│   └── app.py               # Mission Control UI (Gradio 6)
├── env/
│   ├── scene.xml            # MuJoCo environment XML (Arm, Table, Blocks, Bin)
│   └── so101_env.py         # Gymnasium-style MuJoCo environment wrapper
├── outputs/
│   ├── demo.gif             # Animated manipulation demo
│   ├── demo_strip.png       # Sequential phase strip
│   └── dashboard.png        # UI screenshot
├── requirements.txt         # Clean dependencies (boto3 optional)
└── run_agent.py             # CLI runner
```

---

<div align="center">
Built with ❤️ for <b>InnovaHack Chapter-1 — Domain 4: Agentic AI</b>
</div>
