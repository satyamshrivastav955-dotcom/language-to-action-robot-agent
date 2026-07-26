"""Gradio front-end.

Rendering only: the plan/execute/verify/retry loop lives in
agent/task_agent.py and is shared with the CLI. This file previously carried a
second copy of that loop, which drifted from the original.
"""

import os
import sys
import warnings

# See run_agent.py - silence torch's deprecated-pynvml FutureWarning.
warnings.filterwarnings("ignore", message=".*pynvml.*")
from typing import Dict, Generator, List

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.task_agent import RobotTaskAgent
from configs.settings import (
    MAX_RETRIES_PER_SUBTASK,
    MAX_STEPS_PER_ROLLOUT,
    VIDEOS_DIR,
    BEDROCK_MODEL_ID,
)

TASK_STATUS_HTML = """
<style>
    .status-pending {{ color: #888; }}
    .status-in_progress {{ color: #2196F3; font-weight: bold; }}
    .status-completed {{ color: #4CAF50; font-weight: bold; }}
    .status-failed {{ color: #f44336; font-weight: bold; }}
    .subtask-item {{ margin: 8px 0; padding: 8px; border-radius: 4px; background: #f5f5f5; }}
    .attempt {{ color: #666; font-size: 0.9em; }}
    .disclosure {{ margin: 8px 0; padding: 10px; border-radius: 6px;
                   background: #fff8e1; border-left: 4px solid #ffb300;
                   color: #5d4037; font-size: 0.9em; }}
    .disclosure.ok {{ background: #e8f5e9; border-left-color: #43a047;
                      color: #1b5e20; }}
</style>
{content}
"""

STATUS_ICONS = {"pending": "○", "in_progress": "◐", "completed": "✓", "failed": "✗"}


agent = RobotTaskAgent(use_mock_policy=False, console_output=False)


def _disclosure_html() -> str:
    """Banners stating how outcomes and plans are really produced.

    Two independent things can be simulated, so they get two banners: the
    policy (physics vs scripted teleport) and the planner (LLM vs heuristic
    fallback). Neither should be able to pass silently as the real thing.
    """
    parts = []

    info = agent.executor.describe_policy()
    if info.get("outcomes_are_scripted"):
        parts.append(f"<div class='disclosure'><strong>Simulated outcomes.</strong> "
                     f"{info.get('note', '')}</div>")
    elif info.get("physics_backed"):
        parts.append("<div class='disclosure ok'><strong>MuJoCo physics.</strong> "
                     "Outcomes are determined by the simulation and read back from "
                     "physics state, not predetermined.</div>")

    # used_fallback is only meaningful after a plan has been attempted.
    if getattr(agent.planner, "used_fallback", False):
        parts.append("<div class='disclosure'><strong>Heuristic planner.</strong> "
                     "Bedrock was unavailable, so this plan came from the keyword "
                     "fallback parser - the LLM was NOT used.</div>")

    return "\n".join(parts)


def _format_subtasks(subtasks: List[Dict]) -> str:
    if not subtasks:
        return TASK_STATUS_HTML.format(content=_disclosure_html() + "<p>No plan yet</p>")

    items = [_disclosure_html()]
    for st in subtasks:
        status = st.get("status", "pending")
        icon = STATUS_ICONS.get(status, "○")
        attempts = st.get("attempts", 0)
        attempt_str = (f"<span class='attempt'> (attempt {attempts})</span>"
                       if attempts and status == "in_progress" else "")
        items.append(
            f"<div class='subtask-item'>"
            f"<span class='status-{status}'>{icon} Task {st.get('id')}: "
            f"{st.get('action')} {st.get('object')} -> {st.get('target')}"
            f"{attempt_str}</span></div>"
        )

    return TASK_STATUS_HTML.format(content="\n".join(items))


def _format_final(subtasks: List[Dict], summary: Dict) -> str:
    items = [_disclosure_html(), "<h4>Final Results:</h4>"]

    for st in subtasks:
        status = st.get("status", "pending")
        icon = STATUS_ICONS.get(status, "○")
        items.append(
            f"<div class='subtask-item'>"
            f"<span class='status-{status}'>{icon} Task {st.get('id')}: "
            f"{st.get('action')} {st.get('object')} -> {st.get('target')}</span></div>"
        )

    bg = "#e8f5e9" if summary.get("failed", 0) == 0 else "#fff3e0"
    stopped = ("<br><em>Run stopped before completion.</em>"
               if summary.get("stopped_early") else "")
    items.append(
        f"<div style='margin-top:16px;padding:12px;background:{bg};border-radius:8px;'>"
        f"<strong>Summary:</strong><br>"
        f"Successful: {summary.get('successful', 0)}/{summary.get('total_subtasks', 0)}<br>"
        f"Failed: {summary.get('failed', 0)}/{summary.get('total_subtasks', 0)}<br>"
        f"Retries: {summary.get('total_retries', 0)}<br>"
        f"Task time: {summary.get('total_time', 0):.2f}s<br>"
        f"Wall time: {summary.get('elapsed_time', 0):.2f}s"
        f"{stopped}</div>"
    )

    return TASK_STATUS_HTML.format(content="\n".join(items))


def run_agent(instruction: str, max_retries: int, max_steps: int) -> Generator:
    """Stream orchestrator events into the four output components."""
    if not instruction.strip():
        yield ("Please enter an instruction", _format_subtasks([]),
               "Waiting for instruction...", None)
        return

    last_video = None

    for event in agent.run(instruction,
                           max_retries=int(max_retries),
                           max_steps=int(max_steps)):
        last_video = event.video_path or last_video

        if event.summary is not None:
            plan_html = _format_final(event.subtasks, event.summary)
        else:
            plan_html = _format_subtasks(event.subtasks)

        yield (event.status, plan_html, event.log_text, last_video)


def stop_agent() -> str:
    agent.request_stop()
    return "Stop requested - finishing current attempt..."


css = """
.gradio-container {max-width: 1200px !important; margin: auto;}
.output-log {font-family: 'Courier New', monospace; white-space: pre-wrap;}
.video-display {min-height: 300px;}
"""


with gr.Blocks(title="Robot Task Agent") as demo:
    gr.Markdown("""
    # 🤖 Language-to-Action Robot Task Agent
    ### InnovaHack Chapter-1 — Domain 4: Agentic AI

    Enter a natural language instruction to control the robot arm.
    Example: *"Put the red block in the bin, then stack blue on green"*
    """)

    with gr.Row():
        with gr.Column(scale=2):
            instruction_input = gr.Textbox(
                label="Instruction",
                placeholder="Put the red block in the bin...",
                lines=2,
                value="put the red block in the bin",
            )

            with gr.Row():
                submit_btn = gr.Button("Execute", variant="primary", scale=2)
                stop_btn = gr.Button("Stop", variant="stop", scale=1)

            status_output = gr.Textbox(label="Status", value="Ready", interactive=False)

        with gr.Column(scale=1):
            max_retries_slider = gr.Slider(
                1, 5, value=MAX_RETRIES_PER_SUBTASK, step=1,
                label="Max Attempts per Subtask")
            max_steps_slider = gr.Slider(
                10, 200, value=MAX_STEPS_PER_ROLLOUT, step=10,
                label="Max Steps per Rollout")

    gr.Markdown("## Execution Progress")

    with gr.Row():
        with gr.Column(scale=1):
            plan_output = gr.HTML(label="Plan", value=_format_subtasks([]))

        with gr.Column(scale=1):
            video_output = gr.Video(label="Robot View", interactive=False, autoplay=True)

    log_output = gr.Textbox(
        label="Trace Log",
        value="Ready. Enter an instruction to begin.",
        lines=10,
        interactive=False,
        elem_classes=["output-log"],
    )

    # The sliders are real inputs now; previously they were declared and never
    # connected, so changing them had no effect on the run.
    submit_btn.click(
        fn=run_agent,
        inputs=[instruction_input, max_retries_slider, max_steps_slider],
        outputs=[status_output, plan_output, log_output, video_output],
    )

    # Stop writes a flag the orchestrator's loop actually checks.
    stop_btn.click(fn=stop_agent, inputs=[], outputs=[status_output])

    gr.Markdown(f"""
    ---
    **Instructions:**
    - Enter a natural language instruction (e.g., "stack blue on green").
    - Click **Execute** to start the agent.
    - The plan will be decomposed and executed step by step.
    - Retries are automatic with diagnostic logs.

    **Available Objects:** red block, blue block, green block, yellow block, bin

    **Planner backend:** AWS Bedrock `{BEDROCK_MODEL_ID}`
    """)


if __name__ == "__main__":
    os.makedirs(VIDEOS_DIR, exist_ok=True)

    # Try a few ports: during a demo the previous run is often still holding
    # 7860, and a traceback at that moment looks far worse than a port bump.
    for port in range(7860, 7870):
        try:
            # Gradio 6 moved theme/css off the Blocks constructor onto launch().
            demo.launch(server_name="0.0.0.0", server_port=port, share=False,
                        css=css, theme=gr.themes.Soft())
            break
        except OSError as exc:
            print(f"[Dashboard] Port {port} busy ({exc.__class__.__name__}); trying {port + 1}")
    else:
        print("[Dashboard] No free port in 7860-7869. "
              "Stop the other dashboard instance and retry.")
