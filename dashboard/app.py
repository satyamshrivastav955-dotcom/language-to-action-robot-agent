"""Gradio front-end for the Language-to-Action Robot Agent.
"""

import os
import sys
from typing import Dict, Generator, List

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.task_agent import RobotTaskAgent
from agent.planner import create_planner
from configs.settings import (
    MAX_RETRIES_PER_SUBTASK,
    MAX_STEPS_PER_ROLLOUT,
    VIDEOS_DIR,
)

# Initialize agent
agent = RobotTaskAgent(use_mock_policy=False, console_output=False)

# Get backend info
backend_name = getattr(agent.planner, 'backend_name', 'OLLAMA/QWEN2.5:14B')
policy_info = agent.executor.describe_policy()
policy_name = "Mock Policy" if policy_info.get("outcomes_are_scripted") else "MuJoCo Physics"
model_id = getattr(agent.planner, 'model', getattr(agent.planner, 'model_id', 'qwen2.5:14b'))

# Custom CSS for the PRO MAX UI
css = """
body, .gradio-container {
    background-color: #0a0a0f !important;
    color: #e0e0e0 !important;
    font-family: 'Inter', sans-serif !important;
}
.gradio-container {
    max-width: 1400px !important;
}
.panel, .gr-box, .gr-panel, .gr-block {
    background-color: #111118 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 4px !important;
}
button {
    border-radius: 4px !important;
    text-transform: uppercase !important;
    font-family: 'Courier New', monospace !important;
    font-weight: bold !important;
}
.btn-execute {
    background-color: #00d4ff !important;
    color: #0a0a0f !important;
    border: none !important;
}
.btn-abort {
    background-color: transparent !important;
    color: #ff4444 !important;
    border: 1px solid #ff4444 !important;
}
.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 15px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 20px;
}
.header-title {
    font-family: 'Courier New', monospace;
    font-size: 1.5em;
    font-weight: bold;
    color: #fff;
    margin: 0;
}
.header-subtitle {
    font-family: 'Courier New', monospace;
    font-size: 0.8em;
    color: rgba(255, 255, 255, 0.5);
    margin: 0;
}
.live-dot {
    width: 10px;
    height: 10px;
    background-color: #00ff88;
    border-radius: 50%;
    display: inline-block;
    animation: pulse 1.5s infinite;
    margin-right: 10px;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(0, 255, 136, 0.7); }
    70% { box-shadow: 0 0 0 10px rgba(0, 255, 136, 0); }
    100% { box-shadow: 0 0 0 0 rgba(0, 255, 136, 0); }
}
.status-row {
    display: flex;
    gap: 15px;
    font-family: 'Courier New', monospace;
    font-size: 0.8em;
    color: #00d4ff;
    padding: 10px;
    background: rgba(0, 212, 255, 0.05);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 4px;
    margin-top: 10px;
}
.task-row {
    display: flex;
    align-items: center;
    padding: 10px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 8px;
    background: #111118;
    border-radius: 4px;
    font-family: 'Courier New', monospace;
}
.task-badge {
    padding: 2px 6px;
    background: rgba(255,255,255,0.1);
    border-radius: 2px;
    margin-right: 15px;
    font-size: 0.8em;
}
.task-action {
    font-weight: bold;
    color: #00d4ff;
    min-width: 150px;
    display: inline-block;
    letter-spacing: 0.5px;
}
.task-obj {
    flex-grow: 1;
    color: rgba(255, 255, 255, 0.85);
    margin: 0 10px;
}
.task-status {
    min-width: 140px;
    text-align: right;
    font-weight: bold;
}
.status-pending { color: rgba(255, 255, 255, 0.3); }
.status-in_progress { color: #00d4ff; animation: text-pulse 1.5s infinite; }
.status-completed { color: #00ff88; }
.status-failed { color: #ff4444; }
@keyframes text-pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}
.progress-bar-container {
    height: 2px;
    background: rgba(255, 255, 255, 0.1);
    width: 100%;
    margin-top: 5px;
}
.progress-bar {
    height: 100%;
    background: #00d4ff;
    width: 0%;
    transition: width 0.3s;
}
.clarification-banner {
    background: rgba(255, 184, 0, 0.1);
    border-left: 4px solid #ffb800;
    padding: 15px;
    margin-bottom: 15px;
    font-family: 'Inter', sans-serif;
}
.log-container textarea {
    font-family: 'Courier New', monospace !important;
    background-color: #050508 !important;
    color: #00d4ff !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
}
.footer-stats {
    display: flex;
    justify-content: space-between;
    padding: 10px 15px;
    font-family: 'Courier New', monospace;
    font-size: 0.85em;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    color: rgba(255, 255, 255, 0.5);
    background-color: #111118;
    margin-top: 20px;
}
.stat-value {
    color: #fff;
    font-weight: bold;
}
.stat-success { color: #00ff88; }
.stat-failed { color: #ff4444; }
"""

HEADER_HTML = """
<div class="header">
    <div>
        <h1 class="header-title">NEXUS-1 // LANGUAGE-TO-ACTION ROBOT</h1>
        <h2 class="header-subtitle">SO-101 6-DOF ARM // MUJOCO PHYSICS // OFFLINE LLM</h2>
    </div>
    <div style="font-family: monospace; color: #00ff88; display: flex; align-items: center;">
        <span class="live-dot"></span> SYSTEM ONLINE
    </div>
</div>
"""

def generate_status_row():
    return f"""
    <div class="status-row">
        <span>BACKEND: {backend_name.upper()}</span>
        <span>POLICY: {policy_name.upper()}</span>
        <span>MODEL: {model_id}</span>
    </div>
    """

def format_subtasks(subtasks: List[Dict]) -> str:
    if not subtasks:
        return "<div style='color: rgba(255,255,255,0.3); font-family: monospace; padding: 20px;'>NO TASKS QUEUED</div>"
    
    html = []
    for st in subtasks:
        status = st.get("status", "pending")
        attempts = st.get("attempts", 0)
        
        status_text = "○ QUEUED"
        if status == "in_progress": status_text = "◉ EXECUTING"
        elif status == "completed": status_text = "✓ DONE"
        elif status == "failed": status_text = "✗ FAILED"
        
        progress = 0
        if status == "completed": progress = 100
        elif status == "in_progress": progress = 50
        
        attempt_str = f" [A{attempts}]" if attempts > 0 else ""
        
        html.append(f"""
        <div class="task-row">
            <div class="task-badge">T-{st.get('id', 'X'):02d}</div>
            <div class="task-action">{str(st.get('action', '')).upper()}</div>
            <div class="task-obj">{st.get('object', '')} &rarr; {st.get('target', '')}</div>
            <div class="task-status status-{status}">{status_text}{attempt_str}</div>
        </div>
        <div class="progress-bar-container">
            <div class="progress-bar" style="width: {progress}%;"></div>
        </div>
        """)
    return "".join(html)

def format_stats(summary: Dict) -> str:
    if not summary:
        return """
        <div class="footer-stats">
            <span>TOTAL TASKS: <span class="stat-value">0</span></span>
            <span>SUCCESS: <span class="stat-value stat-success">0</span></span>
            <span>FAILED: <span class="stat-value stat-failed">0</span></span>
            <span>RETRIES: <span class="stat-value">0</span></span>
            <span>ELAPSED TIME: <span class="stat-value">0.00s</span></span>
        </div>
        """
    return f"""
    <div class="footer-stats">
        <span>TOTAL TASKS: <span class="stat-value">{summary.get('total_subtasks', 0)}</span></span>
        <span>SUCCESS: <span class="stat-value stat-success">{summary.get('successful', 0)}</span></span>
        <span>FAILED: <span class="stat-value stat-failed">{summary.get('failed', 0)}</span></span>
        <span>RETRIES: <span class="stat-value">{summary.get('total_retries', 0)}</span></span>
        <span>ELAPSED TIME: <span class="stat-value">{summary.get('elapsed_time', 0):.2f}s</span></span>
    </div>
    """

def run_agent_ui(instruction: str, max_retries: int, max_steps: int, current_clarification: dict) -> Generator:
    if current_clarification and current_clarification.get('needed'):
        answer = current_clarification.get('answer', '')
        if answer:
            instruction = f"{instruction} (Clarification: {answer})"
    
    clarification_state = {"needed": False, "question": "", "answer": ""}
    clarification_html = ""
    last_video = None
    
    if not instruction.strip():
        yield (format_subtasks([]), "[SYSTEM] Ready. Enter a command sequence.", clarification_html, clarification_state, None, format_stats({}))
        return

    summary = {}
    for event in agent.run(instruction, max_retries=int(max_retries), max_steps=int(max_steps)):
        if event.video_path:
            last_video = event.video_path
            
        if event.clarification:
            clarification_state["needed"] = True
            clarification_state["question"] = event.clarification
            clarification_html = f"""
            <div class="clarification-banner">
                <strong style="color: #ffb800;">CLARIFICATION REQUIRED</strong><br>
                {event.clarification}
            </div>
            """
            yield (format_subtasks(event.subtasks), event.log_text, clarification_html, clarification_state, last_video, format_stats(summary))
            return
            
        summary = event.summary or summary
        yield (format_subtasks(event.subtasks), event.log_text, "", clarification_state, last_video, format_stats(summary))

def stop_agent() -> str:
    agent.request_stop()
    return "Stop requested..."

with gr.Blocks(title="NEXUS-1 CONTROL") as demo:
    gr.HTML(HEADER_HTML)
    
    clarification_state = gr.State({"needed": False, "question": "", "answer": ""})
    
    with gr.Row():
        with gr.Column(scale=1):
            instruction_input = gr.Textbox(
                show_label=False,
                placeholder="Enter command sequence...",
                lines=2,
                elem_classes=["input-panel"],
                container=False
            )
            
            clarification_html = gr.HTML(value="")
            clarification_answer = gr.Textbox(
                show_label=False,
                placeholder="Enter clarification...",
                visible=False,
                container=False
            )
            clarification_submit = gr.Button("SUBMIT ANSWER", visible=False, elem_classes=["btn-execute"])
            
            with gr.Row():
                submit_btn = gr.Button("EXECUTE", elem_classes=["btn-execute"])
                stop_btn = gr.Button("ABORT", elem_classes=["btn-abort"])
            
            max_retries_slider = gr.Slider(1, 5, value=MAX_RETRIES_PER_SUBTASK, step=1, label="Max Retries")
            max_steps_slider = gr.Slider(10, 200, value=MAX_STEPS_PER_ROLLOUT, step=10, label="Max Steps")
            
            gr.HTML(generate_status_row())
            
        with gr.Column(scale=2):
            timeline_output = gr.HTML(label="Mission Timeline", value=format_subtasks([]))
            
            video_output = gr.Video(
                label="PRIMARY ROBOT TELEMETRY FEED (MUJOCO 3D PHYSICS)",
                show_label=True,
                elem_id="robot-video",
                interactive=False,
                autoplay=True,
                height=340
            )

    log_output = gr.Textbox(
        show_label=False,
        value="[SYSTEM] Ready.",
        lines=12,
        interactive=False,
        elem_classes=["log-container"],
        max_lines=20,
        autoscroll=True
    )
    
    stats_output = gr.HTML(value=format_stats({}))
    
    def on_clarification_change(state):
        if state and state.get("needed"):
            return gr.update(visible=True), gr.update(visible=True)
        return gr.update(visible=False), gr.update(visible=False)
        
    clarification_state.change(
        fn=on_clarification_change,
        inputs=[clarification_state],
        outputs=[clarification_answer, clarification_submit]
    )
    
    def handle_clarification_submit(answer, state):
        if state:
            state["answer"] = answer
        return state

    clarification_submit.click(
        fn=handle_clarification_submit,
        inputs=[clarification_answer, clarification_state],
        outputs=[clarification_state]
    ).then(
        fn=run_agent_ui,
        inputs=[instruction_input, max_retries_slider, max_steps_slider, clarification_state],
        outputs=[timeline_output, log_output, clarification_html, clarification_state, video_output, stats_output]
    )

    submit_btn.click(
        fn=run_agent_ui,
        inputs=[instruction_input, max_retries_slider, max_steps_slider, clarification_state],
        outputs=[timeline_output, log_output, clarification_html, clarification_state, video_output, stats_output]
    )

    stop_btn.click(fn=stop_agent, inputs=[], outputs=[])

if __name__ == "__main__":
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    for port in range(7860, 7870):
        try:
            demo.launch(server_name="0.0.0.0", server_port=port, share=False, css=css, theme=gr.themes.Base())
            break
        except OSError as exc:
            print(f"[Dashboard] Port {port} busy ({exc.__class__.__name__}); trying {port + 1}")
    else:
        print("[Dashboard] No free port.")
