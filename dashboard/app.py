"""Gradio front-end: type an instruction, watch the closed loop work.

    python -m dashboard.app          (needs `pip install -e ".[dashboard]"`)

One robot, one run at a time: a second submit gets a "busy" message instead of
corrupting the simulation. The agent is built per run, so changing a setting
(fault profile, verifier, retry strategy) takes effect on the next run.
"""

import html
import os
import threading
from typing import Dict, List, Optional

import gradio as gr

from agent.planner import create_planner
from agent.task_agent import RobotTaskAgent
from configs.settings import MAX_RETRIES_PER_SUBTASK, VIDEOS_DIR
from env.faults import PROFILES
from env.so101_env import create_environment

_LOCK = threading.Lock()
_STATE: Dict[str, object] = {"env": None, "planner": None, "agent": None}

FAULT_CHOICES = [(f"{p.name} - {p.description}", p.name) for p in PROFILES.values()]
VERIFIER_CHOICES = [("Camera (pixels only)", "perception"), ("Ground truth (oracle)", "oracle"),
                    ("None (trust the rollout)", "none")]
RETRY_CHOICES = [("Corrected: use the measured error", "corrected"), ("Blind: repeat", "blind")]
EXAMPLES = [
    "put the red block in the bin",
    "stack the blue block on the green block",
    "put the green block in the bin, then stack red on yellow",
    "put the block in the bin",
]

CSS = """
.gradio-container { max-width: 1240px !important; margin: auto; }
.rl-head h1 { font-size: 1.45rem; margin: 0 0 4px; letter-spacing: -0.01em; }
.rl-head p { margin: 0; opacity: .72; }
.rl-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
.rl-chip { font-size: .76rem; font-weight: 600; padding: 3px 10px; border-radius: 99px;
           border: 1px solid var(--border-color-primary); background: var(--block-background-fill); }
.rl-card { border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 12px 14px;
           margin-bottom: 10px; background: var(--block-background-fill); }
.rl-card .t { display: flex; justify-content: space-between; gap: 10px; font-weight: 650; }
.rl-badge { font-size: .74rem; font-weight: 700; padding: 2px 9px; border-radius: 99px; border: 1px solid; }
.rl-badge.pending { opacity: .55; } .rl-badge.in_progress { color: var(--color-accent); }
.rl-badge.completed { color: #2fa36b; } .rl-badge.failed { color: #d9534f; }
.rl-atts { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 9px; }
.rl-att { font-size: .8rem; padding: 3px 9px; border-radius: 8px; border: 1px solid var(--border-color-primary); }
.rl-att b.ok { color: #2fa36b; } .rl-att b.bad { color: #d9534f; }
.rl-att small { opacity: .65; }
.rl-empty { opacity: .6; padding: 18px; text-align: center; }
.rl-ask { border-left: 3px solid #e9b44c; padding: 10px 14px; border-radius: 6px;
          background: color-mix(in srgb, #e9b44c 12%, transparent); }
.rl-stats { display: flex; gap: 18px; flex-wrap: wrap; font-size: .9rem; opacity: .85; }
.rl-stats b { font-variant-numeric: tabular-nums; }
"""


def _esc(x) -> str:
    return html.escape(str(x))


def header_html(planner_name: str, fault: str, verifier: str, retry: str) -> str:
    chips = [f"Planner: {planner_name}", "Perception: RGB-D segmentation", f"Verifier: {verifier}",
             f"Retry: {retry}", f"Faults: {fault}"]
    return ("<div class='rl-head'><h1>Language-to-Action Robot Agent</h1>"
            "<p>Plan with a local LLM, act, check the result from camera pixels, retry from the measured error.</p>"
            "<div class='rl-chips'>" + "".join(f"<span class='rl-chip'>{_esc(c)}</span>" for c in chips)
            + "</div></div>")


def timeline_html(subtasks: List[Dict]) -> str:
    if not subtasks:
        return "<div class='rl-empty'>No plan yet. Enter an instruction and press Run.</div>"
    cards = []
    for st in subtasks:
        status = st.get("status", "pending")
        label = {"pending": "Queued", "in_progress": "Running", "completed": "Done", "failed": "Failed"}[status]
        target = "bin" if st.get("target") == "bin" else f"on {str(st.get('target', '')).split('_')[0]}"
        atts = []
        for a in st.get("attempt_log", []):
            if a.get("observed_by") == "none":
                v = "<b>·</b> not checked"
            else:
                mark = "✓" if a["verdict"] else "✗"
                v = f"<b class='{'ok' if a['verdict'] else 'bad'}'>{mark}</b> {_esc(a['observed_by'])}"
            truth = a.get("oracle_success")
            t = "" if truth is None else f" <small>truth {'✓' if truth else '✗'}</small>"
            atts.append(f"<span class='rl-att'>Attempt {a['attempt']}: {v}{t}</span>")
        cards.append(
            f"<div class='rl-card'><div class='t'><span>{st.get('id', '')}. "
            f"{_esc(str(st.get('action', '')).replace('_', ' '))} "
            f"{_esc(str(st.get('object', '')).split('_')[0])} {_esc(target)}</span>"
            f"<span class='rl-badge {status}'>{label}</span></div>"
            f"<div class='rl-atts'>{''.join(atts)}</div></div>")
    return "".join(cards)


def stats_html(summary: Optional[Dict]) -> str:
    if not summary:
        return ""
    return ("<div class='rl-stats'>"
            f"<span>Subtasks <b>{summary.get('successful', 0)}/{summary.get('total_subtasks', 0)}</b> succeeded</span>"
            f"<span>Retries <b>{summary.get('total_retries', 0)}</b></span>"
            f"<span>Elapsed <b>{summary.get('elapsed_time', 0):.1f}s</b></span></div>")


def _resources():
    if _STATE["env"] is None:
        _STATE["env"] = create_environment()
        _STATE["planner"] = create_planner()
    return _STATE["env"], _STATE["planner"]


def run_ui(instruction, fault, verifier, retry, budget, answer):
    """Generator feeding: header, timeline, stats, video, camera image, log, ask group, ask text, clips."""
    no_ask = (gr.update(visible=False), "")
    if not (instruction or "").strip():
        yield (gr.skip(), timeline_html([]), "", None, None, "Enter an instruction first.", *no_ask,
               gr.update(choices=[], value=None))
        return
    if not _LOCK.acquire(blocking=False):
        yield (gr.skip(), gr.skip(), "", gr.skip(), gr.skip(),
               "The robot is busy with another run. Try again in a moment.", *no_ask, gr.skip())
        return
    try:
        env, planner = _resources()
        text = f"{instruction.strip()} ({answer.strip()})" if answer and answer.strip() else instruction.strip()
        agent = RobotTaskAgent(env=env, planner=planner, console_output=False, fault_profile=fault,
                               verifier_mode=verifier, retry_strategy=retry)
        _STATE["agent"] = agent
        os.makedirs(VIDEOS_DIR, exist_ok=True)
        head = header_html(getattr(planner, "backend_name", "?"), fault, verifier, retry)
        clips: List[tuple] = []
        last_video = last_img = None
        summary = None
        for ev in agent.run(text, max_retries=int(budget)):
            if ev.video_path and ev.video_path not in [c[1] for c in clips]:
                clips.append((f"Clip {len(clips) + 1}", ev.video_path))
                last_video = ev.video_path
            if ev.camera_image is not None:
                last_img = ev.camera_image
            summary = ev.summary or summary
            if ev.clarification:
                ask = (gr.update(visible=True),
                       f"<div class='rl-ask'><b>The planner needs one detail.</b><br>{_esc(ev.clarification)}</div>")
                yield (head, timeline_html(ev.subtasks), "", last_video, last_img, ev.log_text, *ask,
                       gr.update(choices=clips, value=last_video))
                return
            yield (head, timeline_html(ev.subtasks), stats_html(summary), last_video, last_img, ev.log_text,
                   *no_ask, gr.update(choices=clips, value=last_video))
        if summary is None:
            yield (head, gr.skip(), "", last_video, last_img, "Run ended without a result.", *no_ask, gr.skip())
    finally:
        _STATE["agent"] = None
        _LOCK.release()


def stop_run():
    agent = _STATE.get("agent")
    if agent is not None:
        agent.request_stop()
        return "Stop requested; the run ends after the current attempt."
    return "Nothing is running."


def build() -> gr.Blocks:
    with gr.Blocks(title="Robot Agent") as demo:
        head = gr.HTML(header_html("local LLM (Ollama)", "nominal", "perception", "corrected"))
        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=320):
                instruction = gr.Textbox(label="Instruction", lines=2, placeholder="put the red block in the bin")
                gr.Examples(EXAMPLES, instruction, label="Try")
                with gr.Row():
                    run_btn = gr.Button("Run", variant="primary", scale=3)
                    stop_btn = gr.Button("Stop", variant="secondary", scale=1)
                with gr.Accordion("Robot settings", open=False):
                    fault = gr.Dropdown(FAULT_CHOICES, value="nominal", label="Fault profile",
                                        info="Faults are injected into the arm, not the camera.")
                    verifier = gr.Radio(VERIFIER_CHOICES, value="perception", label="Verifier")
                    retry = gr.Radio(RETRY_CHOICES, value="corrected", label="Retry")
                    budget = gr.Slider(1, 5, value=MAX_RETRIES_PER_SUBTASK, step=1, label="Attempts per subtask")
                with gr.Group(visible=False) as ask_group:
                    ask_text = gr.HTML()
                    answer = gr.Textbox(label="Your answer", placeholder="the red one")
                    answer_btn = gr.Button("Answer and run", variant="primary")
                stats = gr.HTML()
                timeline = gr.HTML(timeline_html([]))
            with gr.Column(scale=6, min_width=420):
                video = gr.Video(label="Rollout", interactive=False, autoplay=True, height=360)
                clips = gr.Dropdown(label="Clips", choices=[], interactive=True)
                camera = gr.Image(label="What the verifier saw", interactive=False, height=300)
        with gr.Accordion("Run log", open=False):
            log = gr.Textbox(show_label=False, lines=10, max_lines=18, interactive=False, autoscroll=True)

        outputs = [head, timeline, stats, video, camera, log, ask_group, ask_text, clips]
        no_answer = gr.State("")
        run_btn.click(run_ui, [instruction, fault, verifier, retry, budget, no_answer], outputs)
        instruction.submit(run_ui, [instruction, fault, verifier, retry, budget, no_answer], outputs)
        answer_btn.click(run_ui, [instruction, fault, verifier, retry, budget, answer], outputs)
        stop_btn.click(stop_run, None, log)
        clips.change(lambda p: p, clips, video)
    return demo


demo = build()

if __name__ == "__main__":
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    demo.queue().launch(server_name="127.0.0.1", css=CSS, theme=gr.themes.Soft())
