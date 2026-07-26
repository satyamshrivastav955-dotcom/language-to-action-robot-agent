"""CLI front-end for the robot task agent.

The plan/execute/verify/retry loop lives in agent/task_agent.py and is shared
with the Gradio dashboard; this file only parses arguments and prints.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.task_agent import RobotTaskAgent
from configs.settings import (
    MAX_RETRIES_PER_SUBTASK,
    MAX_STEPS_PER_ROLLOUT,
    VIDEOS_DIR,
    LOGS_DIR,
)


def main():
    parser = argparse.ArgumentParser(description="Robot Task Agent")
    parser.add_argument("instruction", nargs="?",
                        default="put the red block in the bin",
                        help="Natural language instruction")
    # Defaults come from settings so the config file is the single source of
    # truth; hardcoding them here made MAX_STEPS_PER_ROLLOUT unreachable.
    parser.add_argument("--retries", type=int, default=MAX_RETRIES_PER_SUBTASK,
                        help=f"Max attempts per subtask (default: {MAX_RETRIES_PER_SUBTASK})")
    parser.add_argument("--steps", type=int, default=MAX_STEPS_PER_ROLLOUT,
                        help=f"Max steps per rollout (default: {MAX_STEPS_PER_ROLLOUT})")
    parser.add_argument("--no-videos", action="store_true",
                        help="Disable video saving")
    parser.add_argument("--vlm-verify", action="store_true",
                        help="Enable VLM verification (Tier B)")
    # Default is the real MuJoCo simulation. --mock opts back in to the
    # scripted Bernoulli policy for a fast, dependency-free run.
    parser.add_argument("--mock", action="store_true", default=False,
                        help="Use the scripted mock policy (teleports blocks, no physics)")
    parser.add_argument("--real-policy", dest="mock", action="store_false",
                        help="Run the MuJoCo physics simulation (default)")

    args = parser.parse_args()

    os.makedirs(VIDEOS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    print("=" * 60)
    print("ROBOT TASK AGENT")
    print("InnovaHack Chapter-1 - Domain 4: Agentic AI")
    print("=" * 60)
    print(f"\nInstruction: {args.instruction}")
    print(f"Max retries: {args.retries}")
    print(f"Max steps:   {args.steps}")
    print(f"Save videos: {not args.no_videos}")
    print(f"VLM verify:  {args.vlm_verify}")
    print(f"Mock policy: {args.mock}")

    with RobotTaskAgent(
        use_mock_policy=args.mock,
        enable_vlm_verifier=args.vlm_verify,
    ) as agent:

        policy_info = agent.executor.describe_policy()
        if policy_info["outcomes_are_scripted"]:
            print("\n" + "!" * 60)
            print("NOTICE: outcomes in this run are SCRIPTED.")
            print(policy_info["note"])
            print("!" * 60)

        print(f"\n{'=' * 60}")
        print(f"Instruction: {args.instruction}")
        print(f"{'=' * 60}\n")

        result = agent.execute_instruction(
            args.instruction,
            max_retries=args.retries,
            max_steps=args.steps,
            save_videos=not args.no_videos,
        )

        if result.get("error"):
            print(f"\nERROR: {result['error']}")
            return result

        summary = result.get("summary", {})

        print(f"\n{'=' * 60}")
        print("EXECUTION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Instruction:  {args.instruction}")
        print(f"Total time:   {summary.get('elapsed_time', 0):.2f}s")
        print(f"Task time:    {summary.get('total_time', 0):.2f}s")
        print(f"Successful:   {summary.get('successful', 0)}/{summary.get('total_subtasks', 0)}")
        print(f"Failed:       {summary.get('failed', 0)}/{summary.get('total_subtasks', 0)}")
        print(f"Total retries:{summary.get('total_retries', 0)}")

        for r in result.get("results", []):
            status = "SUCCESS" if r["success"] else "FAILED"
            print(f"  Subtask {r['subtask_id']}: {status} ({r['attempts']} attempt(s))")

        if summary.get("stopped_early"):
            print("\nRun was stopped before all subtasks completed.")

        print(f"{'=' * 60}\n")

        print("=" * 60)
        print("FINAL RESULT")
        print("=" * 60)
        print("Task completed successfully!" if result["success"]
              else "Task completed with errors.")

        if policy_info["outcomes_are_scripted"]:
            print("(Reminder: outcomes were scripted, not produced by a trained policy.)")

        if summary.get("log_path"):
            print(f"Trace log: {summary['log_path']}")

        return result


if __name__ == "__main__":
    main()
