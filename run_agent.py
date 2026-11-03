"""CLI front-end for the robot task agent.

The plan/execute/verify/retry loop lives in agent/task_agent.py and is shared
with the Gradio dashboard; this file only parses arguments and prints.
"""

import os
import argparse


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
                        help="Enable the (inert) Bedrock VLM second-opinion; it never decides pass/fail")
    parser.add_argument("--verifier", choices=["perception", "oracle", "none"], default="perception",
                        help="Who judges the outcome: the camera (default), simulator truth, or nobody")
    parser.add_argument("--retry", choices=["corrected", "blind"], default="corrected",
                        help="Retry strategy: feed back the measured error, or just repeat")
    parser.add_argument("--faults", default="nominal",
                        choices=["nominal", "noisy", "biased", "slippery"],
                        help="Inject execution faults (see env/faults.py)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for sensor noise and injected faults")
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
    print("LANGUAGE-TO-ACTION ROBOT AGENT")
    print("Plan, act, verify from pixels, retry from the measured error")
    print("=" * 60)
    print(f"\nInstruction: {args.instruction}")
    print(f"Max retries: {args.retries}")
    print(f"Max steps:   {args.steps}")
    print(f"Save videos: {not args.no_videos}")
    print(f"Verifier:    {args.verifier}   Retry: {args.retry}   Faults: {args.faults}   Seed: {args.seed}")
    print(f"Mock policy: {args.mock}")

    with RobotTaskAgent(
        use_mock_policy=args.mock,
        enable_vlm_verifier=args.vlm_verify,
        verifier_mode=args.verifier,
        retry_strategy=args.retry,
        fault_profile=args.faults,
        seed=args.seed,
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

        instruction = args.instruction
        result = agent.execute_instruction(
            instruction,
            max_retries=args.retries,
            max_steps=args.steps,
            save_videos=not args.no_videos,
        )

        # The planner asks rather than guessing when an instruction is
        # genuinely ambiguous. Answer it and re-plan once; a second question
        # would mean the answer did not help, so we stop there.
        if result.get("clarification"):
            print("\n" + "=" * 60)
            print("CLARIFICATION NEEDED")
            print("=" * 60)
            print(result["clarification"])
            try:
                answer = input("\nYour answer: ").strip()
            except EOFError:
                answer = ""
            if not answer:
                print("No answer given; stopping.")
                return result
            instruction = f"{instruction} ({answer})"
            print(f"\nRe-planning with: {instruction}\n")
            result = agent.execute_instruction(
                instruction,
                max_retries=args.retries,
                max_steps=args.steps,
                save_videos=not args.no_videos,
            )
            if result.get("clarification"):
                print(f"\nStill ambiguous: {result['clarification']}")
                return result

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


def cli():
    """Console-script entry point; exit code reflects task success."""
    result = main()
    raise SystemExit(0 if result and result.get("success") else 1)


if __name__ == "__main__":
    main()
