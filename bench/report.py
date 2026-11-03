"""Turn a results JSON into markdown tables and a compact JSON for the web viewer.

    python -m bench.report results/ablation.json --md results/ablation.md --json web/data/results.json
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List

from bench.stats import mcnemar_exact, wilson

ARM_ORDER = ["no_verifier", "verifier_blind", "verifier_corrected", "verifier_damped", "oracle_verifier_corrected"]
ARM_LABEL = {
    "no_verifier": "No verifier",
    "verifier_blind": "Verifier + blind retry",
    "verifier_corrected": "Verifier + corrected retry",
    "verifier_damped": "Verifier + damped correction",
    "oracle_verifier_corrected": "Oracle verifier + corrected retry",
}
PROFILE_ORDER = ["nominal", "noisy", "biased", "slippery"]
TASK_ORDER = ["pick_place", "stack", "two_step"]


def _cell(eps: List[Dict]) -> Dict:
    n = len(eps)
    k = sum(1 for e in eps if e["oracle_success"])
    lo, hi = wilson(k, n)
    believed = sum(1 for e in eps if e["believed_success"])
    return {"n": n, "k": k, "rate": k / n if n else 0.0, "lo": lo, "hi": hi,
            "mean_attempts": sum(e["attempts"] for e in eps) / n if n else 0.0,
            "false_success": sum(1 for e in eps if e["believed_success"] and not e["oracle_success"]),
            "believed": believed}


def _group(episodes, keys):
    g = defaultdict(list)
    for e in episodes:
        g[tuple(e[k] for k in keys)].append(e)
    return g


def paired(episodes, profile, arm_a, arm_b, task=None) -> Dict:
    by = {}
    for e in episodes:
        if e["profile"] == profile and e["arm"] in (arm_a, arm_b) and (task is None or e["task"] == task):
            by.setdefault((e["task"], e["seed"]), {})[e["arm"]] = e["oracle_success"]
    only_a = sum(1 for v in by.values() if len(v) == 2 and v[arm_a] and not v[arm_b])
    only_b = sum(1 for v in by.values() if len(v) == 2 and v[arm_b] and not v[arm_a])
    n = sum(1 for v in by.values() if len(v) == 2)
    return {"n_pairs": n, "only_a": only_a, "only_b": only_b, "p": mcnemar_exact(only_a, only_b)}


def verifier_confusion(episodes) -> Dict:
    tp = fp = tn = fn = 0
    for e in episodes:
        if e["arm"] not in ("verifier_blind", "verifier_corrected", "verifier_damped"):
            continue
        for row in e["rows"]:
            for a in row.get("attempt_log", []):
                if a.get("oracle_success") is None:
                    continue
                v, o = a["verdict"], a["oracle_success"]
                tp += v and o
                fp += v and not o
                tn += (not v) and (not o)
                fn += (not v) and o
    return {"accept_correct": tp, "false_accept": fp, "reject_correct": tn, "false_reject": fn}


def build(data: Dict) -> Dict:
    eps = [e for e in data["episodes"] if not e.get("error")]
    errors = len(data["episodes"]) - len(eps)
    out = {"meta": data["meta"], "errors": errors, "pooled": {}, "per_task": {}, "paired": {}}
    for (prof, arm), g in _group(eps, ["profile", "arm"]).items():
        out["pooled"].setdefault(prof, {})[arm] = _cell(g)
    for (task, prof, arm), g in _group(eps, ["task", "profile", "arm"]).items():
        out["per_task"].setdefault(task, {}).setdefault(prof, {})[arm] = _cell(g)
    for prof in PROFILE_ORDER:
        out["paired"][prof] = {
            "verifier_blind_vs_none": paired(eps, prof, "verifier_blind", "no_verifier"),
            "corrected_vs_blind": paired(eps, prof, "verifier_corrected", "verifier_blind"),
            "perception_vs_oracle_verifier": paired(eps, prof, "verifier_corrected",
                                                    "oracle_verifier_corrected"),
        }
    out["verifier_confusion"] = verifier_confusion(eps)
    return out


def _pct(c):
    return f"{100 * c['rate']:.0f}% [{100 * c['lo']:.0f}-{100 * c['hi']:.0f}]"


def markdown(res: Dict) -> str:
    n = res["meta"]["episodes"]
    L = [f"_{n} seeds per cell, attempt budget {res['meta']['attempt_budget']}; success = simulator ground "
         f"truth at episode end; brackets are 95% Wilson intervals; "
         f"{res['errors']} episodes errored._", ""]
    L += ["### Success rate by fault profile (all tasks pooled)", "",
          "| Fault profile | " + " | ".join(ARM_LABEL[a] for a in ARM_ORDER) + " |",
          "|---|" + "---|" * len(ARM_ORDER)]
    for prof in PROFILE_ORDER:
        row = res["pooled"].get(prof, {})
        L.append(f"| {prof} (n={row.get('no_verifier', {}).get('n', 0)}) | "
                 + " | ".join(_pct(row[a]) if a in row else "-" for a in ARM_ORDER) + " |")
    L += ["", "### Per task", "",
          "| Task | Fault profile | " + " | ".join(ARM_LABEL[a] for a in ARM_ORDER) + " |",
          "|---|---|" + "---|" * len(ARM_ORDER)]
    for task in TASK_ORDER:
        for prof in PROFILE_ORDER:
            row = res["per_task"].get(task, {}).get(prof, {})
            if row:
                L.append(f"| {task} | {prof} | " + " | ".join(_pct(row[a]) if a in row else "-" for a in ARM_ORDER) + " |")
    L += ["", "### Paired comparisons (same seeds, exact McNemar)", "",
          "| Fault profile | Verifier(blind) vs none | Corrected vs blind retry | Perception vs oracle verifier |",
          "|---|---|---|---|"]

    def fmt(p, a, b):
        return f"{p['only_a']} vs {p['only_b']} (p={p['p']:.3g}, n={p['n_pairs']})"

    for prof in PROFILE_ORDER:
        p = res["paired"][prof]
        L.append(f"| {prof} | {fmt(p['verifier_blind_vs_none'], 'v', 'n')} | "
                 f"{fmt(p['corrected_vs_blind'], 'c', 'b')} | {fmt(p['perception_vs_oracle_verifier'], 'p', 'o')} |")
    L += ["", "Each cell reads `episodes only the first arm solved vs only the second arm solved`.", ""]
    L += ["### Mean attempts per episode and false successes", "",
          "| Fault profile | " + " | ".join(ARM_LABEL[a] for a in ARM_ORDER) + " |",
          "|---|" + "---|" * len(ARM_ORDER)]
    for prof in PROFILE_ORDER:
        row = res["pooled"].get(prof, {})
        L.append(f"| {prof} | " + " | ".join(
            f"{row[a]['mean_attempts']:.2f} att, {row[a]['false_success']} false-ok" if a in row else "-"
            for a in ARM_ORDER) + " |")
    c = res["verifier_confusion"]
    tot = sum(c.values()) or 1
    L += ["", "### Perception verifier vs ground truth (attempt level)", "",
          f"accepted-and-correct {c['accept_correct']}, **false accepts {c['false_accept']}**, "
          f"rejected-and-correct {c['reject_correct']}, **false rejects {c['false_reject']}** "
          f"(of {tot} verdicts).", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--md")
    ap.add_argument("--json")
    args = ap.parse_args()
    with open(args.results, encoding="utf-8") as f:
        res = build(json.load(f))
    md = markdown(res)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(md)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f)
    print(md)


if __name__ == "__main__":
    main()
