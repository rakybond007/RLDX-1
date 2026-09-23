#!/usr/bin/env python3
"""Collect per-task success rates from an eval output tree into one table.

The rollout writes a log per task; this scrapes the success rate out of each and
prints a sorted summary plus the mean. Also reports which ATQ expert the router
picked, when the log carries it, because a label-gated run whose router never
leaves ``main`` scores like the baseline and is otherwise indistinguishable.
"""
import argparse
import json
import re
from pathlib import Path

SR = re.compile(r"success[ _]rate[^0-9]*([0-9]*\.?[0-9]+)")
EXPERT = re.compile(r"atq_expert['\"]?[:=]\s*['\"]?(main|m8|m4|n8)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="eval output tree")
    ap.add_argument("--benchmark", default="", help="robocasa | libero (labels only)")
    a = ap.parse_args()
    root = Path(a.root)
    if not root.is_dir():
        print(f"[summary] no eval output at {root}")
        return

    rows, experts = [], {}
    for log in sorted(root.rglob("*.log")):
        text = log.read_text(errors="ignore")
        hits = SR.findall(text)
        if not hits:
            continue
        task = log.parent.relative_to(root).as_posix() or log.stem
        rows.append((task, float(hits[-1])))
        for e in EXPERT.findall(text):
            experts[e] = experts.get(e, 0) + 1

    if not rows:
        print(f"[summary] {root}: no success_rate found in any log")
        return
    width = max(len(t) for t, _ in rows)
    print(f"\n===== {a.benchmark or root.name} success rates ({len(rows)} tasks) =====")
    for task, sr in sorted(rows, key=lambda r: -r[1]):
        print(f"{task:<{width}}  {sr:.3f}")
    mean = sum(sr for _, sr in rows) / len(rows)
    print(f"{'MEAN':<{width}}  {mean:.3f}")
    if experts:
        total = sum(experts.values())
        share = {k: round(v / total, 3) for k, v in sorted(experts.items())}
        print(f"router picks: {share}")
    out = root / "summary.json"
    out.write_text(json.dumps({"tasks": dict(rows), "mean": mean, "router_picks": experts}, indent=2))
    print(f"[summary] wrote {out}")


if __name__ == "__main__":
    main()
