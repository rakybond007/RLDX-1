#!/usr/bin/env python3
"""Collect per-task success rates from an eval output tree into one table.

The rollout writes a log per task; this scrapes the success rate out of each and
prints a sorted summary plus the mean. Also reports which ATQ expert the router
picked, when the log carries it, because a label-gated run whose router never
leaves ``main`` scores like the baseline and is otherwise indistinguishable.
"""
import argparse
import json
import ast
import re
from pathlib import Path

SR = re.compile(r"success[ _]rate[^0-9]*([0-9]*\.?[0-9]+)")
SHARES = re.compile(
    r"atq expert shares: (\{[^}]*\}) over (\d+) chunks, "
    r"mean horizon ([0-9.]+), mean conf ([0-9.]+)"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="eval output tree")
    ap.add_argument("--benchmark", default="", help="robocasa | libero (labels only)")
    ap.add_argument("--markdown", default="", help="also write a markdown table here")
    a = ap.parse_args()
    root = Path(a.root)
    if not root.is_dir():
        print(f"[summary] no eval output at {root}")
        return

    rows, experts, chunks, horizon_w, conf_w = [], {}, 0, 0.0, 0.0
    for log in sorted(root.rglob("*.log")):
        text = log.read_text(errors="ignore")
        hits = SR.findall(text)
        if not hits:
            continue
        task = log.parent.relative_to(root).as_posix() or log.stem
        rows.append((task, float(hits[-1])))
        # Chunk-weighted so a long task does not count the same as a short one.
        for raw, n, hor, conf in SHARES.findall(text):
            n = int(n)
            chunks += n
            horizon_w += float(hor) * n
            conf_w += float(conf) * n
            for name, frac in ast.literal_eval(raw).items():
                experts[name] = experts.get(name, 0.0) + float(frac) * n

    if not rows:
        print(f"[summary] {root}: no success_rate found in any log")
        return
    width = max(len(t) for t, _ in rows)
    print(f"\n===== {a.benchmark or root.name} success rates ({len(rows)} tasks) =====")
    for task, sr in sorted(rows, key=lambda r: -r[1]):
        print(f"{task:<{width}}  {sr:.3f}")
    mean = sum(sr for _, sr in rows) / len(rows)
    print(f"{'MEAN':<{width}}  {mean:.3f}")

    # Suite means: LIBERO logs live under <suite>/<task>, RoboCasa is flat.
    suites = {}
    for task, sr in rows:
        suite = task.split("/")[0] if "/" in task else a.benchmark or root.name
        suites.setdefault(suite, []).append(sr)
    suite_means = {k: sum(v) / len(v) for k, v in sorted(suites.items())}
    if len(suite_means) > 1:
        print("\n----- per-suite mean -----")
        for k, v in suite_means.items():
            print(f"{k:<{width}}  {v:.3f}  (n={len(suites[k])})")

    share = {}
    if chunks:
        share = {k: round(v / chunks, 3) for k, v in sorted(experts.items())}
        print(
            f"\natq expert shares: {share} over {chunks} chunks, "
            f"mean horizon {horizon_w / chunks:.2f}, mean conf {conf_w / chunks:.3f}"
        )
    else:
        print("\n[warn] no 'atq expert shares' line found - router usage unknown")

    out = root / "summary.json"
    out.write_text(
        json.dumps(
            {
                "tasks": dict(rows),
                "mean": mean,
                "suite_means": suite_means,
                "n_tasks": len(rows),
                "expert_shares": share,
                "chunks": chunks,
                "mean_horizon": round(horizon_w / chunks, 3) if chunks else None,
                "mean_conf": round(conf_w / chunks, 3) if chunks else None,
            },
            indent=2,
        )
    )
    print(f"[summary] wrote {out}")

    if a.markdown:
        md = [f"| task | success rate |", "| --- | --- |"]
        md += [f"| `{t}` | {sr:.3f} |" for t, sr in sorted(rows)]
        md.append(f"| **MEAN ({len(rows)} tasks)** | **{mean:.3f}** |")
        Path(a.markdown).write_text("\n".join(md) + "\n")
        print(f"[summary] wrote {a.markdown}")


if __name__ == "__main__":
    main()
