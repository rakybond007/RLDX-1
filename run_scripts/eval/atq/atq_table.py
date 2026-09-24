#!/usr/bin/env python3
"""Combined progress + results table for the two ATQ eval trees.

Scrapes the live rollout logs, so it is meaningful while the jobs are still
running: tasks with no success rate yet are reported as pending rather than
silently dropped, which is what made the earlier summaries look complete when
they were not.
"""
import argparse
import ast
from pathlib import Path

from summarize_eval import SHARES, SR

ROOT = Path(__file__).resolve().parents[3]
TREES = {
    "LIBERO": (
        ROOT / "output_final/libero"
        / "rldx1_img_libero_atq_labelgated_167_confboth_legacy_initrand_disc6_gb32_60k",
        40,
    ),
    "RoboCasa": (
        ROOT / "output_final/robocasa"
        / "rldx1_img_robocasa_atq_labelgated_20_vlm_contact_legacy_initrand_sharded_gb64_60k_h100",
        24,
    ),
}


def scrape(root):
    rows, experts, chunks, hor, conf = [], {}, 0, 0.0, 0.0
    for log in sorted(root.rglob("*.log")):
        if log.parent.name == "_launcher_logs":
            continue
        text = log.read_text(errors="ignore")
        task = log.parent.relative_to(root).as_posix() or log.stem
        hits = SR.findall(text)
        rows.append((task, float(hits[-1]) if hits else None))
        for raw, n, h, c in SHARES.findall(text):
            n = int(n)
            chunks += n
            hor += float(h) * n
            conf += float(c) * n
            for name, frac in ast.literal_eval(raw).items():
                experts[name] = experts.get(name, 0.0) + float(frac) * n
    return rows, experts, chunks, hor, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="", help="also write the markdown here")
    a = ap.parse_args()
    lines = []
    for name, (root, total) in TREES.items():
        if not root.is_dir():
            lines.append(f"\n## {name}\n\n_no output yet_")
            continue
        rows, experts, chunks, hor, conf = scrape(root)
        done = [(t, sr) for t, sr in rows if sr is not None]
        lines.append(f"\n## {name} — {len(done)}/{total} tasks scored")
        suites = {}
        for t, sr in done:
            suites.setdefault(t.split("/")[0] if "/" in t else name, []).append(sr)
        if suites:
            lines.append("\n| suite | tasks | mean success |\n| --- | ---: | ---: |")
            for k, v in sorted(suites.items()):
                lines.append(f"| {k} | {len(v)} | {sum(v) / len(v):.3f} |")
            allv = [sr for _, sr in done]
            lines.append(f"| **ALL** | **{len(allv)}** | **{sum(allv) / len(allv):.3f}** |")
        if chunks:
            sh = ", ".join(f"`{k}` {v / chunks:.3f}" for k, v in sorted(experts.items()))
            lines.append(
                f"\nATQ router over {chunks} chunks: {sh} — "
                f"mean horizon {hor / chunks:.2f}, mean conf {conf / chunks:.3f}"
            )
        pending = [t for t, sr in rows if sr is None]
        if pending:
            lines.append(f"\n_pending ({len(pending)}): " + ", ".join(sorted(pending)[:6]) + "_")
    md = "\n".join(lines).strip() + "\n"
    print(md)
    if a.out:
        Path(a.out).write_text(md)


if __name__ == "__main__":
    main()
