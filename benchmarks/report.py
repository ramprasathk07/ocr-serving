"""``results/*.json`` -> markdown tables, patched into ``docs/comparison.md``.

    python benchmarks/report.py

This is the only path from measurements to prose: no number in the README, the
docs or the blog is hand-typed. Re-run it after every benchmark; if a stack has
no result file it simply does not appear in the table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path("benchmarks/results")
TARGET = Path("docs/comparison.md")
BEGIN, END = "<!-- BENCH:BEGIN -->", "<!-- BENCH:END -->"

#: Display order; anything else found on disk is appended alphabetically.
STACKS = ["vllm", "triton", "ray_serve", "kserve", "bentoml", "mock"]
PRETTY = {
    "vllm": "vLLM (standalone)",
    "triton": "Triton (vLLM backend)",
    "ray_serve": "Ray Serve (serve.llm)",
    "kserve": "KServe (HF runtime)",
    "bentoml": "BentoML",
    "mock": "mock engine (CPU)",
}


def load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def discovered_stacks() -> list[str]:
    found = {
        p.stem for p in RESULTS.glob("*.json")
        if not p.stem.startswith(("coldstart_", "ops_"))
    }
    found.discard("ray_batch")
    return [s for s in STACKS if s in found] + sorted(found - set(STACKS))


def cell(value) -> str:
    """Missing measurements render as an em dash, never as the string 'None'."""
    return "—" if value is None else str(value)


def table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def serving_table() -> str:
    header = ["Stack", "c", "TTFT p50", "TTFT p95", "e2e p50", "e2e p95",
              "tok/s (req)", "tok/s (agg)", "pages/min", "errors"]
    rows: list[list[str]] = []
    for stack in discovered_stacks():
        data = load(RESULTS / f"{stack}.json")
        if not data:
            continue
        rows.extend(
            [
                PRETTY.get(stack, stack), level["concurrency"],
                level["ttft_s"]["p50"], level["ttft_s"]["p95"],
                level["e2e_s"]["p50"], level["e2e_s"]["p95"],
                level.get("req_tokens_per_s_mean"), level["agg_tokens_per_s"],
                level["pages_per_min"], level["errors"],
            ]
            for level in data["levels"]
        )
    return table(rows, header) if rows else "_No serving results yet — run `make bench`._"


def coldstart_table() -> str:
    rows = []
    for path in sorted(RESULTS.glob("coldstart_*.json")):
        data = load(path)
        if not data:
            continue
        rows.append([
            PRETTY.get(data["label"], data["label"]), data.get("mode", "process"),
            data["ready_s"], data["first_token_after_ready_s"],
            data["total_cold_to_first_token_s"],
        ])
    header = ["Stack", "mode", "ready (s)", "first token after ready (s)", "total (s)"]
    return table(rows, header) if rows else "_No cold-start measurements yet._"


def batch_table() -> str:
    data = load(RESULTS / "ray_batch.json")
    if not data:
        return "_No batch run yet — `make ray-batch`._"
    return table(
        [[data.get("label", "ray_batch"), data["pages"], data["wall_s"],
          data["pages_per_min"], data.get("gpu_util_mean", "n/a")]],
        ["Job", "pages", "wall (s)", "pages/min", "mean GPU util"],
    )


def ops_table() -> str:
    """Qualitative ledger (PLAN.md §5) — hand-recorded, but kept in JSON, not prose."""
    data = load(RESULTS / "ops_ledger.json")
    if not data:
        return "_No ops ledger recorded yet — see `benchmarks/ops_ledger.example.json`._"
    header = ["Stack", "install (min)", "control-plane RAM (GB)", "config LOC",
              "moving parts", "notes"]
    rows = [
        [PRETTY.get(k, k), v.get("install_minutes"), v.get("control_plane_ram_gb"),
         v.get("config_loc"), v.get("moving_parts"), v.get("notes", "")]
        for k, v in data.items()
        if isinstance(v, dict)          # skip the "_comment" key in the template
    ]
    return table(rows, header)


def headline() -> str:
    """One-line summary so a reader gets the answer before the tables."""
    best_ttft: tuple[str, float] | None = None
    best_throughput: tuple[str, float] | None = None
    for stack in discovered_stacks():
        data = load(RESULTS / f"{stack}.json")
        if not data:
            continue
        for level in data["levels"]:
            ttft = level["ttft_s"]["p50"]
            if ttft is not None and (best_ttft is None or ttft < best_ttft[1]):
                best_ttft = (f"{PRETTY.get(stack, stack)} @ c={level['concurrency']}", ttft)
            agg = level["agg_tokens_per_s"]
            if agg and (best_throughput is None or agg > best_throughput[1]):
                best_throughput = (f"{PRETTY.get(stack, stack)} @ c={level['concurrency']}", agg)
    if not best_ttft and not best_throughput:
        return ""
    parts = []
    if best_ttft:
        parts.append(f"lowest TTFT p50 **{best_ttft[1]}s** ({best_ttft[0]})")
    if best_throughput:
        parts.append(f"highest aggregate **{best_throughput[1]} tok/s** ({best_throughput[0]})")
    return "Across the measured runs: " + "; ".join(parts) + ".\n"


def build_block() -> str:
    sections = [
        BEGIN,
        headline(),  # empty until there is something to summarise
        "### Serving latency and throughput\n\n" + serving_table(),
        "### Cold start\n\n" + coldstart_table(),
        "### Batch inference (Ray Data)\n\n" + batch_table(),
        "### Ops overhead\n\n" + ops_table(),
        END,
    ]
    return "\n\n".join(section for section in sections if section)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=Path, default=TARGET)
    ap.add_argument("--print", action="store_true", help="print instead of patching the file")
    args = ap.parse_args()

    block = build_block()
    if args.print or not args.target.exists():
        print(block)
        return

    text = args.target.read_text(encoding="utf-8")
    if BEGIN in text and END in text:
        pre, rest = text.split(BEGIN, 1)
        _, post = rest.split(END, 1)
        args.target.write_text(pre + block + post, encoding="utf-8")
        print(f"patched {args.target}")
    else:
        print(f"markers not found in {args.target}; printing instead\n")
        print(block)


if __name__ == "__main__":
    main()
