"""Sweep rollout shapes and model sizes; print one table.

Each configuration runs in its own process so that peak-memory readings are
not contaminated by a previous run's allocator pool.

    .venv/bin/python tools/benchmarks/sweep_seq_rollout.py --sweep hands
    .venv/bin/python tools/benchmarks/sweep_seq_rollout.py --sweep model
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "tools" / "benchmarks" / "report_seq_branch_shape.py"
PYTHON = ROOT / ".venv" / "bin" / "python"

# d_model, n_layers, n_heads, d_ff
MODELS = {
    "tiny": (192, 4, 6, 576),
    "small": (256, 6, 8, 768),
    "large": (320, 8, 10, 960),
    "xl": (384, 10, 12, 1152),
}

COLUMNS = (
    "players hands deals split kvheads params_m sec fwd_sec copy_sec "
    "peak_rows cache_gb alloc_gb mem_gb live_gb leaves positions blk_cache"
).split()


def run(label: str, model: str, extra: list[str], base: list[str]) -> list[str]:
    d_model, n_layers, n_heads, d_ff = MODELS[model]
    command = [
        str(PYTHON),
        str(REPORT),
        "--summary-only",
        "--profile",
        "--d-model", str(d_model),
        "--n-layers", str(n_layers),
        "--n-heads", str(n_heads),
        "--d-ff", str(d_ff),
        *base,
        *extra,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        tail = result.stderr.strip().splitlines()[-1:] or ["failed"]
        return [label, model] + ["ERR"] * (len(COLUMNS) - 2) + tail[-1:]
    return [label, model] + result.stdout.strip().split("\t")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        choices=["hands", "players", "model", "kv", "split"],
        default="hands",
    )
    parser.add_argument("--model", default="small")
    parser.add_argument("--branch-rate", type=float, default=0.5)
    parser.add_argument("--games-per-hand", type=int, default=5)
    parser.add_argument("--deals-per-batch", type=int, default=5)
    parser.add_argument("--play-top-k", type=int, default=3)
    parser.add_argument("--cache-budget-gb", type=float, default=8.0)
    parser.add_argument("--n-kv-heads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = [
        "--branch-rate", str(args.branch_rate),
        "--play-mode", "sample_k",
        "--play-top-k", str(args.play_top_k),
        "--games-per-hand", str(args.games_per_hand),
        "--deals-per-batch", str(args.deals_per_batch),
        "--cache-budget-gb", str(args.cache_budget_gb),
        "--historical-arm", "off",
        "--seed", str(args.seed),
    ]
    if args.n_kv_heads is not None:
        base += ["--n-kv-heads", str(args.n_kv_heads)]

    jobs: list[tuple[str, str, list[str]]] = []
    if args.sweep == "hands":
        for hand in (3, 4, 5, 6, 7, 8, 9, 10):
            jobs.append(
                (f"hand{hand}", args.model, ["--hand-sizes", str(hand), "--players", "5"])
            )
    elif args.sweep == "players":
        for players in (3, 4, 5):
            jobs.append(
                (f"{players}p", args.model, ["--players", str(players)])
            )
    elif args.sweep == "model":
        for name in ("tiny", "small", "large", "xl"):
            jobs.append((name, name, ["--players", "5"]))
    elif args.sweep == "kv":
        for heads in (None, 4, 2, 1):
            label = "mha" if heads is None else f"kv{heads}"
            extra = ["--players", "5"]
            if heads is not None:
                extra += ["--n-kv-heads", str(heads)]
            jobs.append((label, args.model, extra))
    else:
        for split in (1, 2, 4):
            jobs.append(
                (f"split{split}", args.model,
                 ["--players", "5", "--bid-split-groups", str(split)])
            )

    header = ["config", "model"] + COLUMNS
    widths = [max(len(header[i]), 9) for i in range(len(header))]
    print(" ".join(name.rjust(widths[i]) for i, name in enumerate(header)))
    print("-" * (sum(widths) + len(widths) - 1))
    for label, model, extra in jobs:
        row = run(label, model, extra, base)
        print(" ".join(str(cell).rjust(widths[i]) for i, cell in enumerate(row)))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
