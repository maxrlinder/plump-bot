"""Benchmark stratified rollout collection for every training game shape.

Every (players, cards, solo/paired, repeat) case runs in a fresh process so
MPS allocator state from one shape cannot contaminate the next. The benchmark
loads real checkpoint weights, uses the active configured branch-rate table,
and measures collection only: this is the phase whose wave packing and KV
cache determine whether two simultaneous deals fit.

Results are written under the selected run, never the repository root.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from plump.run_config import PROJECT_ROOT, load_training_config
from plump.runs import RunDirectory, atomic_write_json
from plump.seq.config import GameScheduleCell
from plump.seq.model import SeqPlumpModel
from plump.seq.policy import best_seq_device
from plump.seq.rollout import SeqRolloutCollector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="mirror-8m")
    parser.add_argument("--checkpoint", default="200")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/train.toml"
    )
    parser.add_argument("--players", default="3,4,5")
    parser.add_argument("--hand-sizes", default="3,4,5,6,7,8,9,10")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output", type=Path, default=None)
    # Internal child-process arguments.
    parser.add_argument("--one-case", default=None)
    parser.add_argument("--deals", type=int, choices=(1, 2), default=1)
    parser.add_argument("--repeat", type=int, default=0)
    return parser


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part)


def _load_model(checkpoint: Path, device: torch.device) -> SeqPlumpModel:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    from plump.seq.config import SeqModelConfig

    model = SeqPlumpModel(SeqModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval()


def run_one_case(args: argparse.Namespace) -> dict[str, Any]:
    players, hand_size = (int(part) for part in args.one_case.split(","))
    run = RunDirectory(args.run)
    checkpoint = run.resolve_checkpoint(args.checkpoint)
    resolved = load_training_config(args.config)
    rate = resolved.training.branch_budget.rate_for_shape(players, hand_size)
    if rate is None:
        raise RuntimeError(f"No branch rate for {players}p/{hand_size}c.")

    device = torch.device(args.device) if args.device else best_seq_device()
    seed = 91_000 + args.repeat * 1_000 + players * 100 + hand_size
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = _load_model(checkpoint, device)

    rollout = replace(
        resolved.training.rollout,
        auto_deals_per_batch=False,
        deals_per_batch=args.deals,
        parallel_deals_max_hand_size=None,
        historical_arm="off",
    )
    training = replace(
        resolved.training,
        schedule_cells=(
            GameScheduleCell(
                hand_size=hand_size,
                num_players=players,
                games=args.deals,
            ),
        ),
        branch_budget=replace(
            resolved.training.branch_budget,
            branch_rate=rate,
            branch_rate_by_shape=(),
        ),
        rollout=rollout,
    )
    collector = SeqRolloutCollector(model, training, device=device)

    started = time.perf_counter()
    trees = collector.collect(
        None,
        random.Random(seed),
        iteration=int(args.checkpoint) if str(args.checkpoint).isdigit() else 0,
    )
    wall_sec = time.perf_counter() - started
    stats = collector.stats
    length = model.config.seq_len(players, hand_size)
    raw_positions = sum(
        length - leaf.owned_from for tree in trees for leaf in tree.leaves
    )
    result = {
        "players": players,
        "hand_size": hand_size,
        "mode": "solo" if args.deals == 1 else "paired",
        "deals": args.deals,
        "repeat": args.repeat,
        "seed": seed,
        "branch_rate": rate,
        "device": str(device),
        "completed": True,
        "wall_sec": wall_sec,
        "sec_per_deal": wall_sec / args.deals,
        "trees": len(trees),
        "leaves": stats.leaves,
        "decisions": stats.decisions,
        "raw_positions": raw_positions,
        "forward_rows": stats.forward_rows,
        "branch_decisions": stats.branch_decisions,
        "peak_cache_rows": stats.peak_cache_rows,
        "cache_rows_allocated": stats.cache_rows_allocated,
        "cache_pressure": stats.peak_cache_rows / max(stats.cache_rows_allocated, 1),
        "peak_device_gb": stats.peak_device_bytes / (1024**3),
        "blocked_by_cache": stats.blocked_by_cache,
        "skipped_by_placement": stats.skipped_by_placement,
        "sample_sec": stats.sample_sec,
        "step_sec": stats.step_sec,
        "compact_sec": stats.compact_sec,
        "token_build_sec": stats.token_build_sec,
        "forward_sec": stats.forward_sec,
    }
    collector.release_caches()
    return result


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["players"]), int(row["hand_size"]), str(row["mode"]))
        groups.setdefault(key, []).append(row)

    summaries = []
    for (players, hand_size, mode), group in sorted(groups.items()):
        completed = [row for row in group if row.get("completed")]

        def median(name: str) -> float | None:
            values = [float(row[name]) for row in completed]
            return statistics.median(values) if values else None

        summaries.append(
            {
                "players": players,
                "hand_size": hand_size,
                "mode": mode,
                "attempts": len(group),
                "completed": len(completed),
                "branch_rate": (
                    float(completed[0]["branch_rate"]) if completed else None
                ),
                "median_wall_sec": median("wall_sec"),
                "median_sec_per_deal": median("sec_per_deal"),
                "median_leaves": median("leaves"),
                "median_decisions": median("decisions"),
                "median_raw_positions": median("raw_positions"),
                "median_forward_rows": median("forward_rows"),
                "max_peak_cache_rows": (
                    max(int(row["peak_cache_rows"]) for row in completed)
                    if completed
                    else None
                ),
                "max_peak_device_gb": (
                    max(float(row["peak_device_gb"]) for row in completed)
                    if completed
                    else None
                ),
                "max_blocked_by_cache": (
                    max(int(row["blocked_by_cache"]) for row in completed)
                    if completed
                    else None
                ),
                "all_untruncated": bool(completed)
                and len(completed) == len(group)
                and all(int(row["blocked_by_cache"]) == 0 for row in completed),
            }
        )
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_grid(args: argparse.Namespace) -> None:
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1.")
    run = RunDirectory(args.run)
    checkpoint = run.resolve_checkpoint(args.checkpoint)
    output = args.output
    if output is None:
        output = (
            run.path
            / "benchmarks"
            / f"stratified_shape_grid_iter_{int(args.checkpoint):06d}.json"
        )
    output = output.expanduser().resolve()
    players = _parse_ints(args.players)
    hand_sizes = _parse_ints(args.hand_sizes)
    rows: list[dict[str, Any]] = []
    metadata = {
        "run": args.run,
        "checkpoint": str(checkpoint),
        "config": str(args.config.resolve()),
        "repeats": args.repeats,
        "players": list(players),
        "hand_sizes": list(hand_sizes),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    for repeat in range(args.repeats):
        for hand_size in hand_sizes:
            for player_count in players:
                for deals in (1, 2):
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--run",
                        args.run,
                        "--checkpoint",
                        str(args.checkpoint),
                        "--config",
                        str(args.config),
                        "--deals",
                        str(deals),
                        "--repeat",
                        str(repeat),
                        "--one-case",
                        f"{player_count},{hand_size}",
                    ]
                    if args.device:
                        command.extend(("--device", args.device))
                    started = time.perf_counter()
                    try:
                        process = subprocess.run(
                            command,
                            capture_output=True,
                            text=True,
                            timeout=args.timeout,
                        )
                        if process.returncode:
                            row = {
                                "players": player_count,
                                "hand_size": hand_size,
                                "mode": "solo" if deals == 1 else "paired",
                                "deals": deals,
                                "repeat": repeat,
                                "completed": False,
                                "wall_sec": time.perf_counter() - started,
                                "error": (
                                    process.stderr.strip().splitlines()[-1]
                                    if process.stderr.strip()
                                    else f"exit {process.returncode}"
                                ),
                            }
                        else:
                            row = json.loads(process.stdout.strip().splitlines()[-1])
                    except subprocess.TimeoutExpired:
                        row = {
                            "players": player_count,
                            "hand_size": hand_size,
                            "mode": "solo" if deals == 1 else "paired",
                            "deals": deals,
                            "repeat": repeat,
                            "completed": False,
                            "wall_sec": time.perf_counter() - started,
                            "error": f"timeout after {args.timeout:.0f}s",
                        }
                    rows.append(row)
                    status = (
                        f"{float(row['wall_sec']):.2f}s "
                        f"{float(row.get('peak_device_gb', 0)):.2f}GB "
                        f"blocked={int(row.get('blocked_by_cache', 0))}"
                        if row["completed"]
                        else f"FAILED: {row['error']}"
                    )
                    print(
                        f"repeat {repeat + 1}/{args.repeats} "
                        f"{player_count}p/{hand_size}c "
                        f"{'solo' if deals == 1 else 'paired'}: {status}",
                        flush=True,
                    )
                    atomic_write_json(
                        output,
                        {
                            "metadata": metadata,
                            "runs": rows,
                            "summary": _summaries(rows),
                        },
                    )

    summaries = _summaries(rows)
    metadata["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_write_json(
        output,
        {"metadata": metadata, "runs": rows, "summary": summaries},
    )
    _write_csv(output.with_suffix(".csv"), summaries)
    print(f"Wrote {output} and {output.with_suffix('.csv')}.")


def main() -> None:
    args = build_parser().parse_args()
    if args.one_case:
        print(json.dumps(run_one_case(args), sort_keys=True))
    else:
        run_grid(args)


if __name__ == "__main__":
    main()
