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
import hashlib
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
from plump.seq.tokens import card_id


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
    parser.add_argument(
        "--branch-rate-override",
        type=float,
        default=None,
        help="benchmark-only branch-placement rate for every selected shape",
    )
    parser.add_argument(
        "--matched-two-deal",
        action="store_true",
        help=(
            "compare the same two deals as two serial one-deal waves and one "
            "paired two-deal wave"
        ),
    )
    # Internal child-process arguments.
    parser.add_argument("--one-case", default=None)
    parser.add_argument("--deals", type=int, choices=(1, 2), default=1)
    parser.add_argument("--games", type=int, default=None)
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


def _sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tree_fingerprints(trees) -> dict[str, str]:
    """Identify matched deals and their realized discrete rollout workload.

    Policy probabilities are rounded only for the numerical fingerprint so
    harmless backend noise below 1e-6 does not obscure an otherwise identical
    batch-vs-serial rollout. The discrete fingerprint remains exact.
    """

    deal_payload = []
    discrete_payload = []
    canonical_discrete_payload = []
    policy_payload = []
    for tree in trees:
        deal = {
            "players": tree.num_players,
            "hand_size": tree.hand_size,
            "focal": tree.focal,
            "start": tree.bidding_start_player,
            "hands": {
                str(player): sorted(card_id(card) for card in hand)
                for player, hand in sorted(tree.initial_hands.items())
            },
        }
        deal_payload.append(deal)
        leaves = []
        canonical_leaves = []
        policies = []
        for leaf in tree.leaves:
            decisions = []
            probability_rows = []
            for record in leaf.decisions:
                branch = record.branch
                decisions.append(
                    {
                        "position": record.position,
                        "phase": record.phase,
                        "sampled": record.action_index,
                        "candidates": (
                            None if branch is None else branch.candidate_indices
                        ),
                        "children": (
                            None
                            if branch is None
                            else sorted(branch.child_values.items())
                        ),
                    }
                )
                probability_rows.append(
                    {
                        "old": [round(float(value), 6) for value in record.old_probs],
                        "prior": (
                            None
                            if branch is None
                            else [
                                round(float(value), 6) for value in branch.prior_probs
                            ]
                        ),
                        "q": (
                            None
                            if branch is None
                            else [
                                round(float(value), 6)
                                for value in branch.inclusion_probs
                            ]
                        ),
                    }
                )
            round_state = leaf.env.state.current_round
            leaf_payload = {
                "owned_from": leaf.owned_from,
                "spine": leaf.on_policy_spine,
                "terminal": leaf.terminal_value,
                "scores": round_state.round_scores,
                "decisions": decisions,
            }
            leaves.append(leaf_payload)
            canonical_leaves.append(
                {
                    **leaf_payload,
                    "decisions": [
                        {
                            **decision,
                            "candidates": (
                                None
                                if decision["candidates"] is None
                                else sorted(decision["candidates"])
                            ),
                        }
                        for decision in decisions
                    ],
                }
            )
            policies.append(probability_rows)
        discrete_payload.append(
            {
                "deal": deal,
                "leaf_total": tree.leaf_total,
                "decision_total": tree.decision_total,
                "leaves": leaves,
            }
        )
        canonical_discrete_payload.append(
            {
                "deal": deal,
                "leaf_total": tree.leaf_total,
                "decision_total": tree.decision_total,
                "leaves": sorted(
                    canonical_leaves,
                    key=lambda value: json.dumps(
                        value, sort_keys=True, separators=(",", ":")
                    ),
                ),
            }
        )
        policy_payload.append(policies)
    return {
        "deal_fingerprint": _sha256(deal_payload),
        "ordered_discrete_fingerprint": _sha256(discrete_payload),
        "canonical_discrete_fingerprint": _sha256(canonical_discrete_payload),
        "policy_fingerprint_6dp": _sha256(policy_payload),
    }


def run_one_case(args: argparse.Namespace) -> dict[str, Any]:
    players, hand_size = (int(part) for part in args.one_case.split(","))
    run = RunDirectory(args.run)
    checkpoint = run.resolve_checkpoint(args.checkpoint)
    resolved = load_training_config(args.config)
    rate = resolved.training.branch_budget.rate_for_shape(players, hand_size)
    if args.branch_rate_override is not None:
        rate = args.branch_rate_override
    if rate is None:
        raise RuntimeError(f"No branch rate for {players}p/{hand_size}c.")
    if not 0.0 <= rate <= 1.0:
        raise ValueError("--branch-rate-override must be in [0, 1].")

    device = torch.device(args.device) if args.device else best_seq_device()
    seed = 91_000 + args.repeat * 1_000 + players * 100 + hand_size
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = _load_model(checkpoint, device)
    games = args.games if args.games is not None else args.deals
    if games < args.deals or games % args.deals:
        raise ValueError("--games must be a positive multiple of --deals.")

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
                games=games,
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
    shape_cost = stats.by_shape[(players, hand_size)]
    mode = (
        "serial"
        if games == 2 and args.deals == 1
        else "paired"
        if games == 2 and args.deals == 2
        else "solo"
        if games == 1
        else f"batch_{args.deals}"
    )
    result = {
        "players": players,
        "hand_size": hand_size,
        "mode": mode,
        "games": games,
        "batch_size": args.deals,
        "batches": shape_cost.batches,
        "repeat": args.repeat,
        "seed": seed,
        "branch_rate": rate,
        "device": str(device),
        "completed": True,
        "wall_sec": wall_sec,
        "sec_per_deal": wall_sec / games,
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
        **_tree_fingerprints(trees),
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


def _matched_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if not row.get("completed") or row.get("mode") not in ("serial", "paired"):
            continue
        key = (int(row["players"]), int(row["hand_size"]), int(row["repeat"]))
        groups.setdefault(key, {})[str(row["mode"])] = row

    comparisons = []
    for (players, hand_size, repeat), group in sorted(groups.items()):
        if set(group) != {"serial", "paired"}:
            continue
        serial, paired = group["serial"], group["paired"]
        game_speedup = float(serial["wall_sec"]) / float(paired["wall_sec"])
        row_speedup = (
            float(paired["forward_rows"])
            / float(paired["wall_sec"])
            / (float(serial["forward_rows"]) / float(serial["wall_sec"]))
        )
        if row_speedup < 1.9:
            scaling = "sublinear"
        elif row_speedup <= 2.1:
            scaling = "linear"
        else:
            scaling = "superlinear"
        comparisons.append(
            {
                "players": players,
                "hand_size": hand_size,
                "repeat": repeat,
                "branch_rate": float(serial["branch_rate"]),
                "serial_sec": float(serial["wall_sec"]),
                "paired_sec": float(paired["wall_sec"]),
                "game_throughput_speedup": game_speedup,
                "row_throughput_speedup": row_speedup,
                "batch_efficiency": row_speedup / 2.0,
                "scaling": scaling,
                "paired_is_faster": game_speedup > 1.0,
                "identical_deals": (
                    serial["deal_fingerprint"] == paired["deal_fingerprint"]
                ),
                "identical_tree_realization": (
                    serial["canonical_discrete_fingerprint"]
                    == paired["canonical_discrete_fingerprint"]
                ),
                "identical_leaf_order": (
                    serial["ordered_discrete_fingerprint"]
                    == paired["ordered_discrete_fingerprint"]
                ),
                "identical_policy_6dp": (
                    serial["policy_fingerprint_6dp"] == paired["policy_fingerprint_6dp"]
                ),
                "identical_aggregate_work": all(
                    serial[name] == paired[name]
                    for name in (
                        "trees",
                        "leaves",
                        "decisions",
                        "raw_positions",
                        "forward_rows",
                        "branch_decisions",
                        "blocked_by_cache",
                        "skipped_by_placement",
                    )
                ),
                "serial_peak_device_gb": float(serial["peak_device_gb"]),
                "paired_peak_device_gb": float(paired["peak_device_gb"]),
                "serial_peak_cache_rows": int(serial["peak_cache_rows"]),
                "paired_peak_cache_rows": int(paired["peak_cache_rows"]),
                "serial_forward_rows": int(serial["forward_rows"]),
                "paired_forward_rows": int(paired["forward_rows"]),
            }
        )
    return comparisons


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
    if args.branch_rate_override is not None and not (
        0.0 <= args.branch_rate_override <= 1.0
    ):
        raise ValueError("--branch-rate-override must be in [0, 1].")
    run = RunDirectory(args.run)
    checkpoint = run.resolve_checkpoint(args.checkpoint)
    output = args.output
    if output is None:
        report_name = (
            "stratified_matched_batch_grid"
            if args.matched_two_deal
            else "stratified_shape_grid"
        )
        output = (
            run.path
            / "benchmarks"
            / f"{report_name}_iter_{int(args.checkpoint):06d}.json"
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
        "matched_two_deal": args.matched_two_deal,
        "branch_rate_override": args.branch_rate_override,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    for repeat in range(args.repeats):
        for hand_size in hand_sizes:
            for player_count in players:
                cases = (
                    ((1, 2, "serial"), (2, 2, "paired"))
                    if args.matched_two_deal
                    else ((1, 1, "solo"), (2, 2, "paired"))
                )
                for deals, games, expected_mode in cases:
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
                        "--games",
                        str(games),
                        "--repeat",
                        str(repeat),
                        "--one-case",
                        f"{player_count},{hand_size}",
                    ]
                    if args.device:
                        command.extend(("--device", args.device))
                    if args.branch_rate_override is not None:
                        command.extend(
                            (
                                "--branch-rate-override",
                                str(args.branch_rate_override),
                            )
                        )
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
                                "mode": expected_mode,
                                "games": games,
                                "batch_size": deals,
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
                            "mode": expected_mode,
                            "games": games,
                            "batch_size": deals,
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
                        f"{expected_mode}: {status}",
                        flush=True,
                    )
                    comparisons = _matched_comparisons(rows)
                    atomic_write_json(
                        output,
                        {
                            "metadata": metadata,
                            "runs": rows,
                            "summary": _summaries(rows),
                            "comparisons": comparisons,
                        },
                    )

    summaries = _summaries(rows)
    comparisons = _matched_comparisons(rows)
    metadata["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_write_json(
        output,
        {
            "metadata": metadata,
            "runs": rows,
            "summary": summaries,
            "comparisons": comparisons,
        },
    )
    _write_csv(output.with_suffix(".csv"), summaries)
    if comparisons:
        _write_csv(
            output.with_name(f"{output.stem}_comparisons.csv"),
            comparisons,
        )
    print(f"Wrote matched benchmark results under {output.parent}.")


def main() -> None:
    args = build_parser().parse_args()
    if args.one_case:
        print(json.dumps(run_one_case(args), sort_keys=True))
    else:
        run_grid(args)


if __name__ == "__main__":
    main()
