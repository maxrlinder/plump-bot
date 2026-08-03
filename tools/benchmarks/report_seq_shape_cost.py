"""Cost of one rollout per (players, hand size), for sizing the game schedule.

Answers the question the schedule needs answered: what does one deal of each
shape cost in time, memory and cache rows, and how many training positions does
it yield? Run per shape in a fresh subprocess so allocator state from a wide
shape cannot contaminate a narrow one.

    .venv/bin/python tools/benchmarks/report_seq_shape_cost.py --deals 1
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plump.seq.config import (
    BranchBudgetConfig,
    BranchRuleConfig,
    GameScheduleCell,
    RolloutOptions,
    SeqModelConfig,
    SeqTrainingConfig,
    seq_len,
)
from plump.seq.model import SeqPlumpModel
from plump.seq.policy import best_seq_device
from plump.seq.rollout import SeqRolloutCollector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", default="3,4,5")
    parser.add_argument("--hand-sizes", default="3,4,5,6,7,8,9,10")
    parser.add_argument("--deals", type=int, default=1)
    parser.add_argument("--branch-rate", type=float, default=0.5)
    parser.add_argument("--bid-top-k", type=int, default=4)
    parser.add_argument("--play-mode", default="sample_k_plus_uniform")
    parser.add_argument("--play-top-k", type=int, default=3)
    parser.add_argument("--bid-split-groups", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=320)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=10)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=960)
    parser.add_argument("--kv-dtype", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--cache-budget-gb", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    # Internal: run exactly one shape and print a JSON line.
    parser.add_argument("--one-shape", default=None)
    return parser


def run_one_shape(args, num_players: int, hand_size: int) -> dict:
    device = torch.device(args.device) if args.device else best_seq_device()
    model_config = SeqModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        d_ff=args.d_ff,
    )
    train_config = SeqTrainingConfig(
        schedule_cells=(
            GameScheduleCell(
                hand_size=hand_size, num_players=num_players, games=args.deals
            ),
        ),
        branch_rule=BranchRuleConfig(
            bid_top_k=args.bid_top_k,
            play_mode=args.play_mode,
            play_top_k=args.play_top_k,
        ),
        branch_budget=BranchBudgetConfig(branch_rate=args.branch_rate),
        rollout=RolloutOptions(
            deals_per_batch=args.deals,
            opponent_mode="off",
            opponent_fraction=0.0,
            bid_split_groups=args.bid_split_groups,
            cache_budget_gb=args.cache_budget_gb,
        ),
        kv_dtype=args.kv_dtype,
    )
    torch.manual_seed(args.seed)
    model = SeqPlumpModel(model_config).to(device).eval()
    collector = SeqRolloutCollector(model, train_config, device=device)

    started = time.perf_counter()
    trees = collector.collect(None, random.Random(args.seed))
    elapsed = time.perf_counter() - started
    stats = collector.stats

    length = model_config.seq_len(num_players, hand_size)
    positions = sum(
        length - leaf.owned_from for tree in trees for leaf in tree.leaves
    )
    branch_rows = sum(
        1
        for tree in trees
        for leaf in tree.leaves
        for record in leaf.decisions
        if record.branch is not None
    )
    return {
        "players": num_players,
        "hand_size": hand_size,
        "length": length,
        "deals": args.deals,
        "sec": elapsed,
        "leaves": stats.leaves,
        "decisions": stats.decisions,
        "branch_rows": branch_rows,
        "positions": positions,
        "peak_rows": stats.peak_cache_rows,
        "peak_gb": stats.peak_device_bytes / 1e9,
        "bytes_per_row": stats.bytes_per_row,
        "blocked_by_cache": stats.blocked_by_cache,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.one_shape is not None:
        num_players, hand_size = (int(p) for p in args.one_shape.split(","))
        print(json.dumps(run_one_shape(args, num_players, hand_size)))
        return

    player_counts = [int(p) for p in args.players.split(",")]
    hand_sizes = [int(n) for n in args.hand_sizes.split(",")]
    passthrough = [a for a in sys.argv[1:] if not a.startswith("--one-shape")]

    print(
        f"{'P':>2} {'N':>3} {'len':>4} {'sec':>7} {'leaves':>7} {'positions':>10} "
        f"{'pos/s':>8} {'rows':>7} {'peakGB':>7} {'sec/deal':>9} {'blocked':>8}"
    )
    rows = []
    for num_players in player_counts:
        for hand_size in hand_sizes:
            result = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    *passthrough,
                    "--one-shape",
                    f"{num_players},{hand_size}",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                tail = result.stderr.strip().splitlines()[-1:] or ["failed"]
                print(f"{num_players:>2} {hand_size:>3}  {tail[0][:70]}")
                continue
            row = json.loads(result.stdout.strip().splitlines()[-1])
            rows.append(row)
            blocked = row["blocked_by_cache"]
            print(
                f"{row['players']:>2} {row['hand_size']:>3} {row['length']:>4} "
                f"{row['sec']:>7.2f} {row['leaves']:>7d} {row['positions']:>10d} "
                f"{row['positions'] / max(row['sec'], 1e-9):>8.0f} "
                f"{row['peak_rows']:>7d} {row['peak_gb']:>7.2f} "
                f"{row['sec'] / max(row['deals'], 1):>9.2f} {blocked:>8d}"
            )

    out = Path(__file__).resolve().parents[2] / "shape_cost.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
