"""What each branch rate costs at each (players, cards) shape.

The rate is the knob that decides how much of a tree gets built, and its
effect is exponential in hand size: the same rate that fits a 10-card game
leaves a 3-card game nearly unbranched. This sweeps rate x shape so the
per-shape rate table can be read off measurements rather than guessed.

One subprocess per shape (the model is built once and the rates run in
sequence inside it), so a wide shape's allocator pool cannot contaminate a
narrow one's peak reading.

    .venv/bin/python scripts/report_seq_rate_grid.py --rates 0.3,0.5,0.7,0.9
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    parser.add_argument("--rates", default="0.3,0.5,0.7,0.9")
    parser.add_argument("--deals", type=int, default=1)
    parser.add_argument("--max-cache-rows", type=int, default=65536)
    parser.add_argument("--bid-top-k", type=int, default=4)
    parser.add_argument("--play-mode", default="sample_k_plus_uniform")
    parser.add_argument("--play-top-k", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=320)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=10)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=960)
    parser.add_argument("--kv-dtype", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--cache-budget-gb", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="rate_grid.json")
    parser.add_argument("--one-shape", default=None)
    return parser.parse_args()


def measure(args, model, device, num_players: int, hand_size: int, rate: float) -> dict:
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
        branch_budget=BranchBudgetConfig(branch_rate=rate),
        rollout=RolloutOptions(
            deals_per_batch=args.deals,
            historical_arm="off",
            cache_budget_gb=args.cache_budget_gb,
            max_cache_rows=args.max_cache_rows,
        ),
        kv_dtype=args.kv_dtype,
    )
    collector = SeqRolloutCollector(model, train_config, device=device)
    started = time.perf_counter()
    trees = collector.collect(None, random.Random(args.seed))
    elapsed = time.perf_counter() - started
    stats = collector.stats

    length = model_config.seq_len(num_players, hand_size)
    positions = sum(length - leaf.owned_from for tree in trees for leaf in tree.leaves)
    branch_rows = sum(
        1
        for tree in trees
        for leaf in tree.leaves
        for record in leaf.decisions
        if record.branch is not None
    )
    # Did branching reach the endgame, or only the opening? Count decisions in
    # the last third of the tricks that actually branched.
    late_from = max(hand_size - hand_size // 3, 1)
    late = sum(
        count
        for tree in trees
        for stage, count in tree.branch_decisions_by_stage.items()
        if stage >= late_from
    )
    collector.release_caches()
    return {
        "players": num_players,
        "hand_size": hand_size,
        "rate": rate,
        "deals": args.deals,
        "sec": elapsed,
        "leaves": stats.leaves,
        "branch_rows": branch_rows,
        "late_branch_rows": late,
        "positions": positions,
        "peak_rows": stats.peak_cache_rows,
        "peak_gb": stats.peak_device_bytes / 1e9,
        "blocked_by_cache": stats.blocked_by_cache,
    }


def main() -> None:
    args = build_parser()
    rates = [float(part) for part in args.rates.split(",")]

    if args.one_shape is not None:
        num_players, hand_size = (int(p) for p in args.one_shape.split(","))
        device = torch.device(args.device) if args.device else best_seq_device()
        torch.manual_seed(args.seed)
        model = (
            SeqPlumpModel(
                SeqModelConfig(
                    d_model=args.d_model,
                    n_layers=args.n_layers,
                    n_heads=args.n_heads,
                    n_kv_heads=args.n_kv_heads,
                    d_ff=args.d_ff,
                )
            )
            .to(device)
            .eval()
        )
        for rate in rates:
            print(json.dumps(measure(args, model, device, num_players, hand_size, rate)))
        return

    player_counts = [int(p) for p in args.players.split(",")]
    hand_sizes = [int(n) for n in args.hand_sizes.split(",")]
    passthrough = [a for a in sys.argv[1:] if not a.startswith("--one-shape")]

    print(
        f"{'P':>2} {'N':>3} {'rate':>5} {'sec':>7} {'leaves':>7} {'branch':>7} "
        f"{'late':>6} {'positions':>10} {'rows':>7} {'peakGB':>7} {'blocked':>8}"
    )
    rows: list[dict] = []
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
            for line in result.stdout.strip().splitlines():
                row = json.loads(line)
                rows.append(row)
                print(
                    f"{row['players']:>2} {row['hand_size']:>3} {row['rate']:>5.2f} "
                    f"{row['sec']:>7.2f} {row['leaves']:>7d} "
                    f"{row['branch_rows']:>7d} {row['late_branch_rows']:>6d} "
                    f"{row['positions']:>10d} {row['peak_rows']:>7d} "
                    f"{row['peak_gb']:>7.2f} {row['blocked_by_cache']:>8d}"
                )
            sys.stdout.flush()

    out = Path(__file__).resolve().parents[1] / args.out
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
