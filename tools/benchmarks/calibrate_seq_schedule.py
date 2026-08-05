"""Time and size one full collect() over an apportioned game schedule.

This is the sizing tool for the rollout mix: given a total deal count and a
tilt toward longer games, it reports what one update's worth of collection
costs and how the trees are distributed over the (players, cards) grid.

    .venv/bin/python tools/benchmarks/calibrate_seq_schedule.py --games-total 120
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plump.seq.config import (
    GameScheduleCell,
    BranchBudgetConfig,
    BranchRuleConfig,
    RolloutOptions,
    SeqModelConfig,
    SeqTrainingConfig,
    ShapeBranchRate,
    build_branch_rate_table,
    build_game_schedule,
    build_position_balanced_schedule,
    seq_len,
)


def parse_rate_table(spec: str) -> tuple[ShapeBranchRate, ...]:
    """'*:*:0.9,*:10:0.5' -> shape rate rules ('*' matches anything)."""

    rules = []
    for part in spec.split(","):
        players, hand_size, rate = part.split(":")
        rules.append(
            ShapeBranchRate(
                rate=float(rate),
                num_players=None if players == "*" else int(players),
                hand_size=None if hand_size == "*" else int(hand_size),
            )
        )
    return tuple(rules)
from plump.seq.model import SeqPlumpModel
from plump.seq.policy import best_seq_device
from plump.seq.rollout import SeqRolloutCollector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-total", type=int, default=120)
    parser.add_argument("--hand-size-tilt", type=float, default=1.0)
    parser.add_argument(
        "--schedule",
        choices=["tilted", "balanced", "grid"],
        default="tilted",
        help="balanced = one deal per (players, cards, bidding position); "
             "grid = one deal per (players, cards), bidding position drawn",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--branch-rate", type=float, default=None)
    parser.add_argument(
        "--rate-table",
        default=None,
        help="per-shape rates as P:N:rate,... ; P or N may be '*' "
             "(e.g. '*:*:0.9,*:9:0.7,*:10:0.5')",
    )
    parser.add_argument("--max-cache-rows", type=int, default=None)
    parser.add_argument("--bid-top-k", type=int, default=4)
    parser.add_argument("--play-mode", default="sample_k_plus_uniform")
    parser.add_argument("--play-top-k", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=320)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=10)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=960)
    parser.add_argument("--no-trick-win-token", action="store_true")
    parser.add_argument(
        "--turn-token", choices=["off", "bid", "all"], default="off"
    )
    parser.add_argument(
        "--kv-dtype", choices=["fp32", "fp16", "bf16"], default="fp16"
    )
    parser.add_argument("--cache-budget-gb", type=float, default=10.0)
    parser.add_argument("--auto-target-rows", type=int, default=None)
    parser.add_argument("--auto-headroom", type=float, default=0.5)
    parser.add_argument("--max-deals-per-batch", type=int, default=64)
    parser.add_argument("--deals-per-batch", type=int, default=None,
                        help="fixed batch size; disables auto sizing")
    parser.add_argument(
        "--reference-rate",
        type=float,
        default=None,
        help="derive the per-shape table from this rate at 10 cards",
    )
    parser.add_argument("--player-exponent", type=float, default=0.0)
    parser.add_argument("--exhaustive-until", type=int, default=7,
                        help="branch every decision at or below this hand size")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else best_seq_device()
    if args.schedule == "balanced":
        schedule = build_position_balanced_schedule(repeats=args.repeats)
    elif args.schedule == "grid":
        # Same 24 shapes, one deal each: the control for what covering every
        # bidding position actually costs.
        schedule = tuple(
            GameScheduleCell(hand_size=n, num_players=p, games=args.repeats)
            for p in (3, 4, 5)
            for n in range(3, 11)
        )
    else:
        schedule = build_game_schedule(
            games_total=args.games_total, hand_size_tilt=args.hand_size_tilt
        )
    model_config = SeqModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        d_ff=args.d_ff,
        trick_win_token=not args.no_trick_win_token,
        turn_token=args.turn_token,
    )
    if args.rate_table:
        rate_table = parse_rate_table(args.rate_table)
    elif args.reference_rate is not None:
        rate_table = build_branch_rate_table(
            args.reference_rate,
            exhaustive_until=args.exhaustive_until,
            player_exponent=args.player_exponent,
        )
    else:
        rate_table = ()

    auto = args.deals_per_batch is None
    train_config = SeqTrainingConfig(
        schedule_cells=schedule,
        branch_rule=BranchRuleConfig(
            bid_top_k=args.bid_top_k,
            play_mode=args.play_mode,
            play_top_k=args.play_top_k,
        ),
        branch_budget=BranchBudgetConfig(
            branch_rate=args.branch_rate,
            branch_rate_by_shape=rate_table,
        ),
        rollout=RolloutOptions(
            deals_per_batch=args.deals_per_batch or 1,
            auto_deals_per_batch=auto,
            auto_target_rows=args.auto_target_rows,
            auto_deals_headroom=args.auto_headroom,
            max_deals_per_batch=args.max_deals_per_batch,
            opponent_mode="off",
            opponent_fraction=0.0,
            cache_budget_gb=args.cache_budget_gb,
            max_cache_rows=args.max_cache_rows,
        ),
        kv_dtype=args.kv_dtype,
    )
    torch.manual_seed(args.seed)
    model = SeqPlumpModel(model_config).to(device).eval()
    collector = SeqRolloutCollector(model, train_config, device=device)

    print(
        f"device={device} cells={len(schedule)} games={args.games_total} "
        f"schedule={args.schedule} batching="
        f"{'auto' if auto else args.deals_per_batch}"
    )

    for iteration in range(args.iterations):
        started = time.perf_counter()
        trees = collector.collect(None, random.Random(args.seed + iteration))
        elapsed = time.perf_counter() - started
        stats = collector.stats
        positions = sum(
            model_config.seq_len(tree.num_players, tree.hand_size)
            - leaf.owned_from
            for tree in trees
            for leaf in tree.leaves
        )
        branch_rows = sum(
            1
            for tree in trees
            for leaf in tree.leaves
            for record in leaf.decisions
            if record.branch is not None
        )
        label = "warmup" if iteration == 0 else f"iter {iteration}"
        print(
            f"{label:8s} {elapsed:6.1f}s | trees {len(trees):5d} "
            f"leaves {stats.leaves:7d} positions {positions:8d} "
            f"branch rows {branch_rows:7d} | "
            f"{positions / elapsed:8.0f} pos/s | "
            f"peak {stats.peak_device_bytes / 1e9:5.2f}GB "
            f"rows {stats.peak_cache_rows:6d} | "
            f"blocked by cache {stats.blocked_by_cache}"
        )

    positions_by_shape: dict[tuple[int, int], int] = defaultdict(int)
    branch_by_shape: dict[tuple[int, int], int] = defaultdict(int)
    for tree in trees:
        key = (tree.num_players, tree.hand_size)
        positions_by_shape[key] += sum(
            model_config.seq_len(tree.num_players, tree.hand_size)
            - leaf.owned_from
            for leaf in tree.leaves
        )
        branch_by_shape[key] += sum(
            1
            for leaf in tree.leaves
            for record in leaf.decisions
            if record.branch is not None
        )

    costs = collector.stats.by_shape
    total_sec = sum(cost.sec for cost in costs.values())
    total_positions = sum(positions_by_shape.values())
    total_branch = sum(branch_by_shape.values())
    budget = train_config.branch_budget
    print(
        f"\n{'P':>2} {'N':>3} {'rate':>5} {'deals':>6} {'batch':>6} {'sec':>7} "
        f"{'sec%':>6} {'leaves':>8} {'positions':>10} {'pos%':>6} "
        f"{'branch':>7} {'br%':>6} {'sec/deal':>9}"
    )
    for shape in sorted(costs):
        players, hand_size = shape
        cost = costs[shape]
        rate = budget.rate_for_shape(players, hand_size)
        print(
            f"{players:>2} {hand_size:>3} "
            f"{'-' if rate is None else f'{rate:.2f}':>5} "
            f"{cost.deals:>6} {cost.deals_per_batch:>6.1f} {cost.sec:>7.2f} "
            f"{100 * cost.sec / max(total_sec, 1e-9):>5.1f}% {cost.leaves:>8} "
            f"{positions_by_shape[shape]:>10} "
            f"{100 * positions_by_shape[shape] / max(total_positions, 1):>5.1f}% "
            f"{branch_by_shape[shape]:>7} "
            f"{100 * branch_by_shape[shape] / max(total_branch, 1):>5.1f}% "
            f"{cost.sec / max(cost.deals, 1):>9.3f}"
        )

    print("\nby hand size:")
    print(f"{'N':>3} {'sec':>7} {'sec%':>6} {'positions':>10} {'pos%':>6} "
          f"{'branch':>7} {'br%':>6}")
    for hand_size in sorted({n for _, n in costs}):
        sec = sum(c.sec for s, c in costs.items() if s[1] == hand_size)
        pos = sum(v for s, v in positions_by_shape.items() if s[1] == hand_size)
        branch = sum(v for s, v in branch_by_shape.items() if s[1] == hand_size)
        print(
            f"{hand_size:>3} {sec:>7.2f} {100 * sec / max(total_sec, 1e-9):>5.1f}% "
            f"{pos:>10} {100 * pos / max(total_positions, 1):>5.1f}% "
            f"{branch:>7} {100 * branch / max(total_branch, 1):>5.1f}%"
        )


if __name__ == "__main__":
    main()
