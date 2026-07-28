"""Per-hand-size branching shape for one training update.

Answers, for the configured leaf floor: how many hands we play per update, how
many leaves each produces, how deep branching still reaches, and how much of
the game is left running unbranched to terminal.

    .venv/bin/python tools/benchmarks/report_seq_branch_shape.py --max-leaves-per-game 4096
"""

from __future__ import annotations

import argparse
import random
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument("--hand-sizes", default="3,4,5,6,7,8,9,10")
    parser.add_argument("--players", type=int, default=None, help="fixed count")
    parser.add_argument("--branch-rate", type=float, default=0.5)
    parser.add_argument("--branch-rate-decay", type=float, default=0.0)
    parser.add_argument("--bid-top-k", type=int, default=4)
    parser.add_argument("--play-mode", default="all_legal")
    parser.add_argument("--play-top-k", type=int, default=4)
    parser.add_argument(
        "--games-per-hand",
        type=int,
        default=1,
        help="unique deals per hand size (same seed -> same deals across runs)",
    )
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument(
        "--n-kv-heads",
        type=int,
        default=None,
        help="GQA: KV heads per layer (must divide n_heads). Default = n_heads.",
    )
    parser.add_argument("--d-ff", type=int, default=768)
    parser.add_argument("--kv-dtype", default="fp16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--deals-per-batch", type=int, default=1)
    parser.add_argument(
        "--historical-arm", choices=["off", "paired", "separate"], default="off"
    )
    parser.add_argument("--bid-split-groups", type=int, default=1)
    parser.add_argument("--bid-split-min-hand-size", type=int, default=0)
    parser.add_argument(
        "--cache-budget-gb",
        type=float,
        default=8.0,
        help="memory the KV cache may occupy; rows derive from bytes/row",
    )
    parser.add_argument("--max-cache-rows", type=int, default=None)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="sync per wave to attribute wall time to each stage",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="one tab-separated row instead of the full report (for sweeps)",
    )
    return parser.parse_args()


SUMMARY_FIELDS = (
    "players hands deals split kvheads params_m sec fwd_sec copy_sec "
    "peak_rows cache_gb alloc_gb mem_gb live_gb leaves positions blk_cache"
).split()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else best_seq_device()
    hand_sizes = [int(part) for part in args.hand_sizes.split(",") if part]

    torch.manual_seed(0)
    model = SeqPlumpModel(
        SeqModelConfig(
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            d_ff=args.d_ff,
        )
    ).eval().to(device)

    cells = tuple(
        GameScheduleCell(
            hand_size=n, num_players=args.players, games=args.games_per_hand
        )
        for n in hand_sizes
    )
    train = SeqTrainingConfig(
        schedule_cells=cells,
        branch_rule=BranchRuleConfig(
            bid_top_k=args.bid_top_k,
            play_mode=args.play_mode,
            play_top_k=args.play_top_k,
        ),
        branch_budget=BranchBudgetConfig(
            branch_rate=args.branch_rate,
            branch_rate_decay=args.branch_rate_decay,
        ),
        rollout=RolloutOptions(
            deals_per_batch=args.deals_per_batch,
            historical_arm=args.historical_arm,
            bid_split_groups=args.bid_split_groups,
            bid_split_min_hand_size=args.bid_split_min_hand_size,
            cache_budget_gb=args.cache_budget_gb,
            max_cache_rows=args.max_cache_rows,
        ),
        kv_dtype=args.kv_dtype,
    )
    collector = SeqRolloutCollector(model, train, device=device)
    collector.profile_sync = args.profile

    started = time.perf_counter()
    trees = collector.collect(None, random.Random(args.seed))
    elapsed = time.perf_counter() - started

    stats = collector.stats
    accounted = (
        stats.sample_sec
        + stats.step_sec
        + stats.compact_sec
        + stats.token_build_sec
        + stats.forward_sec
    )
    # driver_allocated is what the process holds from the system (pool
    # included); current_allocated is what live tensors actually occupy.
    memory = stats.peak_device_bytes / 1e9
    live = (
        torch.mps.current_allocated_memory() / 1e9 if device.type == "mps" else 0.0
    )
    params = sum(p.numel() for p in model.parameters())
    if args.summary_only:
        total_leaves = sum(tree.leaf_total for tree in trees)
        total_positions = sum(
            sum(model_config.seq_len(tree.num_players, tree.hand_size) - leaf.owned_from
                for leaf in tree.leaves)
            for tree in trees
        )
        values = [
            trees[0].num_players,
            args.hand_sizes,
            args.games_per_hand,
            args.bid_split_groups,
            model.config.kv_heads,
            f"{params / 1e6:.2f}",
            f"{elapsed:.1f}",
            f"{stats.forward_sec:.1f}",
            f"{stats.compact_sec:.1f}",
            stats.peak_cache_rows,
            f"{stats.peak_cache_rows * stats.bytes_per_row / 1e9:.2f}",
            f"{stats.cache_rows_allocated * stats.bytes_per_row / 1e9:.2f}",
            f"{memory:.1f}",
            f"{live:.1f}",
            total_leaves,
            total_positions,
            stats.blocked_by_cache,
        ]
        print("\t".join(str(value) for value in values))
        return

    print(
        f"\ntime: total {elapsed:.1f}s | gpu-forward {stats.forward_sec:.1f}s "
        f"({100 * stats.forward_sec / max(elapsed, 1e-9):.0f}%) | "
        f"env-step {stats.step_sec:.1f}s | sample {stats.sample_sec:.1f}s | "
        f"token-build {stats.token_build_sec:.1f}s | "
        f"branch-copy {stats.compact_sec:.1f}s | "
        f"other {elapsed - accounted:.1f}s | peak mem {memory:.1f}GB "
        f"(resident after collect {live:.1f}GB)"
    )

    print(
        f"model: {params / 1e6:.2f}M params  d{args.d_model} L{args.n_layers} "
        f"H{args.n_heads} KV{model.config.kv_heads} ff{args.d_ff} "
        f"kv={args.kv_dtype}  device={device}"
    )
    print(
        f"kv cache: {stats.bytes_per_row / 1024:.0f} KB/row | "
        f"peak live rows {stats.peak_cache_rows:,} "
        f"({stats.peak_cache_rows * stats.bytes_per_row / 1e9:.1f}GB) | "
        f"preallocated {stats.cache_rows_allocated:,} rows "
        f"({stats.cache_rows_allocated * stats.bytes_per_row / 1e9:.1f}GB)"
    )
    print(
        f"branch points refused by the cache bound: {stats.blocked_by_cache:,} "
        f"(nonzero means the rate is too high for max_cache_rows)"
    )
    print(
        f"branch_rate={args.branch_rate} decay={args.branch_rate_decay}  "
        f"bid_top_k={args.bid_top_k} play={args.play_mode} "
        f"splits={args.bid_split_groups} deals/batch={args.deals_per_batch}"
    )
    print(f"hands played per update: {len(trees)}  (collect {elapsed:.1f}s)\n")

    by_hand: dict[int, list] = {}
    for tree in trees:
        by_hand.setdefault(tree.hand_size, []).append(tree)

    def mean(values) -> float:
        return sum(values) / len(values) if values else 0.0

    header = (
        f"{'hand':>5} {'deals':>6} {'leaves avg':>11} {'min':>6} {'max':>6} "
        f"{'decis avg':>10} {'brdec avg':>10} {'deepest avg':>12} {'max':>5} "
        f"{'tricks':>7} {'positions':>11}"
    )
    print(header)
    print("-" * len(header))
    total_leaves = total_positions = total_decisions = 0
    for hand_size in sorted(by_hand):
        group = by_hand[hand_size]
        length = model_config.seq_len(group[0].num_players, hand_size)
        leaves = [t.leaf_total for t in group]
        positions = [
            sum(length - leaf.owned_from for leaf in t.leaves) for t in group
        ]
        deepest = [t.deepest_branch_trick for t in group]
        print(
            f"{hand_size:>5} {len(group):>6} {mean(leaves):>11.0f} "
            f"{min(leaves):>6} {max(leaves):>6} "
            f"{mean([t.decision_total for t in group]):>10.0f} "
            f"{mean([t.branch_decisions for t in group]):>10.0f} "
            f"{mean(deepest):>12.1f} {max(deepest):>5} {hand_size:>7} "
            f"{sum(positions):>11,}"
        )
        total_leaves += sum(leaves)
        total_positions += sum(positions)
        total_decisions += sum(t.decision_total for t in group)
    print("-" * len(header))
    print(
        f"{'TOTAL':>5} {len(trees):>6} {total_leaves:>11,} {'':>6} {'':>6} "
        f"{total_decisions:>10,} {'':>10} {'':>12} {'':>5} {'':>7} "
        f"{total_positions:>11,}"
    )
    print(
        f"\ntraining batch per update: {total_positions:,} token positions "
        f"across {total_leaves:,} leaves"
    )

    print(
        "\n\nBranching by depth: mean cumulative leaves after each stage, "
        "averaged over deals\n"
    )
    for hand_size in sorted(by_hand):
        group = by_hand[hand_size]
        max_stage = max((t.deepest_branch_trick for t in group), default=-1)
        print(
            f"  hand {hand_size:>2} ({group[0].num_players}p, {hand_size} tricks, "
            f"{len(group)} deals) -> mean {mean([t.leaf_total for t in group]):.0f} leaves"
        )
        print(
            f"    {'stage':>7} {'br.decis':>9} {'leaves+':>9} "
            f"{'cum.leaves':>11} {'growth':>8}"
        )
        running = [1.0] * len(group)
        for stage in range(-1, max_stage + 1):
            added = [t.leaves_added_by_stage.get(stage, 0) for t in group]
            decis = [t.branch_decisions_by_stage.get(stage, 0) for t in group]
            previous = mean(running)
            running = [r + a for r, a in zip(running, added)]
            if not any(added) and not any(decis):
                continue
            label = "bid" if stage == -1 else f"trick{stage}"
            print(
                f"    {label:>7} {mean(decis):>9.1f} {mean(added):>9.1f} "
                f"{mean(running):>11.0f} {mean(running) / max(previous, 1):>7.2f}x"
            )
        unbranched = [max(t.hand_size - 1 - t.deepest_branch_trick, 0) for t in group]
        print(
            f"    branched through trick {mean([t.deepest_branch_trick for t in group]):.1f} "
            f"on average; {mean(unbranched):.1f} of {hand_size} tricks "
            f"run out unbranched\n"
        )


if __name__ == "__main__":
    main()
