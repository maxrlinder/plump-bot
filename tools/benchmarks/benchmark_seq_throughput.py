"""Benchmark schema-v6 collect+update throughput."""

from __future__ import annotations

import argparse
import statistics
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
    build_branch_rate_table,
    build_game_schedule,
    build_position_balanced_schedule,
)
from plump.seq.model import SeqPlumpModel
from plump.seq.policy import best_seq_device
from plump.seq.trainer import SeqTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-kv-heads", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=768)
    parser.add_argument("--hand-sizes", default="3,4,5,6,7,8,9,10")
    parser.add_argument("--games-per-cell", type=int, default=1)
    parser.add_argument(
        "--games-total",
        type=int,
        default=None,
        help="apportion this many deals over the (players, cards) grid "
        "instead of --games-per-cell per hand size",
    )
    parser.add_argument("--hand-size-tilt", type=float, default=1.0)
    parser.add_argument("--auto-deals", action="store_true")
    parser.add_argument("--microbatch-positions", type=int, default=16384)
    parser.add_argument("--kv-dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--players", type=int, default=None)
    parser.add_argument("--play-mode", default="all_legal")
    parser.add_argument("--play-top-k", type=int, default=4)
    parser.add_argument("--deals-per-batch", type=int, default=1)
    parser.add_argument("--bid-split-groups", type=int, default=1)
    parser.add_argument("--cache-budget-gb", type=float, default=8.0)
    parser.add_argument(
        "--historical-arm", choices=["off", "paired", "separate"], default="paired"
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="one deal per (players, cards, bidding position) per update",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--reference-rate",
        type=float,
        default=None,
        help="derive the per-shape branch rate table from this rate at 10 cards",
    )
    parser.add_argument("--exhaustive-until", type=int, default=7)
    parser.add_argument("--no-trick-win-token", action="store_true")
    parser.add_argument(
        "--turn-token", choices=["off", "bid", "all"], default="off"
    )
    parser.add_argument("--max-cache-rows", type=int, default=None)
    return parser.parse_args()


def peak_memory(device: torch.device) -> float:
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else best_seq_device()
    hand_sizes = tuple(int(part) for part in args.hand_sizes.split(","))
    model_config = SeqModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        d_ff=args.d_ff,
        trick_win_token=not args.no_trick_win_token,
        turn_token=args.turn_token,
    )
    if args.balanced:
        schedule = build_position_balanced_schedule(
            hand_sizes=hand_sizes,
            player_counts=(args.players,) if args.players else (3, 4, 5),
            repeats=args.repeats,
        )
    elif args.games_total is not None:
        schedule = build_game_schedule(
            games_total=args.games_total,
            hand_sizes=hand_sizes,
            player_counts=(args.players,) if args.players else (3, 4, 5),
            hand_size_tilt=args.hand_size_tilt,
        )
    else:
        schedule = tuple(
            GameScheduleCell(
                hand_size=n, num_players=args.players, games=args.games_per_cell
            )
            for n in hand_sizes
        )
    train_config = SeqTrainingConfig(
        schedule_cells=schedule,
        branch_rule=BranchRuleConfig(
            play_mode=args.play_mode, play_top_k=args.play_top_k
        ),
        rollout=RolloutOptions(
            deals_per_batch=args.deals_per_batch,
            auto_deals_per_batch=args.auto_deals,
            historical_arm=args.historical_arm,
            bid_split_groups=args.bid_split_groups,
            cache_budget_gb=args.cache_budget_gb,
            max_cache_rows=args.max_cache_rows,
        ),
        branch_budget=BranchBudgetConfig(
            branch_rate_by_shape=(
                ()
                if args.reference_rate is None
                else build_branch_rate_table(
                    args.reference_rate,
                    exhaustive_until=args.exhaustive_until,
                    hand_sizes=hand_sizes,
                    player_counts=(args.players,) if args.players else (3, 4, 5),
                )
            ),
        ),
        microbatch_positions=args.microbatch_positions,
        kv_dtype=args.kv_dtype,
        epochs=args.epochs,
        snapshot_every=10_000,
    )
    torch.manual_seed(0)
    trainer = SeqTrainer(SeqPlumpModel(model_config), train_config, device=device)
    print(
        f"device={device} dims={args.d_model}x{args.n_layers} "
        f"rate={args.reference_rate}@10c "
        f"rows={args.max_cache_rows} "
        f"deals={sum(cell.games for cell in train_config.schedule_cells)}"
    )

    collect_times, update_times, decision_rates, position_rates = [], [], [], []
    for cycle in range(args.cycles + 1):
        started = time.perf_counter()
        trees, summary = trainer.collect()
        collect_sec = time.perf_counter() - started
        stats = trainer.update(trees)
        label = "warmup" if cycle == 0 else f"cycle {cycle}"
        print(
            f"{label:8s} collect {collect_sec:6.1f}s update {stats.update_sec:6.1f}s | "
            f"leaves {summary.leaves:6d} decisions {summary.decisions:6d} "
            f"positions {stats.positions:7d} | "
            f"{summary.decisions / collect_sec:7.0f} dec/s "
            f"{stats.positions / max(stats.update_sec, 1e-9):8.0f} pos/s | "
            f"mem collect {trainer.collector.stats.peak_device_bytes / 1e9:5.2f}GB "
            f"after-update {peak_memory(device):5.2f}GB"
        )
        if cycle == 0:
            continue
        collect_times.append(collect_sec)
        update_times.append(stats.update_sec)
        decision_rates.append(summary.decisions / collect_sec)
        position_rates.append(stats.positions / max(stats.update_sec, 1e-9))

    print(
        "median: "
        f"collect {statistics.median(collect_times):.1f}s "
        f"update {statistics.median(update_times):.1f}s "
        f"total {statistics.median(collect_times) + statistics.median(update_times):.1f}s | "
        f"{statistics.median(decision_rates):.0f} dec/s "
        f"{statistics.median(position_rates):.0f} pos/s"
    )


if __name__ == "__main__":
    main()
