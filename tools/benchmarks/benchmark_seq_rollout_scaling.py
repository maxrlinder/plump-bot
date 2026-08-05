"""Rollout throughput vs leaf budget, with and without the KV cache.

Runs one fixed deal (default 5 players / 10 cards — the worst case) at each
leaf budget and reports where the wall time goes. Use this to check that work
scales linearly in leaves and that the cache actually pays for itself at the
batch sizes we intend to train at.

    .venv/bin/python tools/benchmarks/benchmark_seq_rollout_scaling.py \
        --budgets 64,128,256,512,1024,2048 --modes cache,nocache
"""

from __future__ import annotations

import argparse
import gc
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
    parser.add_argument("--budgets", default="64,128,256,512,1024,2048")
    parser.add_argument("--modes", default="cache,nocache")
    parser.add_argument("--players", type=int, default=5)
    parser.add_argument("--cards", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=768)
    parser.add_argument(
        "--kv-dtype", choices=["fp32", "fp16", "bf16"], default="fp16"
    )
    # Per-stage timings need syncs, which serialize CPU against GPU and so
    # understate the real overlap. --no-profile-sync reports honest wall time
    # with a meaningless breakdown.
    parser.add_argument(
        "--no-profile-sync", dest="profile_sync", action="store_false"
    )
    parser.set_defaults(profile_sync=True)
    return parser.parse_args()


def device_memory_gb(device: torch.device) -> float:
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def reset_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run_case(
    model: SeqPlumpModel,
    device: torch.device,
    budget: int,
    use_cache: bool,
    args: argparse.Namespace,
) -> dict | None:
    train = SeqTrainingConfig(
        schedule_cells=(
            GameScheduleCell(hand_size=args.cards, num_players=args.players),
        ),
        player_counts=(args.players,),
        player_count_weights=(1.0,),
        branch_rule=BranchRuleConfig(),
        branch_budget=BranchBudgetConfig(branch_rate=budget),
        kv_dtype=args.kv_dtype,
        use_kv_cache=use_cache,
    )
    reset_memory(device)
    collector = SeqRolloutCollector(model, train, device=device)
    collector.profile_sync = args.profile_sync
    best: dict | None = None
    for repeat in range(args.repeats):
        try:
            started = time.perf_counter()
            # Fixed seed for every case: identical deal, seats and bidding
            # order across budgets and model sizes.
            trees = collector.collect(None, random.Random(1234))
            elapsed = time.perf_counter() - started
        except RuntimeError as error:
            print(f"  budget {budget} cache={use_cache}: FAILED ({error})")
            return None
        stats = collector.stats
        tree = trees[0]
        row = {
            "deepest_trick": tree.deepest_branch_trick,
            "branch_layers": tree.branch_layers,
            "branch_decisions": tree.branch_decisions,
            "budget": budget,
            "cache": use_cache,
            "seconds": elapsed,
            "leaves": stats.leaves,
            "decisions": stats.decisions,
            "token_rows": stats.forward_rows,
            "sample": stats.sample_sec,
            "step": stats.step_sec,
            "compact": stats.compact_sec,
            "build": stats.token_build_sec,
            "forward": stats.forward_sec,
            "memory": device_memory_gb(device),
        }
        if best is None or row["seconds"] < best["seconds"]:
            best = row
    del collector
    reset_memory(device)
    return best


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else best_seq_device()
    budgets = [int(part) for part in args.budgets.split(",") if part]
    modes = [part.strip() for part in args.modes.split(",") if part.strip()]

    torch.manual_seed(0)
    model = SeqPlumpModel(
        SeqModelConfig(
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
        )
    ).eval().to(device)

    length = model_config.seq_len(args.players, args.cards)
    params = sum(p.numel() for p in model.parameters())
    print(
        f"device={device} model=d{args.d_model} L{args.n_layers} H{args.n_heads} "
        f"ff{args.d_ff} params={params / 1e6:.2f}M "
        f"game={args.players}p/{args.cards}c seq_len={length} kv={args.kv_dtype}"
    )
    # Warm up MPS/CUDA kernel compilation so the first timed case is not
    # charged for it.
    run_case(model, device, 8, True, argparse.Namespace(**{**vars(args), "repeats": 1}))

    rows: list[dict] = []
    for budget in budgets:
        for mode in modes:
            row = run_case(model, device, budget, mode == "cache", args)
            if row is not None:
                rows.append(row)
                print(
                    f"  done budget={budget:5d} {mode:8s} "
                    f"{row['seconds']:7.2f}s leaves={row['leaves']:5d}",
                    flush=True,
                )

    header = (
        f"{'floor':>7} {'mode':>8} {'leaves':>8} {'decis':>7} {'sec':>8} "
        f"{'dec/s':>8} {'us/leaf':>8} {'layers':>7} {'deepest':>8} "
        f"{'brdecis':>8} {'fwd':>7} {'step':>7} {'copy':>7} {'mem_GB':>7}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        deepest = (
            "bid" if row["deepest_trick"] == -1 else f"trick{row['deepest_trick']}"
        )
        print(
            f"{row['budget']:>7} {'cache' if row['cache'] else 'nocache':>8} "
            f"{row['leaves']:>8} {row['decisions']:>7} "
            f"{row['seconds']:>8.2f} {row['decisions'] / row['seconds']:>8.0f} "
            f"{1e6 * row['seconds'] / max(row['leaves'], 1):>8.0f} "
            f"{row['branch_layers']:>7} {deepest:>8} {row['branch_decisions']:>8} "
            f"{row['forward']:>7.2f} {row['step']:>7.2f} "
            f"{row['compact']:>7.2f} {row['memory']:>7.2f}"
        )

    print("\nScaling (time per leaf, relative to the smallest budget of each mode):")
    for mode in modes:
        mode_rows = [row for row in rows if row["cache"] == (mode == "cache")]
        if not mode_rows:
            continue
        base = mode_rows[0]["seconds"] / max(mode_rows[0]["leaves"], 1)
        parts = [
            f"{row['budget']}:{(row['seconds'] / max(row['leaves'], 1)) / base:.2f}x"
            for row in mode_rows
        ]
        print(f"  {mode:8s} " + "  ".join(parts))


if __name__ == "__main__":
    main()
