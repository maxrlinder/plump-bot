#!/usr/bin/env python3
"""Measure how the PPO gradient's signal and noise scale with games per update.

A minibatch gradient is the true gradient plus sampling noise, and the two
scale differently with the batch: the signal is fixed while the noise averages
down as 1/B. So the *squared* norm of the measured gradient is

    E|g_B|^2  =  |G|^2  +  tr(Sigma) / B

which is a straight line in 1/B. Fitting it separates the two and gives the
noise scale B_simple = tr(Sigma) / |G|^2 -- the batch size at which noise and
signal contribute equally, and beyond which more games per update buy
progressively less per unit of compute.

Both halves of the actor objective are measured, because the PPO half and the
auxiliary half need not be equally noisy. If they are not, then the observed
ratio between them -- which is what ppo_auxiliary_grad_fraction rebalances --
is itself a function of the batch size rather than a property of the objective.

The model is held fixed throughout: every sub-batch is evaluated at the same
weights and no optimizer step is ever taken, so the spread across sub-batches
is sampling noise and nothing else.

Example:

    .venv/bin/python tools/benchmarks/benchmark_gradient_noise.py \
        --checkpoint runs/ppo-oracle-mps-768-v4-5/checkpoints/iter_048300.pt \
        --pools 10
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from plump.run_config import load_training_config
from plump.seq.model import SeqPlumpModel
from plump.seq.policy import best_seq_device
from plump.seq.ppo import build_ppo_training_batch, normalize_ppo_advantages
from plump.seq.trainer import SeqTrainer, SeqUpdateStats


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--config", type=Path, default=Path("configs/ppo-mps.toml"))
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--device", default="auto")
    result.add_argument(
        "--pools",
        type=int,
        default=10,
        help="independent collections; each is re-split at every batch size",
    )
    result.add_argument(
        "--deals-per-shape",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32],
        help=(
            "batch sizes to measure, in deals per (players, cards) cell -- the "
            "knob a real run would turn. Games per update is 24x this."
        ),
    )
    result.add_argument(
        "--collections-per-pool",
        type=int,
        default=1,
        help=(
            "concatenate this many collections into one pool, to reach batch "
            "sizes above what a single schedule provides"
        ),
    )
    result.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
    )
    result.add_argument("--seed", type=int, default=20260824)
    result.add_argument("--json", type=Path)
    return result


def stratified_chunks(
    trees: list, per_shape: int, rng: random.Random
) -> list[list]:
    """Split a pool into disjoint sub-batches of ``per_shape`` deals per cell.

    Stratified rather than uniformly random because that is what shrinking the
    schedule actually does: `deals_per_shape` is per cell, so a smaller update
    keeps the same shape mix. Sampling games uniformly instead would fold the
    mix's own variance into the measurement and inflate the noise.
    """

    by_shape: dict[tuple[int, int], list] = defaultdict(list)
    for tree in trees:
        by_shape[(tree.num_players, tree.hand_size)].append(tree)
    for group in by_shape.values():
        rng.shuffle(group)
    count = min(len(group) // per_shape for group in by_shape.values())
    return [
        [
            tree
            for group in by_shape.values()
            for tree in group[index * per_shape : (index + 1) * per_shape]
        ]
        for index in range(count)
    ]


def measure(trainer: SeqTrainer, trees: list) -> dict[str, float]:
    """Both gradient halves for one sub-batch, at unchanged weights."""

    batch = build_ppo_training_batch(trees, trainer.model.config, trainer.train)
    groups = batch.policy_groups
    stats = SeqUpdateStats()
    _, advantage_std = trainer._prepare_ppo_advantages(
        groups, batch.critic_groups, stats
    )
    if trainer.train.ppo_advantage_normalize:
        normalize_ppo_advantages(groups)
    for model in trainer.models:
        model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer._auxiliary_grads = None
    trainer._policy_grads = None
    trainer._ppo_actor_backward(groups)
    policy = trainer._stash_norm(trainer._policy_grads)
    auxiliary = trainer._stash_norm(trainer._auxiliary_grads)
    trainer._auxiliary_grads = None
    trainer._policy_grads = None
    trainer.optimizer.zero_grad(set_to_none=True)
    return {
        "policy": policy,
        "auxiliary": auxiliary,
        "advantage_std": advantage_std,
        "rows": sum(
            len(rows.weight) for group in groups for rows in group.policy.values()
        ),
    }


def fit(sizes: list[int], mean_square: list[float]) -> tuple[float, float]:
    """Least squares of E|g|^2 against 1/B; returns (|G|^2, tr(Sigma))."""

    x = [1.0 / size for size in sizes]
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(mean_square) / n
    denominator = sum((xi - mean_x) ** 2 for xi in x)
    slope = (
        sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, mean_square))
        / denominator
        if denominator > 0
        else 0.0
    )
    return mean_y - slope * mean_x, slope


def main() -> int:
    args = parser().parse_args()
    resolved = load_training_config(args.config, overrides=args.overrides)
    if resolved.training.policy_objective != "ppo":
        raise ValueError("This benchmark requires policy_objective='ppo'.")
    if resolved.training.ppo_auxiliary_grad_fraction <= 0:
        raise ValueError(
            "Set training.ppo_auxiliary_grad_fraction > 0: the two halves are "
            "only separable when the split backward runs."
        )
    device = (
        best_seq_device()
        if args.device in ("", "auto")
        else torch.device(args.device)
    )
    torch.manual_seed(resolved.training.seed)
    np.random.seed(resolved.training.seed)
    trainer = SeqTrainer(
        SeqPlumpModel(resolved.model), resolved.training, device=device
    )
    if args.checkpoint:
        trainer.load_checkpoint(
            args.checkpoint.expanduser(), allow_training_config_mismatch=True
        )
    cells = len(resolved.training.schedule_cells)
    rng = random.Random(args.seed)
    samples: dict[int, list[dict[str, float]]] = defaultdict(list)
    started = time.perf_counter()
    for pool in range(args.pools):
        trees = []
        for part in range(args.collections_per_pool):
            trees.extend(
                trainer.collector.collect(
                    None,
                    random.Random(
                        args.seed + pool * args.collections_per_pool + part
                    ),
                )
            )
            trainer.collector.release_caches()
        for per_shape in sorted(args.deals_per_shape):
            for chunk in stratified_chunks(trees, per_shape, rng):
                samples[per_shape * cells].append(measure(trainer, chunk))
        print(
            f"pool {pool + 1}/{args.pools} done "
            f"({time.perf_counter() - started:.0f}s)",
            flush=True,
        )

    sizes = sorted(samples)
    print()
    header = (
        f"{'games':>7} {'n':>4} {'|g_ppo|':>9} {'E|g_ppo|^2':>11} "
        f"{'|g_aux|':>9} {'E|g_aux|^2':>11} {'ppo/aux':>8} {'adv_std':>8}"
    )
    print(header)
    print("-" * len(header))
    report: dict[str, object] = {"sizes": {}}
    for size in sizes:
        values = samples[size]
        policy = [v["policy"] for v in values]
        auxiliary = [v["auxiliary"] for v in values]
        row = {
            "n": len(values),
            "policy_mean": statistics.mean(policy),
            "policy_mean_square": statistics.mean(p * p for p in policy),
            "auxiliary_mean": statistics.mean(auxiliary),
            "auxiliary_mean_square": statistics.mean(a * a for a in auxiliary),
            "ratio": statistics.mean(p / a for p, a in zip(policy, auxiliary)),
            "advantage_std": statistics.mean(v["advantage_std"] for v in values),
            "rows": statistics.mean(v["rows"] for v in values),
        }
        report["sizes"][size] = row
        print(
            f"{size:>7} {row['n']:>4} {row['policy_mean']:>9.4f} "
            f"{row['policy_mean_square']:>11.4f} {row['auxiliary_mean']:>9.4f} "
            f"{row['auxiliary_mean_square']:>11.5f} {row['ratio']:>8.2f} "
            f"{row['advantage_std']:>8.4f}"
        )

    print()
    for half in ("policy", "auxiliary"):
        squares = [report["sizes"][size][f"{half}_mean_square"] for size in sizes]
        signal_square, trace = fit(sizes, squares)
        report[half] = {"signal_square": signal_square, "trace": trace}
        if signal_square <= 0:
            print(f"{half:>10}: no positive signal term; the fit is noise-only.")
            continue
        b_simple = trace / signal_square
        report[half]["b_simple"] = b_simple
        largest = sizes[-1]
        noise_fraction = (trace / largest) / (signal_square + trace / largest)
        report[half]["noise_fraction_at_largest"] = noise_fraction
        print(
            f"{half:>10}: |G| = {math.sqrt(signal_square):.4f}   "
            f"tr(Sigma) = {trace:.1f}   B_simple = {b_simple:.0f} games   "
            f"noise is {100 * noise_fraction:.1f}% of |g|^2 at {largest} games"
        )

    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
