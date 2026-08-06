#!/usr/bin/env python3
"""Measure steady PPO collection against a resident historical league."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from plump.run_config import load_training_config
from plump.seq.model import SeqPlumpModel
from plump.seq.policy import best_seq_device
from plump.seq.trainer import SeqTrainer


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ppo-mps.toml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--games-per-shape", type=int, default=4)
    parser.add_argument("--pool-size", type=int, default=5)
    parser.add_argument(
        "--packing", choices=("concurrent", "sequential"), default="concurrent"
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    resolved = load_training_config(
        args.config,
        overrides=[
            f"training.deals_per_shape={args.games_per_shape}",
            f"training.league_pool_size={args.pool_size}",
            f'rollout.opponent_packing="{args.packing}"',
        ],
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
    trainer.load_checkpoint(
        args.checkpoint.expanduser(), allow_training_config_mismatch=True
    )

    samples = []
    for cycle in range(args.warmup + args.repeats):
        trainer.iteration += 1
        synchronize(device)
        started = time.perf_counter()
        trees, _ = trainer.collect()
        synchronize(device)
        elapsed = time.perf_counter() - started
        stats = trainer.collector.stats
        sample = {
            "collect_sec": elapsed,
            "games": len(trees),
            "forward_rows": stats.forward_rows,
            "forward_sec": stats.forward_sec,
            "games_per_sec": len(trees) / elapsed,
            "forward_rows_per_sec": stats.forward_rows / elapsed,
            "peak_device_gb": stats.peak_device_bytes / 1e9,
        }
        label = "warmup" if cycle < args.warmup else f"repeat_{cycle}"
        print(json.dumps({label: sample}), flush=True)
        if cycle >= args.warmup:
            samples.append(sample)
        del trees
        gc.collect()

    print(
        json.dumps(
            {
                "summary": {
                    "packing": args.packing,
                    "pool_size": args.pool_size,
                    "games": samples[0]["games"],
                    "mean_collect_sec": statistics.mean(
                        sample["collect_sec"] for sample in samples
                    ),
                    "mean_games_per_sec": statistics.mean(
                        sample["games_per_sec"] for sample in samples
                    ),
                    "mean_forward_rows_per_sec": statistics.mean(
                        sample["forward_rows_per_sec"] for sample in samples
                    ),
                    "max_peak_device_gb": max(
                        sample["peak_device_gb"] for sample in samples
                    ),
                }
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
