#!/usr/bin/env python3
"""Measure branch-free PPO rollout/update throughput and accelerator memory.

Example (small MPS smoke):

    .venv/bin/python tools/benchmarks/benchmark_ppo.py \
        --games-per-shape 2 --warmup 1 --repeats 2

The benchmark uses the production collector, PPO actor loss, oracle critic,
and the configured compute/cache dtypes. It never creates a run or checkpoint.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from plump.run_config import load_training_config
from plump.seq.model import SeqPlumpModel
from plump.seq.policy import best_seq_device
from plump.seq.trainer import SeqTrainer


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--config", type=Path, default=Path("configs/ppo-mps.toml"))
    result.add_argument("--device", default="auto")
    result.add_argument("--precision", choices=("fp32", "fp16", "bf16"))
    result.add_argument("--kv-dtype", choices=("fp32", "fp16", "bf16"))
    result.add_argument("--games-per-shape", type=int, default=2)
    result.add_argument("--microbatch-positions", type=int)
    result.add_argument("--bucket-width", type=int)
    result.add_argument("--critic-bucket-width", type=int)
    result.add_argument("--policies", type=int)
    result.add_argument("--warmup", type=int, default=1)
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--checkpoint", type=Path)
    result.add_argument(
        "--profile-stages",
        action="store_true",
        help=(
            "synchronize around the major PPO update stages and report their "
            "wall times (diagnostic only; synchronization lowers throughput)"
        ),
    )
    result.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
    )
    return result


def allocated_bytes(device: torch.device) -> int:
    if device.type == "mps":
        return int(torch.mps.driver_allocated_memory())
    if device.type == "cuda":
        return int(torch.cuda.memory_allocated(device))
    return 0


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> int:
    args = parser().parse_args()
    overrides = [
        f"training.deals_per_shape={args.games_per_shape}",
        *args.overrides,
    ]
    if args.precision:
        overrides.append(f'training.precision="{args.precision}"')
    if args.kv_dtype:
        overrides.append(f'rollout.kv_dtype="{args.kv_dtype}"')
    if args.microbatch_positions:
        overrides.append(
            f"training.microbatch_positions={args.microbatch_positions}"
        )
    if args.bucket_width is not None:
        overrides.append(
            f"training.ppo_sequence_bucket_width={args.bucket_width}"
        )
    if args.critic_bucket_width is not None:
        overrides.append(
            "training.ppo_critic_sequence_bucket_width="
            f"{args.critic_bucket_width}"
        )
    if args.policies:
        overrides.append(f"training.ppo_trainable_policies={args.policies}")
    resolved = load_training_config(args.config, overrides=overrides)
    if resolved.training.policy_objective != "ppo":
        raise ValueError("benchmark_ppo requires policy_objective='ppo'.")
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

    stage_times: dict[str, float] = {}
    if args.profile_stages:
        for stage, method_name in (
            ("advantage", "_prepare_ppo_advantages"),
            ("actor_backward", "_ppo_actor_backward"),
            ("actor_kl", "_evaluate_ppo_kl"),
            ("critic", "_train_ppo_critic"),
        ):
            original = getattr(trainer, method_name)

            def timed(*call_args, _stage=stage, _original=original, **call_kwargs):
                synchronize(device)
                stage_started = time.perf_counter()
                value = _original(*call_args, **call_kwargs)
                synchronize(device)
                stage_times[_stage] = stage_times.get(_stage, 0.0) + (
                    time.perf_counter() - stage_started
                )
                return value

            setattr(trainer, method_name, timed)

    samples = []
    total_iterations = args.warmup + args.repeats
    for index in range(total_iterations):
        trainer.iteration += 1
        stage_times.clear()
        synchronize(device)
        baseline = allocated_bytes(device)
        started = time.perf_counter()
        trees, summary = trainer.collect()
        synchronize(device)
        collected = time.perf_counter()
        rollout_peak = trainer.collector.stats.peak_device_bytes
        stats = trainer.update(trees)
        synchronize(device)
        finished = time.perf_counter()
        if index < args.warmup:
            continue
        games = len(trees)
        sample = {
            "games": games,
            "policy_rows": stats.policy_rows,
            "collect_sec": collected - started,
            "update_sec": finished - collected,
            "build_sec": stats.build_sec,
            "total_sec": finished - started,
            "games_per_sec": games / max(finished - started, 1e-9),
            "policy_rows_per_sec": stats.policy_rows
            / max(finished - started, 1e-9),
            "forward_rows": trainer.collector.stats.forward_rows,
            "forward_sec": trainer.collector.stats.forward_sec,
            "rollout_peak_gb": rollout_peak / 1e9,
            "update_peak_gb": stats.peak_update_device_bytes / 1e9,
            "baseline_gb": baseline / 1e9,
            "policy_kl_checked": stats.policy_kl_checked,
            "policy_kl": stats.policy_kl if stats.policy_kl_checked else None,
            "behavior_replay_kl": stats.ppo_behavior_replay_kl,
        }
        if args.profile_stages:
            sample["update_stage_sec"] = {
                **stage_times,
                "build": stats.build_sec,
                "other": max(
                    (finished - collected)
                    - stats.build_sec
                    - sum(stage_times.values()),
                    0.0,
                ),
            }
        samples.append(sample)
        print(json.dumps({"iteration": index - args.warmup + 1, **sample}))

    summary = {
        "device": str(device),
        "precision": resolved.training.precision,
        "kv_dtype": resolved.training.kv_dtype,
        "trainable_policies": resolved.training.ppo_trainable_policies,
        "games_per_update": samples[0]["games"] if samples else 0,
        "mean_collect_sec": statistics.mean(s["collect_sec"] for s in samples),
        "mean_update_sec": statistics.mean(s["update_sec"] for s in samples),
        "mean_total_sec": statistics.mean(s["total_sec"] for s in samples),
        "mean_games_per_sec": statistics.mean(
            s["games_per_sec"] for s in samples
        ),
        "max_rollout_peak_gb": max(s["rollout_peak_gb"] for s in samples),
        "max_update_peak_gb": max(s["update_peak_gb"] for s in samples),
        # ru_maxrss is bytes on macOS and KiB on Linux.
        "process_max_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    print(json.dumps({"summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
