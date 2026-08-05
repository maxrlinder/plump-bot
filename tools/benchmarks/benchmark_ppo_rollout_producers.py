#!/usr/bin/env python3
"""Compare one versus several independent MPS PPO rollout producers.

Each case collects the same total number of games. Workers use ``spawn``, own
model/cache state, disjoint RNG seeds, and a barrier after an untimed warmup so
process startup and Metal graph compilation are excluded from timed throughput.

Example:

    .venv/bin/python tools/benchmarks/benchmark_ppo_rollout_producers.py \
        --total-games-per-shape 64 --producers 1,2,4
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from plump.run_config import load_training_config
from plump.seq.model import SeqPlumpModel
from plump.seq.rollout import SeqRolloutCollector


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=Path("configs/ppo-mps.toml"))
    result.add_argument("--device", default="mps")
    result.add_argument("--precision", choices=("fp32", "fp16", "bf16"))
    result.add_argument("--kv-dtype", choices=("fp32", "fp16", "bf16"))
    result.add_argument("--total-games-per-shape", type=int, default=64)
    result.add_argument("--producers", default="1,2,4")
    result.add_argument("--repeats", type=int, default=1)
    result.add_argument("--seed", type=int, default=20260805)
    return result


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _allocated(device: torch.device) -> int:
    if device.type == "mps":
        return int(torch.mps.driver_allocated_memory())
    if device.type == "cuda":
        return int(torch.cuda.memory_allocated(device))
    return 0


def _worker(
    worker: int,
    config: str,
    device_name: str,
    precision: str | None,
    kv_dtype: str | None,
    games_per_shape: int,
    repeats: int,
    seed: int,
    barrier: mp.synchronize.Barrier,
    output: mp.Queue,
) -> None:
    try:
        overrides = [f"training.deals_per_shape={games_per_shape}"]
        if precision is not None:
            overrides.append(f'training.precision="{precision}"')
        if kv_dtype is not None:
            overrides.append(f'rollout.kv_dtype="{kv_dtype}"')
        resolved = load_training_config(Path(config), overrides=overrides)
        if resolved.training.policy_objective != "ppo":
            raise ValueError("producer benchmark requires policy_objective='ppo'")
        device = torch.device(device_name)
        worker_seed = seed + 1_000_003 * worker
        torch.manual_seed(worker_seed)
        np.random.seed(worker_seed)
        model = SeqPlumpModel(resolved.model).eval().to(device)
        collector = SeqRolloutCollector(model, resolved.training, device=device)

        # Compile every shape and establish steady cache capacities before the
        # parent starts the clock.
        collector.collect(
            None,
            random.Random(worker_seed),
            iteration=0,
            opponent_phase="heuristic",
        )
        _sync(device)
        barrier.wait()

        started = time.perf_counter()
        games = 0
        forward_rows = 0
        sample_sec = 0.0
        step_sec = 0.0
        token_build_sec = 0.0
        forward_sec = 0.0
        for repeat in range(repeats):
            trees = collector.collect(
                None,
                random.Random(worker_seed + repeat + 1),
                iteration=repeat + 1,
                opponent_phase="heuristic",
            )
            games += len(trees)
            forward_rows += collector.stats.forward_rows
            sample_sec += collector.stats.sample_sec
            step_sec += collector.stats.step_sec
            token_build_sec += collector.stats.token_build_sec
            forward_sec += collector.stats.forward_sec
        _sync(device)
        elapsed = time.perf_counter() - started
        output.put(
            {
                "worker": worker,
                "games": games,
                "forward_rows": forward_rows,
                "sample_sec": sample_sec,
                "step_sec": step_sec,
                "token_build_sec": token_build_sec,
                "forward_sec": forward_sec,
                "elapsed_sec": elapsed,
                "allocated_gb": _allocated(device) / 1e9,
            }
        )
    except BaseException as error:
        output.put(
            {
                "worker": worker,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        try:
            barrier.abort()
        except BaseException:
            pass


def _run_case(args: argparse.Namespace, producers: int) -> dict:
    if args.total_games_per_shape % producers:
        raise ValueError(
            "total games per shape must be divisible by every producer count"
        )
    games_per_shape = args.total_games_per_shape // producers
    context = mp.get_context("spawn")
    barrier = context.Barrier(producers + 1)
    output = context.Queue()
    processes = [
        context.Process(
            target=_worker,
            args=(
                worker,
                str(args.config),
                args.device,
                args.precision,
                args.kv_dtype,
                games_per_shape,
                args.repeats,
                args.seed,
                barrier,
                output,
            ),
        )
        for worker in range(producers)
    ]
    for process in processes:
        process.start()
    try:
        barrier.wait()
    except BaseException:
        pass
    started = time.perf_counter()
    rows = [output.get() for _ in processes]
    for process in processes:
        process.join()
    wall = time.perf_counter() - started
    failures = [row for row in rows if "error" in row]
    if failures:
        return {"producers": producers, "failures": failures}
    games = sum(int(row["games"]) for row in rows)
    return {
        "producers": producers,
        "games_per_shape_per_producer": games_per_shape,
        "games": games,
        "wall_sec": wall,
        "aggregate_games_per_sec": games / wall,
        "max_worker_sec": max(float(row["elapsed_sec"]) for row in rows),
        "sum_allocated_gb": sum(float(row["allocated_gb"]) for row in rows),
        "workers": sorted(rows, key=lambda row: int(row["worker"])),
    }


def main() -> int:
    args = parser().parse_args()
    producer_counts = [int(value) for value in args.producers.split(",")]
    results = []
    for producers in producer_counts:
        row = _run_case(args, producers)
        results.append(row)
        print(json.dumps(row), flush=True)
        if "failures" in row:
            return 1
    baseline = results[0]["aggregate_games_per_sec"]
    for row in results:
        row["throughput_vs_first"] = row["aggregate_games_per_sec"] / baseline
    print(json.dumps({"summary": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
