#!/usr/bin/env python3
"""Time actor materialization, resident reuse, and full trainer checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from plump.run_config import load_training_config
from plump.seq.config import SeqModelConfig
from plump.seq.model import SeqPlumpModel, load_seq_model_state_dict
from plump.seq.policy import SeqLeague, SeqModelPolicy, best_seq_device
from plump.seq.trainer import SeqTrainer


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def release(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def materialize(
    checkpoint: Path, device: torch.device, *, mmap: bool
) -> dict[str, float]:
    synchronize(device)
    started = time.perf_counter()
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=mmap,
    )
    loaded = time.perf_counter()
    model = SeqPlumpModel(SeqModelConfig(**payload["model_config"]))
    constructed = time.perf_counter()
    load_seq_model_state_dict(model, payload["model_state_dict"])
    state_loaded = time.perf_counter()
    model.to(device).eval()
    synchronize(device)
    transferred = time.perf_counter()
    actor_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in payload["model_state_dict"].values()
    )
    del model, payload
    release(device)
    return {
        "torch_load_sec": loaded - started,
        "construct_sec": constructed - loaded,
        "state_load_sec": state_loaded - constructed,
        "device_transfer_sec": transferred - state_loaded,
        "total_sec": transferred - started,
        "actor_mb": actor_bytes / 1e6,
    }


def resident_lookup(checkpoint: Path, device: torch.device) -> dict[str, float]:
    policy = SeqModelPolicy.from_checkpoint(
        checkpoint, device=device, greedy=True, name="resident"
    )
    league = SeqLeague(max_snapshots=5)
    league.add("resident", str(checkpoint), 0)
    league._policies["resident"] = policy
    rng = __import__("random").Random(0)
    repetitions = 10_000
    started = time.perf_counter()
    for _ in range(repetitions):
        selected = league.draw_pool(rng, 1, iteration=0, device=device)
        if selected[0][1] is not policy:
            raise AssertionError("resident policy cache was not reused")
    elapsed = time.perf_counter() - started
    del league, policy
    release(device)
    return {
        "lookups": repetitions,
        "total_sec": elapsed,
        "microseconds_per_lookup": elapsed * 1e6 / repetitions,
    }


def checkpoint_save(
    config: Path, checkpoint: Path, device: torch.device
) -> dict[str, float]:
    resolved = load_training_config(config)
    trainer = SeqTrainer(
        SeqPlumpModel(resolved.model), resolved.training, device=device
    )
    trainer.load_checkpoint(checkpoint, allow_training_config_mismatch=True)
    synchronize(device)
    with tempfile.NamedTemporaryFile(
        prefix="plump-save-bench-", suffix=".pt", dir="/private/tmp", delete=False
    ) as handle:
        target = Path(handle.name)
    target.unlink()
    try:
        started = time.perf_counter()
        trainer.save_checkpoint(target)
        synchronize(device)
        elapsed = time.perf_counter() - started
        return {"save_sec": elapsed, "saved_mb": target.stat().st_size / 1e6}
    finally:
        target.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/ppo-mps.toml"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    device = best_seq_device() if args.device == "auto" else torch.device(args.device)

    results: dict[str, object] = {
        "checkpoint_mb": checkpoint.stat().st_size / 1e6,
        "device": str(device),
    }
    for mmap in (False, True):
        samples = [
            materialize(checkpoint, device, mmap=mmap)
            for _ in range(args.repeats)
        ]
        results[f"materialize_mmap_{str(mmap).lower()}"] = {
            "samples": samples,
            "mean_total_sec": statistics.mean(row["total_sec"] for row in samples),
            "mean_torch_load_sec": statistics.mean(
                row["torch_load_sec"] for row in samples
            ),
            "mean_device_transfer_sec": statistics.mean(
                row["device_transfer_sec"] for row in samples
            ),
        }
    results["resident_lookup"] = resident_lookup(checkpoint, device)
    results["checkpoint_save"] = checkpoint_save(
        args.config.expanduser().resolve(), checkpoint, device
    )
    print(json.dumps(results, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
