"""Modal L40S sidecar for interruptible, checkpointed Plump training.

The ordinary trainer remains the source of truth. This module only declares
the remote container, CUDA resources, persistent runs volume, and reentrant
entrypoints needed by Modal.
"""

from __future__ import annotations

import csv
import json
import os
import resource
import shutil
import time
from pathlib import Path

import modal


APP_NAME = "plump-l40s-training"
VOLUME_NAME = "plump-training-runs"
DEFAULT_RUN = "stratified-modal-8m"
RUNS_ROOT = Path("/runs")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.13.0",
        "numpy==2.5.1",
        "matplotlib==3.11.1",
    )
    .env(
        {
            "MPLBACKEND": "Agg",
            "PYTHONUNBUFFERED": "1",
            # KV pools grow between waves. Expandable segments reduce the
            # fragmentation left by those increasingly large allocations.
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "CUDA_MODULE_LOADING": "LAZY",
        }
    )
    .add_local_python_source("plump")
)

app = modal.App(APP_NAME, image=image)
runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

gpu_resources = {
    "gpu": "L40S",
    "cpu": 8.0,
    "memory": 64 * 1024,
    "volumes": {RUNS_ROOT: runs_volume},
    "max_containers": 1,
    "single_use_containers": True,
}


def _configure_cuda() -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Modal assigned the function no CUDA device.")
    torch.set_float32_matmul_precision("high")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    details = {
        # torch.__version__ is a TorchVersion subclass; cast it so Modal's
        # client can deserialize the result without importing torch locally.
        "torch": str(torch.__version__),
        "device": str(properties.name),
        "device_bytes": properties.total_memory,
        "cuda": str(torch.version.cuda),
        "matmul_precision": str(torch.get_float32_matmul_precision()),
    }
    print("CUDA " + json.dumps(details, sort_keys=True), flush=True)
    return details


def _remove_container_stale_lock(run_dir: Path) -> None:
    """Clear a lock left by a preempted prior container attempt.

    Modal serializes this Function at one container and gives retries fresh
    containers, so no live trainer can own a lock when this wrapper starts.
    """

    lock = run_dir / "run.lock"
    if lock.exists():
        print(f"Removing prior-container lock {lock}.", flush=True)
        lock.unlink()


def _run_training(run_name: str, *, extra_args: list[str] | None = None) -> int:
    from plump.cli import main as plump_main

    run_dir = Path(os.environ["PLUMP_RUNS_DIR"]) / run_name
    config = run_dir / "config.toml"
    if not config.is_file():
        raise FileNotFoundError(f"Prepared run config is missing: {config}")
    _remove_container_stale_lock(run_dir)
    args = ["train", run_name, "--config", str(config), "--device", "cuda"]
    args.extend(extra_args or [])
    result = plump_main(args)
    if result:
        raise RuntimeError(f"plump train exited with status {result}.")
    return result


@app.function(
    **gpu_resources,
    timeout=24 * 60 * 60,
    startup_timeout=20 * 60,
    retries=modal.Retries(initial_delay=0.0, max_retries=10),
)
def train_interruptible(run_name: str = DEFAULT_RUN) -> dict[str, object]:
    """Resume a prepared run; retries re-enter through its latest manifest."""

    os.environ["PLUMP_RUNS_DIR"] = str(RUNS_ROOT)
    hardware = _configure_cuda()
    try:
        _run_training(run_name)
    finally:
        # Volumes also background-commit every few seconds, but make successful
        # exits and handled exceptions explicit before this container goes away.
        runs_volume.commit()
    latest = json.loads(
        (RUNS_ROOT / run_name / "checkpoints" / "latest.json").read_text()
    )
    return {"run": run_name, "latest": latest, "hardware": hardware}


@app.function(
    **gpu_resources,
    timeout=4 * 60 * 60,
    startup_timeout=20 * 60,
)
def smoke(
    run_name: str = DEFAULT_RUN,
    updates: int = 2,
    wave_deals: int = 4,
    deals_per_shape: int = 8,
    microbatch_positions: int = 131_072,
) -> dict[str, object]:
    """Run a few updates against an ephemeral copy of a prepared Volume run."""

    if updates < 1:
        raise ValueError("updates must be positive.")
    if wave_deals < 1:
        raise ValueError("wave_deals must be positive.")
    if deals_per_shape < 1:
        raise ValueError("deals_per_shape must be positive.")
    if wave_deals > deals_per_shape:
        raise ValueError("wave_deals cannot exceed deals_per_shape.")
    if microbatch_positions < 1:
        raise ValueError("microbatch_positions must be positive.")
    import torch

    hardware = _configure_cuda()
    source = RUNS_ROOT / run_name
    source_latest = json.loads(
        (source / "checkpoints" / "latest.json").read_text()
    )
    start = int(source_latest["iteration"])
    smoke_root = Path("/tmp/plump-smoke-runs")
    smoke_name = f"{run_name}-smoke"
    destination = smoke_root / smoke_name
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    smoke_root.mkdir(parents=True)
    shutil.copytree(source, destination)
    (destination / "run.lock").unlink(missing_ok=True)

    os.environ["PLUMP_RUNS_DIR"] = str(smoke_root)
    extra_args = [
        "--iterations",
        str(start + updates),
        "--reconfigure",
        "--reconfigure-reason",
        f"isolated Modal L40S {wave_deals}-wide smoke test",
    ]
    if deals_per_shape != 8:
        extra_args.extend(
            [
                "--set",
                f"training.deals_per_shape={deals_per_shape}",
            ]
        )
    if wave_deals != 4:
        extra_args.extend(
            ["--set", f"rollout.deals_per_batch={wave_deals}"]
        )
    if microbatch_positions != 131_072:
        extra_args.extend(
            [
                "--set",
                f"training.microbatch_positions={microbatch_positions}",
            ]
        )
    _run_training(smoke_name, extra_args=extra_args)
    with (destination / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    produced = [row for row in rows if int(row["iteration"]) > start]
    result = {
        "run": run_name,
        "wave_deals": wave_deals,
        "deals_per_shape": deals_per_shape,
        "microbatch_positions": microbatch_positions,
        "start_iteration": start,
        "completed_iteration": int(produced[-1]["iteration"]),
        "updates": len(produced),
        "trees": [int(row["trees"]) for row in produced],
        "trees_self": [int(row["trees_self"]) for row in produced],
        "trees_heuristic": [int(row["trees_heuristic"]) for row in produced],
        "positions": [int(row["positions"]) for row in produced],
        "forward_rows": [int(row["forward_rows"]) for row in produced],
        "seconds": [float(row["total_sec"]) for row in produced],
        "collect_seconds": [float(row["collect_sec"]) for row in produced],
        "update_seconds": [float(row["update_sec"]) for row in produced],
        "positions_per_second": [
            float(row["positions_per_sec"]) for row in produced
        ],
        "forward_rows_per_second": [
            float(row["forward_rows_per_sec"]) for row in produced
        ],
        "peak_collection_gb": [
            float(row["peak_device_gb"]) for row in produced
        ],
        "blocked_by_cache": [int(row["blocked_by_cache"]) for row in produced],
        "max_cuda_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "max_cuda_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        "max_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2),
        "hardware": hardware,
    }
    print("SMOKE_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    return result


@app.function(
    **gpu_resources,
    timeout=4 * 60 * 60,
    startup_timeout=20 * 60,
)
def benchmark_rollout_widths(
    run_name: str = DEFAULT_RUN,
    deals_per_shape: int = 8,
    widths: str = "2,4,8",
    profile: bool = False,
) -> dict[str, object]:
    """Compare rollout packing at a fixed trajectory count, without updates."""

    import torch

    from plump.run_config import load_training_config
    from plump.seq.model import SeqPlumpModel
    from plump.seq.trainer import SeqTrainer

    if deals_per_shape < 1:
        raise ValueError("deals_per_shape must be positive.")
    parsed_widths = tuple(int(value) for value in widths.split(","))
    if not parsed_widths or any(value < 1 for value in parsed_widths):
        raise ValueError("widths must be a comma-separated list of positive ints.")
    if any(value > deals_per_shape for value in parsed_widths):
        raise ValueError("rollout widths cannot exceed deals_per_shape.")

    hardware = _configure_cuda()
    source = RUNS_ROOT / run_name
    latest = json.loads((source / "checkpoints" / "latest.json").read_text())
    checkpoint = source / "checkpoints" / latest["path"]
    results = []
    for width in parsed_widths:
        resolved = load_training_config(
            source / "config.toml",
            overrides=[
                f"training.deals_per_shape={deals_per_shape}",
                f"rollout.deals_per_batch={width}",
            ],
        )
        trainer = SeqTrainer(
            SeqPlumpModel(resolved.model),
            resolved.training,
            device="cuda",
        )
        trainer.resolved_config = resolved.raw
        trainer.load_checkpoint(checkpoint, allow_training_config_mismatch=True)
        trainer.iteration += 1
        trainer.collector.profile_sync = True
        profiler = None
        if profile:
            import cProfile

            profiler = cProfile.Profile()
            profiler.enable()
        started = time.perf_counter()
        trees, summary = trainer.collect()
        elapsed = time.perf_counter() - started
        if profiler is not None:
            import io
            import pstats

            profiler.disable()
            profile_stream = io.StringIO()
            pstats.Stats(profiler, stream=profile_stream).sort_stats(
                "cumulative"
            ).print_stats(40)
            profile_text = profile_stream.getvalue()
            print(profile_text, flush=True)
        else:
            profile_text = None
        stats = trainer.collector.stats
        width_result = {
            "wave_deals": width,
            "trees": summary.trees,
            "trees_self": summary.trees_self,
            "trees_heuristic": summary.trees_heuristic,
            "leaves": summary.leaves,
            "seconds": elapsed,
            "sample_seconds": stats.sample_sec,
            "step_seconds": stats.step_sec,
            "compact_seconds": stats.compact_sec,
            "token_build_seconds": stats.token_build_sec,
            "forward_seconds": stats.forward_sec,
            "forward_rows": stats.forward_rows,
            "forward_rows_per_second": stats.forward_rows / max(elapsed, 1e-9),
            "peak_collection_gib": stats.peak_device_bytes / (1024**3),
            "cache_rows_allocated": stats.cache_rows_allocated,
            "blocked_by_cache": stats.blocked_by_cache,
            "profile": profile_text,
        }
        print(
            "ROLLOUT_WIDTH_RESULT "
            + json.dumps(width_result, sort_keys=True),
            flush=True,
        )
        results.append(width_result)
        trainer.collector.release_caches()
        del trees, trainer
        torch.cuda.empty_cache()

    result = {
        "run": run_name,
        "start_iteration": int(latest["iteration"]),
        "deals_per_shape": deals_per_shape,
        "total_trees": deals_per_shape * 24,
        "results": results,
        "hardware": hardware,
        "max_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2),
    }
    print("ROLLOUT_BENCHMARK " + json.dumps(result, sort_keys=True), flush=True)
    return result


@app.local_entrypoint()
def main(run_name: str = DEFAULT_RUN) -> None:
    """Launch with ``modal run --detach infra/modal_training.py``."""

    print(f"Launching interruptible L40S training for {run_name}.", flush=True)
    print(train_interruptible.spawn(run_name).get(), flush=True)
