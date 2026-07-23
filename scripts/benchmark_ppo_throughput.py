"""Checkpoint-backed PPO throughput benchmark with no training side effects.

The script intentionally bypasses evaluation, diagnostics, plotting, league
payoff refreshes, and checkpoint writes. Each process loads the same schema-v4
checkpoint (including optimizer and league state), performs optional warm-up
cycles, and reports synchronized collect/update wall times plus MPS memory.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import time
from dataclasses import asdict, fields
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plump.modeling import ModelConfig
from plump.modeling.torch_model import PlumpTransformerModel
from plump.training import PPOTrainer, TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--mode",
        choices=("baseline", "optimized", "recursive"),
        required=True,
    )
    parser.add_argument(
        "--phase",
        choices=("full", "collect"),
        default="full",
    )
    parser.add_argument("--packing", choices=("torch", "numpy"), default="numpy")
    parser.add_argument("--rounds-per-configuration", type=int, default=16)
    parser.add_argument("--num-envs", type=int, default=384)
    parser.add_argument("--env-workers", type=int, default=0)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=1440)
    parser.add_argument("--microbatch-size", type=int, default=480)
    parser.add_argument("--branch-decision-budget-per-arm", type=int, default=30_000)
    parser.add_argument("--branch-update-decisions-per-arm", type=int, default=2_400)
    parser.add_argument("--branch-max-active", type=int, default=768)
    parser.add_argument(
        "--branch-policy-objective",
        choices=("ppo", "neurd"),
        default="neurd",
    )
    parser.add_argument("--branch-neurd-regret-coef", type=float, default=0.25)
    parser.add_argument("--branch-neurd-kl-coef", type=float, default=1.0)
    parser.add_argument(
        "--branch-tree-decision-budget-per-arm",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--branch-tree-update-decisions-per-arm",
        type=int,
        default=800,
    )
    parser.add_argument("--self-play-fraction", type=float)
    parser.add_argument("--heuristic-fraction", type=float)
    parser.add_argument("--mixed-fraction", type=float)
    parser.add_argument("--historical-fraction", type=float)
    parser.add_argument("--explore-self-fraction", type=float)
    parser.add_argument("--explore-historical-fraction", type=float)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measured", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--memory-watermark", type=float, default=0.95)
    parser.add_argument("--require-memory-watermark", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


class MemoryMonitor:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.peak_current = 0
        self.peak_driver = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "MemoryMonitor":
        self.sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.sample()

    def _run(self) -> None:
        while not self._stop.wait(0.05):
            self.sample()

    def sample(self) -> None:
        if self.device.type == "mps":
            self.peak_current = max(
                self.peak_current,
                int(torch.mps.current_allocated_memory()),
            )
            self.peak_driver = max(
                self.peak_driver,
                int(torch.mps.driver_allocated_memory()),
            )
        elif self.device.type == "cuda":
            self.peak_current = max(
                self.peak_current,
                int(torch.cuda.max_memory_allocated(self.device)),
            )
            self.peak_driver = self.peak_current


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def checkpoint_config(
    checkpoint: Path,
    *,
    args: argparse.Namespace,
) -> tuple[ModelConfig, TrainingConfig]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = ModelConfig(**payload["model_config"])
    stored = payload.get("training_config", {})
    defaults = TrainingConfig(model_config=model_config)
    values = {
        field.name: stored.get(field.name, getattr(defaults, field.name))
        for field in fields(TrainingConfig)
        if field.name != "model_config"
    }
    optimized = args.mode in {"optimized", "recursive"}
    values.update(
        {
            "player_counts": (3, 4, 5),
            "hand_sizes": tuple(range(3, 11)),
            "rounds_per_configuration": args.rounds_per_configuration,
            "num_envs": args.num_envs,
            "ppo_epochs": args.ppo_epochs,
            "target_kl": 0.02,
            "pipeline_rollouts": False,
            "env_workers": args.env_workers,
            "event_length_buckets": (8, 16, 32, 64) if optimized else (),
            "batch_packing": args.packing if optimized else "torch",
            "lean_rollout_forward": optimized,
            "batched_league_sampling": optimized,
            "league_probe_fraction": 0.10,
            "minibatch_size": args.minibatch_size,
            "microbatch_size": args.microbatch_size,
            "league_eval_every": 0,
            "seed": args.seed,
            "device": args.device,
            "model_config": model_config,
        }
    )
    for name in (
        "self_play_fraction",
        "heuristic_fraction",
        "mixed_fraction",
        "historical_fraction",
        "explore_self_fraction",
        "explore_historical_fraction",
    ):
        override = getattr(args, name)
        if override is not None:
            values[name] = override
    if args.mode == "recursive":
        values.update(
            {
                "self_play_fraction": 0.5,
                "heuristic_fraction": 0.0,
                "mixed_fraction": 0.0,
                "historical_fraction": 0.5,
                "explore_self_fraction": 0.0,
                "explore_historical_fraction": 0.0,
                "tempered_self_fraction": 0.0,
                "tempered_historical_fraction": 0.0,
                "capped_self_fraction": 0.0,
                "capped_historical_fraction": 0.0,
                "epsilon_self_fraction": 0.0,
                "epsilon_historical_fraction": 0.0,
                "explore_eps_bid": 0.0,
                "explore_eps_play": 0.0,
                "explore_eps_by_arm": {},
                "explore_temperature_fraction": 0.0,
                "explore_uniform_round_probability": 0.0,
                "entropy_coef": 0.0,
                "mmd_enabled": False,
                "branch_rollouts": True,
                "branch_decision_budget_per_arm": (
                    args.branch_decision_budget_per_arm
                ),
                "branch_update_decision_budget_per_arm": (
                    args.branch_update_decisions_per_arm
                ),
                "branch_max_active": args.branch_max_active,
                "branch_bid_max_actions": 4,
                "branch_support_floor": 0.0,
                "branch_target_temperature": 1.0,
                "branch_advantage_clip": 4.0,
                "branch_policy_coef": 1.0,
                "branch_policy_objective": args.branch_policy_objective,
                "branch_neurd_regret_coef": args.branch_neurd_regret_coef,
                "branch_neurd_kl_coef": args.branch_neurd_kl_coef,
                "branch_kl_cap": 0.005,
                "branch_tree_decision_budget_per_arm": (
                    args.branch_tree_decision_budget_per_arm
                ),
                "branch_tree_update_decisions_per_arm": (
                    args.branch_tree_update_decisions_per_arm
                ),
            }
        )
    del payload
    return model_config, TrainingConfig(**values)


def finite_update(update) -> bool:
    return all(
        math.isfinite(float(value))
        for value in asdict(update).values()
        if isinstance(value, float)
    )


def main() -> None:
    args = parse_args()
    model_config, training_config = checkpoint_config(args.checkpoint, args=args)
    trainer = PPOTrainer(
        PlumpTransformerModel(model_config),
        training_config,
    )
    if args.env_workers > 0:
        from plump.training.env_workers import EnvWorkerPool

        trainer.env_pool = EnvWorkerPool(
            num_workers=args.env_workers,
            model_config=model_config,
            include_game_context=training_config.include_game_context,
            num_envs=training_config.num_envs,
        )
    resume = trainer.load_checkpoint(args.checkpoint, load_optimizer=True)
    if not resume["optimizer_loaded"]:
        raise RuntimeError("Optimizer state did not resume.")
    if resume["league_snapshots_missing"]:
        raise RuntimeError(
            f"Missing league snapshots: {resume['league_snapshots_missing']}"
        )

    device = trainer.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total_cycles = args.warmup + args.measured
    measured_rows = []
    all_rows = []
    with MemoryMonitor(device) as memory:
        for cycle in range(total_cycles):
            iteration = int(resume["iteration"] or 0) + cycle + 1
            synchronize(device)
            started = time.perf_counter()
            buffer = trainer.collect_rollouts(iteration=iteration)
            collect_sec = trainer.last_collect_sec
            update = (
                trainer.update(buffer)
                if args.phase == "full"
                else None
            )
            synchronize(device)
            wall_sec = time.perf_counter() - started
            memory.sample()
            row = {
                "cycle": cycle + 1,
                "iteration": iteration,
                "warmup": cycle < args.warmup,
                "wall_sec": wall_sec,
                "collect_sec": collect_sec,
                "update_sec": wall_sec - collect_sec,
                "rollout_rounds": len(buffer.round_outcomes),
                "samples": len(buffer.ready_samples()),
                "finite": (
                    finite_update(update)
                    if update is not None
                    else True
                ),
                "skipped_steps": (
                    update.skipped_steps
                    if update is not None
                    else 0
                ),
                "epochs_run": (
                    update.epochs_run
                    if update is not None
                    else 0
                ),
                "approx_kl": (
                    update.approx_kl
                    if update is not None
                    else 0.0
                ),
                "branch_kl": (
                    update.branch_kl
                    if update is not None
                    else 0.0
                ),
                "branch_policy_loss": (
                    update.branch_policy_loss
                    if update is not None
                    else 0.0
                ),
                "branch_samples": (
                    update.branch_samples
                    if update is not None
                    else 0
                ),
                "total_loss": (
                    update.total_loss
                    if update is not None
                    else 0.0
                ),
                "collection": asdict(trainer.last_collection_stats),
                "peak_current_memory_bytes": memory.peak_current,
                "peak_driver_memory_bytes": memory.peak_driver,
            }
            if (
                not training_config.branch_rollouts
                and row["rollout_rounds"] != training_config.rounds_per_batch
            ):
                raise RuntimeError(
                    "Rollout count changed: "
                    f"{row['rollout_rounds']} != {training_config.rounds_per_batch}"
                )
            if not row["finite"] or row["skipped_steps"] != 0:
                raise RuntimeError(f"Invalid update: {row}")
            all_rows.append(row)
            if not row["warmup"]:
                measured_rows.append(row)
            print(json.dumps({"type": "cycle", **row}, sort_keys=True), flush=True)

    recommended = (
        int(torch.mps.recommended_max_memory())
        if device.type == "mps"
        else 0
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "resume": resume,
        "mode": args.mode,
        "phase": args.phase,
        "packing": training_config.batch_packing,
        "microbatch_size": args.microbatch_size,
        "warmup_cycles": args.warmup,
        "measured_cycles": args.measured,
        "median_wall_sec": statistics.median(
            row["wall_sec"] for row in measured_rows
        ),
        "median_collect_sec": statistics.median(
            row["collect_sec"] for row in measured_rows
        ),
        "median_update_sec": statistics.median(
            row["update_sec"] for row in measured_rows
        ),
        "peak_current_memory_bytes": memory.peak_current,
        "peak_driver_memory_bytes": memory.peak_driver,
        "recommended_max_memory_bytes": recommended,
        "memory_watermark": args.memory_watermark,
        "under_memory_watermark": (
            not recommended
            or memory.peak_driver <= args.memory_watermark * recommended
        ),
        "rows": all_rows,
    }
    print(json.dumps({"type": "result", **result}, sort_keys=True), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if trainer.env_pool is not None:
        trainer.env_pool.close()
    if args.require_memory_watermark and not result["under_memory_watermark"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
