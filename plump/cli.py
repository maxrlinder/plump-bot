"""Unified command-line interface for schema-v6 Plump workflows."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from plump.dashboard import DEFAULT_SMOOTH_WINDOW, render_dashboard
from plump.evaluation import DealBank, evaluate_policy
from plump.gui.app import run as run_gui
from plump.policies import HeuristicPolicy
from plump.run_evaluation import (
    EvaluationProtocol,
    discover_interval_checkpoints,
    evaluate_checkpoint,
)
from plump.run_config import (
    DEFAULT_CONFIG_PATH,
    config_diff,
    load_training_config,
    resolve_training_config,
)
from plump.runs import RunDirectory
from plump.seq.model import SeqPlumpModel
from plump.seq.policy import SeqModelPolicy, best_seq_device
from plump.seq.trainer import SeqTrainer

METRIC_COLUMNS = (
    "iteration",
    "optimizer_steps",
    "learning_rate",
    "auxiliary_learning_rate",
    "elapsed_sec",
    "total_sec",
    "collect_sec",
    "update_sec",
    "trees",
    "trees_self",
    "trees_heuristic",
    "trees_historical",
    "leaves",
    "decisions",
    "policy_rows",
    "branched_rows",
    "unbranched_rows",
    "positions",
    "forward_rows",
    "branch_decisions",
    "bid_hit_rate",
    "bid_hit_focal",
    "bid_hit_non_focal",
    "reward_focal",
    "reward_non_focal",
    "reward_self",
    "reward_heuristic",
    "reward_historical",
    "opponent_phase",
    "heuristic_eval_win_streak",
    "spine_entropy",
    "loss_policy",
    "loss_value",
    "loss_value_zero",
    "value_rmse",
    "value_zero_rmse",
    "value_correlation",
    "value_prediction_std",
    "value_rows",
    "loss_suit",
    "loss_trick",
    "loss_bid_hit",
    "entropy",
    "entropy_bid_normalized",
    "entropy_play_normalized",
    "entropy_alpha_bid",
    "entropy_alpha_play",
    "ppo_ratio_clip_fraction",
    "ppo_behavior_replay_kl",
    "advantage_mean",
    "advantage_std",
    "policy_logit_shift",
    "policy_kl",
    "policy_kl_p95",
    "policy_kl_p99",
    "policy_kl_max",
    "proposed_policy_kl",
    "proposed_policy_kl_p95",
    "proposed_policy_kl_p99",
    "proposed_policy_kl_max",
    "proposed_mean_exceeded",
    "proposed_p99_exceeded",
    "backtracks",
    "step_scale",
    "rolled_back",
    "core_grad_norm",
    "auxiliary_grad_norm",
    "critic_grad_norm",
    "peak_update_device_gb",
    "eval_reward_vs_heuristic",
    "eval_bid_hit",
    "peak_cache_rows",
    "cache_rows_allocated",
    "cache_pressure",
    "peak_device_gb",
    "blocked_by_cache",
    "skipped_by_placement",
    "sample_sec",
    "step_sec",
    "compact_sec",
    "token_build_sec",
    "forward_sec",
    "positions_per_sec",
    "forward_rows_per_sec",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plump",
        description="Schema-v6 Plump training and inspection tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="create or safely resume a run")
    train.add_argument("run")
    train.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    train.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
    )
    train.add_argument("--iterations", type=int)
    train.add_argument("--device")
    train.add_argument(
        "--from-checkpoint",
        type=Path,
        help="start a new run from a compatible schema-v6 checkpoint",
    )
    train.add_argument(
        "--reset-league",
        action="store_true",
        help=(
            "when forking, omit historical checkpoint references from the new "
            "run so it is self-contained"
        ),
    )
    train.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "create/fork and checkpoint a new run without entering its "
            "training loop"
        ),
    )
    train.add_argument(
        "--reconfigure",
        action="store_true",
        help=(
            "explicitly adopt config/--set changes when resuming an existing "
            "run, writing a resume checkpoint and config audit record first"
        ),
    )
    train.add_argument(
        "--reconfigure-reason",
        default="CLI-authorized training reconfiguration",
        help="reason stored in the run metadata with --reconfigure",
    )
    train.set_defaults(handler=train_command)

    dashboard = subparsers.add_parser(
        "dashboard",
        help="render a run's static metrics dashboard",
    )
    dashboard.add_argument("run")
    dashboard.add_argument(
        "--smooth",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW,
        help=(
            "trailing window for per-iteration series "
            f"(default: {DEFAULT_SMOOTH_WINDOW})"
        ),
    )
    dashboard.add_argument("--dpi", type=int, default=150)
    dashboard.add_argument(
        "--include-learning-rate",
        action="store_true",
        help="overlay learning rate on the trust-region panel",
    )
    dashboard.set_defaults(handler=dashboard_command)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="evaluate a schema-v6 checkpoint against the heuristic",
    )
    evaluate.add_argument("run")
    evaluate.add_argument(
        "--checkpoint",
        default="latest",
        help="latest, best, an iteration/path, or all interval checkpoints",
    )
    evaluate.add_argument("--opponent", choices=("heuristic",), default="heuristic")
    evaluate.add_argument("--device")
    evaluate.add_argument("--deals", type=int)
    evaluate.add_argument("--seed", type=int)
    evaluate.add_argument("--action-seed", type=int, default=17)
    evaluate.add_argument("--bootstrap-samples", type=int, default=2000)
    evaluate.add_argument(
        "--action-mode",
        choices=("argmax", "sample", "both"),
        default="argmax",
        help=(
            "choose deterministic legal-action argmax, reproducible policy "
            "sampling, or evaluate both (default: argmax)"
        ),
    )
    evaluate.add_argument(
        "--batch-size",
        type=int,
        help="inference batch size (default: min(configured value, 64))",
    )
    evaluate.add_argument("--force", action="store_true")
    evaluate.add_argument(
        "--watch",
        action="store_true",
        help="evaluate each new interval checkpoint until interrupted",
    )
    evaluate.add_argument("--poll-seconds", type=float, default=10.0)
    evaluate.set_defaults(handler=evaluate_command)

    analyze = subparsers.add_parser(
        "analyze",
        help="write run-scoped card-representation analyses",
    )
    analyze.add_argument("run")
    analyze.add_argument("--checkpoint", default="latest")
    analyze.add_argument("--seed", type=int, default=42)
    analyze.add_argument("--permutations", type=int, default=1000)
    analyze.add_argument("--dpi", type=int, default=180)
    analyze.set_defaults(handler=analyze_command)

    play = subparsers.add_parser("play", help="launch the local browser GUI")
    play.add_argument("run", nargs="?")
    play.add_argument("--checkpoint", default="latest")
    play.add_argument("--device")
    play.add_argument("--host", default="127.0.0.1")
    play.add_argument("--port", type=int, default=8765)
    play.set_defaults(handler=play_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.invocation = ["plump", *(argv if argv is not None else sys.argv[1:])]
    try:
        return int(args.handler(args) or 0)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def train_command(args: argparse.Namespace) -> int:
    overrides = list(args.overrides)
    if args.iterations is not None:
        overrides.append(f"run.iterations={args.iterations}")
    if args.device is not None:
        overrides.append(f"run.device={json.dumps(args.device)}")
    requested = load_training_config(args.config, overrides=overrides)
    run = RunDirectory(args.run)
    existed = run.exists
    if existed and args.from_checkpoint is not None:
        raise ValueError("--from-checkpoint requires a new run name.")
    if not existed and args.reconfigure:
        raise ValueError("--reconfigure requires an existing run.")
    if args.reset_league and args.from_checkpoint is None:
        raise ValueError("--reset-league requires --from-checkpoint.")
    if args.prepare_only and existed:
        raise ValueError("--prepare-only requires a new run name.")

    with run.acquire_lock():
        reconfiguration_differences: list[str] = []
        if existed:
            recorded_raw = run.recorded_config()
            differences = config_diff(recorded_raw, requested.raw)
            if differences and not args.reconfigure:
                rendered = "\n  ".join(differences)
                raise ValueError(
                    "Run configuration differs from the recorded config:\n  " + rendered
                )
            if differences:
                reconfiguration_differences = differences
                resolved = requested
            else:
                resolved = resolve_training_config(recorded_raw)
        else:
            run.create(requested.raw, args.invocation)
            resolved = requested

        device_value = str(resolved.run["device"])
        # Preparing a CUDA run on a laptop only needs to rewrite portable CPU
        # checkpoint state. Keep the recorded runtime device untouched while
        # avoiding any attempt to instantiate unavailable accelerator storage.
        device = (
            "cpu"
            if args.prepare_only
            else (
                str(best_seq_device())
                if device_value in {"", "auto"}
                else device_value
            )
        )
        seed = int(resolved.training.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        trainer = SeqTrainer(
            SeqPlumpModel(resolved.model),
            resolved.training,
            device=device,
        )
        trainer.resolved_config = resolved.raw
        if existed:
            checkpoint = run.resolve_checkpoint("latest")
            trainer.load_checkpoint(
                checkpoint,
                allow_training_config_mismatch=bool(reconfiguration_differences),
            )
            discarded = _truncate_metrics_after(run.metrics, trainer.iteration)
            _emit(
                run,
                f"Resumed {run.name} from {checkpoint.name} "
                f"at iteration {trainer.iteration}.",
            )
            if discarded:
                _emit(
                    run,
                    f"Discarded {discarded} metric rows newer than the "
                    "resumed checkpoint.",
                )
            if reconfiguration_differences:
                stem = f"resume_{trainer.iteration:06d}_reconfigured"
                resume = run.checkpoints / f"{stem}.pt"
                suffix = 1
                while resume.exists():
                    resume = run.checkpoints / f"{stem}_{suffix}.pt"
                    suffix += 1
                trainer.save_checkpoint(resume)
                # Point latest at the config-compatible checkpoint before
                # replacing config.toml. If interrupted between these atomic
                # writes, rerunning --reconfigure is always recoverable.
                run.record_latest(resume, trainer.iteration)
                archive = run.record_reconfiguration(
                    resolved.raw,
                    iteration=trainer.iteration,
                    reason=str(args.reconfigure_reason),
                    changes=reconfiguration_differences,
                    source_checkpoint=checkpoint,
                    resume_checkpoint=resume,
                )
                _emit(
                    run,
                    f"Reconfigured {run.name} at iteration {trainer.iteration}; "
                    f"previous config archived as {archive.name} and "
                    f"resume checkpoint saved as {resume.name}.",
                )
        elif args.from_checkpoint is not None:
            source = args.from_checkpoint.expanduser().resolve()
            trainer.load_checkpoint(
                source,
                allow_training_config_mismatch=True,
            )
            trainer.resolved_config = resolved.raw
            if args.reset_league:
                trainer.league.clear()
            imported = run.interval_checkpoint(trainer.iteration)
            trainer.save_checkpoint(imported)
            run.record_latest(imported, trainer.iteration)
            run.update_metadata(
                parent_checkpoint=str(source),
                parent_iteration=trainer.iteration,
            )
            _emit(
                run,
                f"Forked {run.name} from {source.name} "
                f"at iteration {trainer.iteration}.",
            )
        else:
            _emit(run, f"Created run {run.name} on {device}.")
            initial = run.interval_checkpoint(0)
            trainer.save_checkpoint(initial)
            run.record_latest(initial, 0)
            _emit(run, f"Saved reproducible initial checkpoint {initial.name}.")

        _ensure_metrics_header(run.metrics)
        if args.prepare_only:
            run.update_metadata(
                status="prepared",
                prepared_iteration=trainer.iteration,
                target_device=device_value,
                seed=seed,
                target_iterations=int(resolved.run["iterations"]),
            )
            _emit(
                run,
                f"Prepared {run.name} at iteration {trainer.iteration}; "
                "training was not started.",
            )
            return 0
        elapsed_before = _recorded_elapsed(run.metrics)
        run.update_metadata(
            status="running",
            device=device,
            seed=seed,
            target_iterations=int(resolved.run["iterations"]),
        )

        evaluation = resolved.evaluation
        training_action_mode = str(
            evaluation.get("training_action_mode", "argmax")
        )
        switch_reward = float(evaluation.get("opponent_switch_reward", 0.0))
        switch_consecutive = int(
            evaluation.get("opponent_switch_consecutive", 1)
        )
        eval_bank = DealBank.generate(
            player_counts=resolved.training.player_counts,
            hand_sizes=tuple(
                sorted({cell.hand_size for cell in resolved.training.schedule_cells})
            ),
            deals_per_configuration=int(evaluation["deals"]),
            seed=int(evaluation["seed"]),
        )
        heuristic = HeuristicPolicy()
        target = int(resolved.run["iterations"])
        checkpoint_every = int(resolved.run["checkpoint_every"])
        dashboard_every = int(resolved.run["dashboard_every"])

        for iteration in range(trainer.iteration + 1, target + 1):
            trainer.iteration = iteration
            started = time.perf_counter()
            trees, summary = trainer.collect()
            stats = trainer.update(trees)
            collector = trainer.collector.stats

            eval_reward: float | None = None
            eval_bid_hit: float | None = None
            opponent_switched = False
            if (
                int(evaluation["every"]) > 0
                and iteration % int(evaluation["every"]) == 0
            ):
                policy = SeqModelPolicy(
                    trainer.model,
                    device=device,
                    greedy=training_action_mode == "argmax",
                    name="candidate",
                )
                report = evaluate_policy(
                    policy,
                    heuristic,
                    eval_bank,
                    seed=int(evaluation.get("action_seed", 17)),
                    batch_size=int(evaluation["batch_size"]),
                )
                eval_reward = report.macro_relative_reward
                eval_bid_hit = report.macro_bid_hit_rate
                opponent_switched = trainer.record_heuristic_evaluation(
                    eval_reward,
                    threshold=switch_reward,
                    consecutive=switch_consecutive,
                )
                trainer.model.eval()
                best = run.best_metric()
                if best is None or eval_reward > best:
                    best_path = run.checkpoints / "best.pt"
                    trainer.save_checkpoint(best_path)
                    run.record_best(best_path, iteration, eval_reward)

            total = time.perf_counter() - started
            row = _metric_row(
                trainer,
                summary,
                stats,
                collector,
                total,
                elapsed_before + total,
                eval_reward,
                eval_bid_hit,
            )
            _append_metric(run.metrics, row)
            elapsed_before += total

            if (
                checkpoint_every > 0 and iteration % checkpoint_every == 0
            ) or iteration == target:
                checkpoint = run.interval_checkpoint(iteration)
                snapshot_id = f"iter_{iteration}"
                trainer.league.add(snapshot_id, str(checkpoint), iteration)
                try:
                    trainer.save_checkpoint(checkpoint)
                except Exception:
                    trainer.league.snapshots = [
                        snap
                        for snap in trainer.league.snapshots
                        if snap.snapshot_id != snapshot_id
                    ]
                    raise
                run.record_latest(checkpoint, iteration)

            if dashboard_every > 0 and (
                iteration % dashboard_every == 0 or iteration == target
            ):
                try:
                    render_dashboard(
                        run.metrics,
                        run.dashboard,
                        title=f"Plump schema-v6 · {run.name}",
                    )
                except Exception as error:
                    _emit(run, f"Dashboard refresh failed: {error}")

            message = (
                f"iter {iteration:5d} | {total:6.1f}s "
                f"(collect {summary.collect_sec:.1f} update {stats.update_sec:.1f}) "
                f"| leaves {summary.leaves:5d} positions {stats.positions:6d} "
                f"| bid F/N {summary.bid_hit_focal:.3f}/"
                f"{summary.bid_hit_non_focal:.3f} "
                f"| kl {stats.policy_kl:.5f} "
                f"| step {stats.step_scale:.3f}"
            )
            if stats.backtracks:
                message += f" ({stats.backtracks} backtracks)"
            if stats.rolled_back:
                message += " ROLLBACK"
            if eval_reward is not None:
                message += (
                    f" | eval-{training_action_mode} {eval_reward:.4f}"
                    f" | anchor {trainer.opponent_phase}"
                )
                if trainer.opponent_phase == "heuristic":
                    message += (
                        f" gate {trainer.heuristic_eval_win_streak}/"
                        f"{switch_consecutive}"
                    )
            if opponent_switched:
                message += " | HEURISTIC GATE PASSED; SWITCHED TO HISTORY"
            _emit(run, message)

        run.update_metadata(
            status="complete",
            completed_iteration=trainer.iteration,
        )
    return 0


def dashboard_command(args: argparse.Namespace) -> int:
    run = RunDirectory(args.run)
    rows = render_dashboard(
        run.metrics,
        run.dashboard,
        title=f"Plump schema-v6 · {run.name}",
        smooth=args.smooth,
        dpi=args.dpi,
        include_learning_rate=args.include_learning_rate,
    )
    print(f"Wrote {run.dashboard} from {rows} rows.")
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    run = RunDirectory(args.run)
    resolved = resolve_training_config(run.recorded_config())
    evaluation = resolved.evaluation
    device = args.device or str(best_seq_device())
    deals = int(
        evaluation["deals"] if args.deals is None else args.deals
    )
    deal_seed = int(
        evaluation["seed"] if args.seed is None else args.seed
    )
    batch_size = int(
        min(int(evaluation["batch_size"]), 64)
        if args.batch_size is None
        else args.batch_size
    )
    if deals < 1:
        raise ValueError("--deals must be positive.")
    if batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive.")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive.")
    if args.watch and args.checkpoint != "all":
        raise ValueError("--watch requires --checkpoint all.")

    hand_sizes = tuple(
        sorted({cell.hand_size for cell in resolved.training.schedule_cells})
    )
    action_modes = (
        ("argmax", "sample")
        if args.action_mode == "both"
        else (args.action_mode,)
    )
    protocols = tuple(
        EvaluationProtocol(
            opponent=args.opponent,
            player_counts=resolved.training.player_counts,
            hand_sizes=hand_sizes,
            deals_per_configuration=deals,
            deal_seed=deal_seed,
            action_seed=int(args.action_seed),
            bootstrap_samples=int(args.bootstrap_samples),
            batch_size=batch_size,
            greedy=mode == "argmax",
        )
        for mode in action_modes
    )
    bank = DealBank.generate(
        player_counts=protocols[0].player_counts,
        hand_sizes=protocols[0].hand_sizes,
        deals_per_configuration=protocols[0].deals_per_configuration,
        seed=protocols[0].deal_seed,
    )
    processed: set[tuple[Path, bool]] = set()
    metrics_signature: tuple[int, int] | None = None
    while True:
        if args.checkpoint == "all":
            checkpoints = discover_interval_checkpoints(run)
        else:
            checkpoints = [run.resolve_checkpoint(args.checkpoint)]
        pending = [
            (path, protocol)
            for path in checkpoints
            for protocol in protocols
            if (path, protocol.greedy) not in processed
        ]
        if not checkpoints and not args.watch:
            raise FileNotFoundError(
                f"No interval checkpoints in {run.checkpoints}"
            )

        refreshed = False
        for checkpoint, protocol in pending:
            payload, created = evaluate_checkpoint(
                run,
                checkpoint,
                protocol=protocol,
                deal_bank=bank,
                device=device,
                force=args.force,
            )
            report = payload["report"]
            status = "evaluated" if created else "cached"
            action_mode = "argmax" if protocol.greedy else "sample"
            print(
                f"{checkpoint.name} [{action_mode}]: {status} on {device} in "
                f"{float(payload['elapsed_sec']):.1f}s | "
                f"reward={float(report['macro_relative_reward']):.4f} "
                f"[{float(report['relative_reward_ci_low']):.4f}, "
                f"{float(report['relative_reward_ci_high']):.4f}] | "
                f"bid_hit={float(report['macro_bid_hit_rate']):.4f} "
                f"raw_score={float(report['macro_raw_score']):.4f} "
                f"rounds={int(report['rounds'])}",
                flush=True,
            )
            processed.add((checkpoint, protocol.greedy))
            refreshed = True

        current_signature = _file_signature(run.metrics)
        dashboard_changed = (
            refreshed
            or (args.watch and current_signature != metrics_signature)
        )
        if dashboard_changed:
            try:
                rows = render_dashboard(
                    run.metrics,
                    run.evaluation_dashboard,
                    title=f"Plump schema-v6 · {run.name}",
                )
                print(
                    f"Wrote {run.evaluation_dashboard} from {rows} training rows.",
                    flush=True,
                )
            except (FileNotFoundError, ValueError) as error:
                print(f"Dashboard refresh deferred: {error}", flush=True)
            metrics_signature = current_signature

        if not args.watch:
            return 0
        time.sleep(args.poll_seconds)


def analyze_command(args: argparse.Namespace) -> int:
    from plump.analysis.card_geometry import analyze_checkpoint

    run = RunDirectory(args.run)
    checkpoint = run.resolve_checkpoint(args.checkpoint)
    output = run.analysis / _checkpoint_output_name(checkpoint)
    report = analyze_checkpoint(
        checkpoint,
        output,
        seed=args.seed,
        permutations=args.permutations,
        dpi=args.dpi,
    )
    print(f"Wrote {len(report['outputs'])} analyses and {output / 'report.json'}.")
    return 0


def play_command(args: argparse.Namespace) -> int:
    checkpoint = None
    if args.run is not None:
        checkpoint = RunDirectory(args.run).resolve_checkpoint(args.checkpoint)
    run_gui(
        host=args.host,
        port=args.port,
        checkpoint_path=checkpoint,
        device=args.device,
    )
    return 0


def _metric_row(
    trainer,
    summary,
    stats,
    collector,
    total: float,
    elapsed: float,
    eval_reward: float | None,
    eval_bid_hit: float | None,
) -> dict[str, Any]:
    return {
        "iteration": trainer.iteration,
        "optimizer_steps": trainer.optimizer_steps,
        "learning_rate": trainer.optimizer.param_groups[0]["lr"],
        "auxiliary_learning_rate": trainer.optimizer.param_groups[1]["lr"],
        "elapsed_sec": elapsed,
        "total_sec": total,
        "collect_sec": summary.collect_sec,
        "update_sec": stats.update_sec,
        "trees": summary.trees,
        "trees_self": summary.trees_self,
        "trees_heuristic": summary.trees_heuristic,
        "trees_historical": summary.trees_historical,
        "leaves": summary.leaves,
        "decisions": summary.decisions,
        "policy_rows": stats.policy_rows,
        "branched_rows": stats.branched_rows,
        "unbranched_rows": stats.unbranched_rows,
        "positions": stats.positions,
        "forward_rows": collector.forward_rows,
        "branch_decisions": collector.branch_decisions,
        "bid_hit_rate": summary.bid_hit_rate,
        "bid_hit_focal": summary.bid_hit_focal,
        "bid_hit_non_focal": summary.bid_hit_non_focal,
        "reward_focal": summary.reward_focal,
        "reward_non_focal": summary.reward_non_focal,
        "reward_self": summary.reward_self,
        "reward_heuristic": summary.reward_heuristic,
        "reward_historical": summary.reward_historical,
        "opponent_phase": trainer.opponent_phase,
        "heuristic_eval_win_streak": trainer.heuristic_eval_win_streak,
        "spine_entropy": summary.spine_entropy,
        "loss_policy": stats.loss_policy,
        "loss_value": stats.loss_value,
        "loss_value_zero": stats.loss_value_zero,
        "value_rmse": stats.value_rmse,
        "value_zero_rmse": stats.value_zero_rmse,
        "value_correlation": stats.value_correlation,
        "value_prediction_std": stats.value_prediction_std,
        "value_rows": stats.value_rows,
        "loss_suit": stats.loss_suit,
        "loss_trick": stats.loss_trick,
        "loss_bid_hit": stats.loss_bid_hit,
        "entropy": stats.entropy,
        "entropy_bid_normalized": stats.entropy_bid_normalized,
        "entropy_play_normalized": stats.entropy_play_normalized,
        "entropy_alpha_bid": stats.entropy_alpha_bid,
        "entropy_alpha_play": stats.entropy_alpha_play,
        "ppo_ratio_clip_fraction": stats.ppo_ratio_clip_fraction,
        "ppo_behavior_replay_kl": stats.ppo_behavior_replay_kl,
        "advantage_mean": stats.advantage_mean,
        "advantage_std": stats.advantage_std,
        "policy_logit_shift": stats.policy_logit_shift,
        "policy_kl": stats.policy_kl,
        "policy_kl_p95": stats.policy_kl_p95,
        "policy_kl_p99": stats.policy_kl_p99,
        "policy_kl_max": stats.policy_kl_max,
        "proposed_policy_kl": stats.proposed_policy_kl,
        "proposed_policy_kl_p95": stats.proposed_policy_kl_p95,
        "proposed_policy_kl_p99": stats.proposed_policy_kl_p99,
        "proposed_policy_kl_max": stats.proposed_policy_kl_max,
        "proposed_mean_exceeded": int(stats.proposed_mean_exceeded),
        "proposed_p99_exceeded": int(stats.proposed_p99_exceeded),
        "backtracks": stats.backtracks,
        "step_scale": stats.step_scale,
        "rolled_back": int(stats.rolled_back),
        "core_grad_norm": stats.core_grad_norm,
        "auxiliary_grad_norm": stats.auxiliary_grad_norm,
        "critic_grad_norm": stats.critic_grad_norm,
        "peak_update_device_gb": stats.peak_update_device_bytes / (1024**3),
        "eval_reward_vs_heuristic": ("" if eval_reward is None else eval_reward),
        "eval_bid_hit": "" if eval_bid_hit is None else eval_bid_hit,
        "peak_cache_rows": collector.peak_cache_rows,
        "cache_rows_allocated": collector.cache_rows_allocated,
        "cache_pressure": collector.peak_cache_rows
        / max(collector.cache_rows_allocated, 1),
        "peak_device_gb": collector.peak_device_bytes / (1024**3),
        "blocked_by_cache": collector.blocked_by_cache,
        "skipped_by_placement": collector.skipped_by_placement,
        "sample_sec": collector.sample_sec,
        "step_sec": collector.step_sec,
        "compact_sec": collector.compact_sec,
        "token_build_sec": collector.token_build_sec,
        "forward_sec": collector.forward_sec,
        "positions_per_sec": stats.positions / max(total, 1e-9),
        "forward_rows_per_sec": collector.forward_rows / max(summary.collect_sec, 1e-9),
    }


def _ensure_metrics_header(path: Path) -> None:
    if path.exists():
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            header = tuple(reader.fieldnames or ())
            rows = list(reader)
        if header != METRIC_COLUMNS:
            upgradable = {
                "bid_hit_focal",
                "bid_hit_non_focal",
                "reward_focal",
                "reward_non_focal",
                "trees_self",
                "trees_heuristic",
                "trees_historical",
                "reward_heuristic",
                "opponent_phase",
                "heuristic_eval_win_streak",
                "loss_value_zero",
                "auxiliary_learning_rate",
                "value_rmse",
                "value_zero_rmse",
                "value_correlation",
                "value_prediction_std",
                "value_rows",
                "proposed_policy_kl",
                "proposed_policy_kl_p95",
                "proposed_policy_kl_p99",
                "proposed_policy_kl_max",
                "proposed_mean_exceeded",
                "proposed_p99_exceeded",
                "core_grad_norm",
                "auxiliary_grad_norm",
                "policy_logit_shift",
                "entropy_bid_normalized",
                "entropy_play_normalized",
                "entropy_alpha_bid",
                "entropy_alpha_play",
                "ppo_ratio_clip_fraction",
                "ppo_behavior_replay_kl",
                "advantage_mean",
                "advantage_std",
                "critic_grad_norm",
                "peak_update_device_gb",
            }
            missing = set(METRIC_COLUMNS) - set(header)
            expected_existing = tuple(
                column for column in METRIC_COLUMNS if column in header
            )
            if header != expected_existing or not missing <= upgradable:
                raise ValueError("Existing metrics.csv has an incompatible header.")
            # Reporting-schema upgrade only. Historical rows did not retain
            # enough information to reconstruct added outcome or proposal
            # diagnostics, so leave those cells blank and start each series
            # after resume.
            temporary = path.with_name(f".{path.name}.upgrade.tmp")
            try:
                with temporary.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
                    writer.writeheader()
                    writer.writerows(rows)
                temporary.replace(path)
            finally:
                if temporary.exists():
                    temporary.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=METRIC_COLUMNS).writeheader()


def _append_metric(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writerow(row)
        handle.flush()


def _recorded_elapsed(path: Path) -> float:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return 0.0
    return float(rows[-1].get("elapsed_sec") or 0.0)


def _truncate_metrics_after(path: Path, iteration: int) -> int:
    """Atomically discard reporting rows newer than a resumed checkpoint."""

    if not path.is_file():
        return 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        return 0
    kept = [
        row
        for row in rows
        if int(float(row.get("iteration") or 0)) <= iteration
    ]
    discarded = len(rows) - len(kept)
    if not discarded:
        return 0

    temporary = path.with_name(f".{path.name}.resume.tmp")
    try:
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)
            handle.flush()
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return discarded


def _file_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _checkpoint_output_name(checkpoint: Path) -> str:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    iteration = int(payload.get("iteration", 0))
    return f"iter_{iteration:06d}" if iteration > 0 else checkpoint.stem


def _emit(run: RunDirectory, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rendered = f"[{timestamp}] {message}"
    print(rendered, flush=True)
    with run.train_log.open("a") as handle:
        handle.write(rendered + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
