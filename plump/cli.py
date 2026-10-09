"""Unified command-line interface for schema-v6 Plump workflows."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import subprocess
import sys
from dataclasses import replace
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from plump.dashboard import DEFAULT_SMOOTH_WINDOW, render_dashboard
from plump.distill_dashboard import render_distill_dashboard
from plump.evaluation import DealBank
from plump.gui.app import run as run_gui
from plump.run_evaluation import (
    EvaluationProtocol,
    discover_interval_checkpoints,
    ensure_evaluation_summary,
    evaluate_checkpoint,
    evaluation_output,
    result_matches_protocol,
)
from plump.run_config import (
    DEFAULT_CONFIG_PATH,
    config_diff,
    load_training_config,
    resolve_training_config,
)
from plump.runs import RunDirectory, atomic_write_json
from plump.tournament import (
    TournamentProgress,
    TournamentProtocol,
    build_plan,
    build_roster,
    estimate_seconds,
    load_policies,
    run_tournament,
    summarize,
    tournament_payload,
    write_tournament_csv,
)
from plump.tournament_dashboard import render_tournament_dashboard
from plump.seq.model import EMBEDDING_NORM_KEYS, SeqPlumpModel
from plump.seq.autorelease import autorelease_pool
from plump.seq.policy import best_seq_device
from plump.seq.trainer import SeqTrainer, build_training_groups

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
    "loss_rank_boundary",
    "loss_trick",
    "loss_oracle_trick",
    "loss_next_winner",
    "loss_oracle_next_winner",
    "loss_oracle_card_order",
    "loss_bid_hit",
    "suit_accuracy_10c_0",
    "suit_accuracy_10c_4",
    "suit_accuracy_10c_8",
    "suit_loss_10c_0",
    "suit_loss_10c_4",
    "suit_loss_10c_8",
    "rank_boundary_balanced_accuracy_10c_0",
    "rank_boundary_balanced_accuracy_10c_4",
    "rank_boundary_balanced_accuracy_10c_8",
    "rank_boundary_loss_10c_0",
    "rank_boundary_loss_10c_4",
    "rank_boundary_loss_10c_8",
    "trick_accuracy_10c_0",
    "trick_accuracy_10c_4",
    "trick_accuracy_10c_8",
    "oracle_trick_accuracy",
    "next_winner_accuracy",
    "oracle_next_winner_accuracy",
    "oracle_card_order_top1_accuracy",
    "entropy",
    "entropy_bid_normalized",
    "entropy_play_normalized",
    "entropy_target_bid",
    "entropy_target_play",
    "entropy_alpha_bid",
    "entropy_alpha_play",
    "ppo_ratio_clip_fraction",
    "ppo_behavior_replay_kl",
    "advantage_mean",
    "advantage_std",
    "policy_logit_shift",
    "policy_kl_checked",
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
    "embedding_grad_norm",
    "auxiliary_grad_norm",
    "policy_loss_grad_norm",
    "auxiliary_loss_grad_norm",
    "auxiliary_grad_scale",
    "critic_grad_norm",
    "critic_all_player_rmse",
    "critic_all_player_correlation",
    "oracle_value_loss",
    "actor_value_loss",
    "actor_all_player_rmse",
    "actor_value_zero_rmse",
    "critic_loss_first_epoch",
    "critic_loss_last_epoch",
    "critic_loss_reduction",
    "peak_update_device_gb",
    "eval_reward_vs_heuristic",
    "eval_bid_hit",
    "eval_reward_vs_heuristic_sample",
    "eval_bid_hit_sample",
    "eval_reward_vs_heuristic_argmax",
    "eval_bid_hit_argmax",
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
    "distill_kl",
    "distill_kl_bid",
    "distill_kl_play",
    "distill_agree",
    "distill_agree_bid",
    "distill_agree_play",
    "distill_top1_mass_bid",
    "distill_top1_mass_play",
    "distill_teacher_entropy_bid",
    "distill_teacher_entropy_play",
    "distill_student_entropy_bid",
    "distill_student_entropy_play",
    "distill_loss_policy",
    "distill_loss_value",
    "distill_loss_aux",
    "distill_value_rmse",
    "distill_value_correlation",
    "distill_aux_kl_suit",
    "distill_aux_kl_rank_boundary",
    "distill_aux_kl_trick",
    "distill_aux_kl_bid_hit",
    "distill_aux_kl_next_winner",
    "distill_skipped_steps",
    "distill_aux_entropy_floor",
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
        "--resume-checkpoint",
        default="latest",
        help=(
            "checkpoint selector for an existing run (latest, best, or an "
            "iteration); reporting artifacts newer than it are discarded"
        ),
    )
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
            "create/fork a run, or apply an explicit reconfiguration, "
            "without entering its training loop"
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
    train.add_argument(
        "--recycle-after",
        type=int,
        default=0,
        metavar="ITERATIONS",
        help=(
            "exit with status 75 at the first checkpoint after this many "
            "iterations in this process, so a supervisor restarts it from that "
            "checkpoint with a fresh heap (0 disables; see "
            "tools/train_supervised.sh)"
        ),
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

    monitor = subparsers.add_parser(
        "monitor",
        help=(
            "evaluate due checkpoints, apply argmax selection/gates, and "
            "refresh dashboard.png"
        ),
    )
    monitor.add_argument("run")
    monitor.add_argument("--device")
    monitor.add_argument("--deals", type=int)
    monitor.add_argument("--batch-size", type=int)
    monitor.add_argument("--force", action="store_true")
    monitor.set_defaults(handler=monitor_command)

    analyze = subparsers.add_parser(
        "analyze",
        help="write run-scoped card-representation analyses",
    )
    analyze.add_argument("run")
    analyze.add_argument("--checkpoint", default="latest")
    analyze.add_argument("--seed", type=int, default=42)
    analyze.add_argument("--permutations", type=int, default=1000)
    analyze.add_argument("--dpi", type=int, default=180)
    analyze.add_argument(
        "--history",
        action="store_true",
        help=(
            "compute cached scalar card-geometry diagnostics across every "
            "interval checkpoint and write a longitudinal graph"
        ),
    )
    analyze.add_argument(
        "--force",
        action="store_true",
        help="with --history, recompute checkpoints already present in the cache",
    )
    analyze.set_defaults(handler=analyze_command)

    tournament = subparsers.add_parser(
        "tournament",
        help=(
            "play lineage checkpoints against each other on seat-symmetric "
            "mixed tables"
        ),
    )
    tournament.add_argument("run")
    tournament.add_argument(
        "--iterations",
        help=(
            "exact comma-separated lineage iterations to enter; overrides "
            "--from-iteration, --to-iteration, --stride, and latest inclusion"
        ),
    )
    tournament.add_argument(
        "--from-iteration",
        type=int,
        default=20000,
        help="oldest checkpoint to enter, and the anchor for --stride",
    )
    tournament.add_argument(
        "--to-iteration",
        type=int,
        help="newest checkpoint to enter (default: newest in the lineage)",
    )
    tournament.add_argument("--stride", type=int, default=200)
    tournament.add_argument(
        "--no-include-latest",
        dest="include_latest",
        action="store_false",
        help="omit the newest checkpoint when it does not land on the stride",
    )
    tournament.add_argument(
        "--no-heuristic",
        dest="include_heuristic",
        action="store_false",
        help="drop the heuristic anchor from the field",
    )
    tournament.add_argument(
        "--deals-per-match",
        type=int,
        default=64,
        help="shared deals per table; also the inference batch width",
    )
    tournament.add_argument(
        "--waves",
        type=int,
        default=1,
        help="independent re-draws of the tables at every shape",
    )
    tournament.add_argument("--player-counts", default="3,4,5")
    tournament.add_argument("--hand-sizes", default="3-10")
    tournament.add_argument(
        "--action-mode",
        choices=("argmax", "sample"),
        default="argmax",
    )
    tournament.add_argument(
        "--device",
        default="cpu",
        help="cpu, mps, or cuda (default: cpu, so training keeps the accelerator)",
    )
    tournament.add_argument("--deal-seed", type=int, default=1234)
    tournament.add_argument("--action-seed", type=int, default=17)
    tournament.add_argument("--schedule-seed", type=int, default=7)
    tournament.add_argument("--bootstrap-samples", type=int, default=2000)
    tournament.add_argument("--max-concurrent-rounds", type=int, default=8192)
    tournament.add_argument(
        "--name",
        help="output stem under runs/<run>/tournaments (default: from the range)",
    )
    tournament.add_argument(
        "--dry-run",
        action="store_true",
        help="print the roster, table plan, and cost estimate without playing",
    )
    tournament.add_argument(
        "--rate",
        type=float,
        help=(
            "override the measured per-shape decision rate used by --dry-run "
            "with one flat decisions/sec figure"
        ),
    )
    tournament.set_defaults(handler=tournament_command)

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


def _calibrate_embedding_norm(
    run: RunDirectory,
    trainer: SeqTrainer,
    resolved: Any,
) -> None:
    """Fit the embedding-norm gain when forking a model that trained without it.

    RMSNorm throws away the magnitude of the embedding sum, so a gain of 1.0
    rescales the trunk's input by ~1/rms and wrecks the policy. Calibrating on
    real rollout tokens restores the pre-norm scale, which is the closest a
    single gain can get to leaving the forked policy alone.
    """

    if not EMBEDDING_NORM_KEYS <= trainer.migrated_model_keys:
        return
    trees, _ = trainer.collect()
    groups = build_training_groups(trees, resolved.model, resolved.training)
    targets = list(trainer.models)
    if getattr(trainer, "critic", None) is not None:
        targets.append(trainer.critic)
    gains = []
    batches = [
        (
            torch.from_numpy(group.tokens).to(trainer.device),
            torch.from_numpy(np.asarray(group.owned)).to(trainer.device),
        )
        for group in groups
    ]
    for model in targets:
        backbone = getattr(model, "backbone", model)
        if getattr(backbone, "embedding_norm", None) is None:
            continue
        gains.append(backbone.calibrate_embedding_norm(batches))
    _emit(
        run,
        "Calibrated input-embedding RMSNorm gain to "
        + ", ".join(f"{gain:.5f}" for gain in gains)
        + f" over {sum(int(group.owned.sum()) for group in groups)} real tokens.",
    )


# EX_TEMPFAIL. `plump train --recycle-after` exits with this once it has saved
# a checkpoint, and tools/train_supervised.sh restarts the run on seeing it.
RECYCLE_EXIT_CODE = 75


def _restore_evaluation_state(run: RunDirectory, trainer: SeqTrainer) -> None:
    try:
        state = json.loads(run.evaluation_state.read_text())
        iteration = int(state["last_heuristic_eval_iteration"])
        if iteration > trainer.iteration:
            return
        if iteration >= trainer.last_heuristic_eval_iteration:
            trainer.last_heuristic_eval_iteration = iteration
            if (
                trainer.train.rollout.opponent_mode
                == "heuristic_then_historical"
            ):
                trainer.heuristic_eval_win_streak = int(state["win_streak"])
                trainer.opponent_phase = str(state["opponent_phase"])
            else:
                trainer.heuristic_eval_win_streak = 0
                trainer.opponent_phase = (
                    trainer.train.rollout.initial_opponent
                )
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return


def _apply_completed_evaluations(
    run: RunDirectory,
    *,
    opponent_mode: str,
    switch_reward: float,
    switch_consecutive: int,
    maximum_iteration: int,
    best_action_mode: str = "argmax",
    gate_action_mode: str = "argmax",
) -> dict[str, Any]:
    """Rebuild best/gate state from completed paired reports."""

    candidates: list[int] = []
    for directory in run.evaluations.glob("iter_*"):
        match = re.fullmatch(r"iter_(\d+)", directory.name)
        if match is not None:
            candidates.append(int(match.group(1)))
    paired: list[tuple[int, float, float, float, float, Path]] = []
    for iteration in sorted(candidates):
        if iteration > maximum_iteration:
            continue
        sample_path = evaluation_output(
            run, iteration, "heuristic", greedy=False
        )
        argmax_path = evaluation_output(
            run, iteration, "heuristic", greedy=True
        )
        if not sample_path.is_file() or not argmax_path.is_file():
            continue
        try:
            sample = ensure_evaluation_summary(sample_path)["report"]
            argmax = ensure_evaluation_summary(argmax_path)["report"]
            sample_reward = float(sample["macro_relative_reward"])
            sample_bid = float(sample["macro_bid_hit_rate"])
            argmax_reward = float(argmax["macro_relative_reward"])
            argmax_bid = float(argmax["macro_bid_hit_rate"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            continue

        checkpoint = run.interval_checkpoint(iteration)
        if not checkpoint.is_file():
            continue
        paired.append(
            (
                iteration,
                sample_reward,
                sample_bid,
                argmax_reward,
                argmax_bid,
                checkpoint,
            )
        )

    reward_index = {"sample": 1, "argmax": 3}
    if best_action_mode not in reward_index or gate_action_mode not in reward_index:
        raise ValueError("best/gate action modes must be 'sample' or 'argmax'.")
    curriculum_enabled = opponent_mode == "heuristic_then_historical"
    phase = "heuristic" if curriculum_enabled else opponent_mode
    streak = 0
    if curriculum_enabled:
        for row in paired:
            iteration = row[0]
            gate_reward = row[reward_index[gate_action_mode]]
            if iteration <= 0 or phase != "heuristic":
                continue
            streak = streak + 1 if gate_reward > switch_reward else 0
            if streak >= switch_consecutive:
                phase = "historical"

    best_iteration: int | None = None
    best_reward: float | None = None
    if paired:
        best_row = max(paired, key=lambda row: row[reward_index[best_action_mode]])
        best_iteration = best_row[0]
        best_reward = best_row[reward_index[best_action_mode]]
        best_checkpoint = best_row[5]
        best_manifest: dict[str, Any] = {}
        try:
            best_manifest = json.loads(
                (run.checkpoints / "best.json").read_text()
            )
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        if (
            best_manifest.get("action_mode") != best_action_mode
            or best_manifest.get("opponent") != "heuristic"
            or int(best_manifest.get("iteration", -1)) != best_iteration
            or float(best_manifest.get("metric", float("-inf"))) != best_reward
        ):
            run.promote_best(
                best_checkpoint,
                best_iteration,
                best_reward,
                action_mode=best_action_mode,
                opponent="heuristic",
            )

    state = {
        "last_heuristic_eval_iteration": paired[-1][0] if paired else -1,
        "win_streak": streak,
        "opponent_phase": phase,
        "selection_action_mode": best_action_mode,
        "gate_action_mode": gate_action_mode,
        "best_iteration": best_iteration,
        "best_metric": best_reward,
    }
    try:
        previous_state = json.loads(run.evaluation_state.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        previous_state = None
    state_changed = previous_state != state
    if state_changed:
        atomic_write_json(run.evaluation_state, state)
    if paired and state_changed:
        iteration, sample_reward, sample_bid, argmax_reward, argmax_bid, _ = paired[-1]
        print(
            f"Applied paired evals through {iteration} | "
            f"sample reward={sample_reward:.4f} bid_hit={sample_bid:.4f} | "
            f"argmax reward={argmax_reward:.4f} bid_hit={argmax_bid:.4f} | "
            f"best={best_iteration} ({best_reward:.4f}) | anchor {phase}.",
            flush=True,
        )
    return state


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
    if not existed and args.resume_checkpoint != "latest":
        raise ValueError("--resume-checkpoint requires an existing run.")
    if args.reset_league and args.from_checkpoint is None:
        raise ValueError("--reset-league requires --from-checkpoint.")
    if args.prepare_only and existed and not args.reconfigure:
        raise ValueError(
            "--prepare-only on an existing run requires --reconfigure."
        )

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
            checkpoint = run.resolve_checkpoint(args.resume_checkpoint)
            trainer.load_checkpoint(
                checkpoint,
                allow_training_config_mismatch=bool(reconfiguration_differences),
            )
            discarded, stale_evaluations = _discard_reporting_after(
                run, trainer.iteration
            )
            run.record_latest(checkpoint, trainer.iteration)
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
            if stale_evaluations:
                _emit(
                    run,
                    f"Removed {stale_evaluations} stale evaluation directories "
                    "newer than the resumed checkpoint.",
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
            _calibrate_embedding_norm(run, trainer, resolved)
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
        switch_consecutive = int(
            evaluation.get("opponent_switch_consecutive", 1)
        )
        target = int(resolved.run["iterations"])
        checkpoint_every = int(resolved.run["checkpoint_every"])
        dashboard = _BackgroundDashboard(
            run.name, int(resolved.run.get("dashboard_every", 0))
        )
        _restore_evaluation_state(run, trainer)
        pending_slices: list = []
        wave_summary = None
        recycle_after = int(args.recycle_after)
        if recycle_after < 0:
            raise ValueError("--recycle-after must be >= 0.")
        first_iteration = trainer.iteration + 1
        for iteration in range(trainer.iteration + 1, target + 1):
            # One pool per iteration: MPS leaves autoreleased Metal objects
            # behind that a run-loop-less Python process otherwise never
            # frees (see plump.seq.autorelease).
            with autorelease_pool():
                previous_phase = trainer.opponent_phase
                _restore_evaluation_state(run, trainer)
                opponent_switched = (
                    previous_phase == "heuristic"
                    and trainer.opponent_phase == "historical"
                )
                trainer.iteration = iteration
                started = time.perf_counter()
                # One rollout wave feeds several updates. Collect only when the
                # previous wave is exhausted, then consume one slice per
                # iteration, so every per-iteration concern below -- metrics,
                # checkpoints, evaluation, the dashboard, the iteration counter
                # itself -- is completely unchanged.
                if not pending_slices:
                    trees, wave_summary = trainer.collect()
                    pending_slices = trainer.split_for_updates(trees)
                    # Attribute the wave's rollout cost across the updates it
                    # feeds. Reporting it in full on each row would multiply the
                    # run's apparent rollout time by the slice count and make
                    # seconds-per-iteration meaningless.
                    wave_summary = replace(
                        wave_summary,
                        collect_sec=wave_summary.collect_sec / len(pending_slices),
                    )
                summary = wave_summary
                stats = trainer.update(pending_slices.pop(0))
                collector = trainer.collector.stats

                total = time.perf_counter() - started
                row = _metric_row(
                    trainer,
                    summary,
                    stats,
                    collector,
                    total,
                    elapsed_before + total,
                    {},
                )
                _append_metric(run.metrics, row)
                elapsed_before += total
                dashboard.maybe_refresh(iteration)

                checkpoint_due = (
                    checkpoint_every > 0 and iteration % checkpoint_every == 0
                )
                if checkpoint_due or iteration == target:
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

                kl_status = (
                    f"{stats.policy_kl:.5f}" if stats.policy_kl_checked else "off"
                )
                message = (
                    f"iter {iteration:5d} | {total:6.1f}s "
                    f"(collect {summary.collect_sec:.1f} "
                    f"update {stats.update_sec:.1f}) "
                    f"| leaves {summary.leaves:5d} positions {stats.positions:6d} "
                    f"| bid F/N {summary.bid_hit_focal:.3f}/"
                    f"{summary.bid_hit_non_focal:.3f} "
                    f"| kl {kl_status} "
                    f"| step {stats.step_scale:.3f}"
                )
                if stats.backtracks:
                    message += f" ({stats.backtracks} backtracks)"
                if stats.rolled_back:
                    message += " ROLLBACK"
                if (
                    trainer.train.rollout.opponent_mode
                    == "heuristic_then_historical"
                    and trainer.opponent_phase == "heuristic"
                ):
                    message += (
                        f" | gate {trainer.heuristic_eval_win_streak}/"
                        f"{switch_consecutive}"
                    )
                if opponent_switched:
                    message += " | ARGMAX HEURISTIC GATE PASSED; SWITCHED TO HISTORY"
                _emit(run, message)
                if (
                    recycle_after
                    and checkpoint_due
                    and iteration != target
                    and iteration - first_iteration + 1 >= recycle_after
                ):
                    # Exit only straight after a checkpoint, so the restart
                    # resumes from exactly this iteration and discards no
                    # metric rows. The fresh process starts with a clean heap.
                    dashboard.finish()
                    _emit(
                        run,
                        f"Recycling the training process after "
                        f"{iteration - first_iteration + 1} iterations at "
                        f"checkpoint {iteration}; a supervisor resumes from it.",
                    )
                    return RECYCLE_EXIT_CODE

        # One final render so the dashboard reflects the last iteration
        # rather than whatever the last interval happened to catch.
        dashboard.maybe_refresh(trainer.iteration, force=True)
        dashboard.finish()
        run.update_metadata(
            status="complete",
            completed_iteration=trainer.iteration,
        )
    return 0


class _BackgroundDashboard:
    """Keep ``dashboard.png`` fresh while training runs, without stalling it.

    ``dashboard_every`` used to be honoured only by the separate ``monitor``
    command, so a plain ``plump train`` produced no dashboard at all unless
    something else was watching the run. It is refreshed here instead.

    The render runs in a subprocess rather than inline: it costs ~1s, which is
    a real fraction of a fast iteration, and matplotlib in-process would also
    hold figure state inside the training loop. A refresh still in flight when
    the next one is due is simply skipped, so a slow render can never queue up
    behind itself or slow training down.
    """

    def __init__(self, run_name: str, every: int) -> None:
        self.run_name = run_name
        self.every = max(int(every), 0)
        self.last_iteration = 0
        self.process: subprocess.Popen | None = None

    def maybe_refresh(self, iteration: int, *, force: bool = False) -> None:
        if not self.every:
            return
        if not force and iteration < self.last_iteration + self.every:
            return
        if self.process is not None and self.process.poll() is None:
            # Previous render still going; let it finish rather than pile on.
            return
        self.last_iteration = iteration
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-m", "plump.cli", "dashboard", self.run_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            # A dashboard is a convenience; never take training down for it.
            self.process = None

    def finish(self, timeout: float = 30.0) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()


def _render_run_dashboard(run: "RunDirectory", output, **kwargs) -> int:
    """Render whichever dashboard matches the run's objective.

    A distillation run has no reward, advantage or trust region to plot, and
    the reinforcement dashboard would render nine empty panels for it. The
    objective is read from the run's own recorded config so this stays correct
    for a run resumed under a different config file.
    """

    objective = ""
    try:
        objective = str(
            (run.recorded_config().get("training", {}) or {}).get(
                "policy_objective", ""
            )
        )
    except (FileNotFoundError, ValueError, AttributeError):
        objective = ""
    if objective == "distill":
        return render_distill_dashboard(
            run.metrics,
            output,
            title=f"Plump distillation · {run.name}",
            # The reinforcement renderer takes options this one has no
            # analogue for (evaluation overlays, the learning-rate axis);
            # ``title`` is already passed above.
            **{
                key: value
                for key, value in kwargs.items()
                if key in {"smooth", "dpi", "update_budget"}
            },
        )
    return render_dashboard(run.metrics, output, **kwargs)


def dashboard_command(args: argparse.Namespace) -> int:
    run = RunDirectory(args.run)
    rows = _render_run_dashboard(
        run,
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
                rows = _render_run_dashboard(
                    run,
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


def monitor_command(args: argparse.Namespace) -> int:
    """One restartable evaluation/dashboard pass for the shell watcher."""

    run = RunDirectory(args.run)
    resolved = resolve_training_config(run.recorded_config())
    evaluation = resolved.evaluation
    device = args.device or str(best_seq_device())
    deals = int(evaluation["deals"] if args.deals is None else args.deals)
    batch_size = int(
        min(int(evaluation["batch_size"]), 64)
        if args.batch_size is None
        else args.batch_size
    )
    if deals < 1 or batch_size < 1:
        raise ValueError("evaluation deals and batch size must be positive.")

    maximum_iteration = _latest_manifest_iteration(run)
    every = int(evaluation["every"])
    checkpoints = []
    if every > 0:
        checkpoints = [
            checkpoint
            for checkpoint in discover_interval_checkpoints(run)
            if (
                (iteration := _interval_iteration(checkpoint))
                <= maximum_iteration
                and (iteration == 0 or iteration % every == 0)
            )
        ]

    hand_sizes = tuple(
        sorted({cell.hand_size for cell in resolved.training.schedule_cells})
    )
    protocols = tuple(
        EvaluationProtocol(
            opponent="heuristic",
            player_counts=resolved.training.player_counts,
            hand_sizes=hand_sizes,
            deals_per_configuration=deals,
            deal_seed=int(evaluation["seed"]),
            action_seed=int(evaluation.get("action_seed", 17)),
            bootstrap_samples=int(evaluation.get("bootstrap_samples", 2000)),
            batch_size=batch_size,
            greedy=greedy,
        )
        for greedy in (False, True)
    )
    pending = [
        (checkpoint, protocol)
        for checkpoint in checkpoints
        for protocol in protocols
        if args.force
        or not result_matches_protocol(
            evaluation_output(
                run,
                _interval_iteration(checkpoint),
                protocol.opponent,
                greedy=protocol.greedy,
            ),
            protocol,
        )
    ]
    created_any = False
    if pending:
        bank = DealBank.generate(
            player_counts=protocols[0].player_counts,
            hand_sizes=protocols[0].hand_sizes,
            deals_per_configuration=protocols[0].deals_per_configuration,
            seed=protocols[0].deal_seed,
        )
        for checkpoint, protocol in pending:
            payload, created = evaluate_checkpoint(
                run,
                checkpoint,
                protocol=protocol,
                deal_bank=bank,
                device=device,
                force=args.force,
            )
            if created:
                created_any = True
                report = payload["report"]
                mode = "argmax" if protocol.greedy else "sample"
                print(
                    f"{checkpoint.name} [{mode}] evaluated | "
                    f"reward={float(report['macro_relative_reward']):.4f} "
                    f"bid_hit={float(report['macro_bid_hit_rate']):.4f}",
                    flush=True,
                )

    _apply_completed_evaluations(
        run,
        opponent_mode=resolved.training.rollout.opponent_mode,
        switch_reward=float(evaluation.get("opponent_switch_reward", 0.0)),
        switch_consecutive=int(
            evaluation.get("opponent_switch_consecutive", 1)
        ),
        maximum_iteration=maximum_iteration,
        best_action_mode=str(evaluation.get("best_action_mode", "argmax")),
        gate_action_mode=str(
            evaluation.get("opponent_switch_action_mode", "argmax")
        ),
    )
    last_metric_iteration = _last_metric_iteration(run.metrics)
    monitor_state_path = run.evaluations / "monitor.json"
    try:
        monitor_state = json.loads(monitor_state_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        monitor_state = {}
    last_dashboard_iteration = int(
        monitor_state.get("dashboard_iteration", -1)
    )
    dashboard_every = int(resolved.run.get("dashboard_every", 0))
    dashboard_due = (
        not run.dashboard.is_file()
        or created_any
        or (
            dashboard_every > 0
            and last_metric_iteration >= last_dashboard_iteration + dashboard_every
        )
    )
    if dashboard_due:
        rows = _render_run_dashboard(
            run,
            run.dashboard,
            title=f"Plump schema-v6 · {run.name}",
        )
        atomic_write_json(
            monitor_state_path,
            {
                "dashboard_iteration": last_metric_iteration,
                "latest_checkpoint_iteration": maximum_iteration,
            },
        )
        print(
            f"Wrote {run.dashboard} from {rows} rows through checkpoint "
            f"{maximum_iteration}.",
            flush=True,
        )
    return 0


def analyze_command(args: argparse.Namespace) -> int:
    from plump.analysis.card_geometry import (
        analyze_checkpoint,
        analyze_checkpoint_history,
    )

    run = RunDirectory(args.run)
    if args.history:
        checkpoints = discover_interval_checkpoints(run)
        report = analyze_checkpoint_history(
            checkpoints,
            run.analysis,
            seed=args.seed,
            permutations=args.permutations,
            dpi=args.dpi,
            force=args.force,
        )
        print(
            f"Wrote card-geometry history for "
            f"{len(report['checkpoints'])} checkpoints to "
            f"{run.analysis / 'card_geometry_history.png'}.",
            flush=True,
        )
        return 0
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


def _parse_int_list(value: str, label: str) -> tuple[int, ...]:
    """Accept comma lists and inclusive a-b ranges, e.g. "3,4,5" or "3-10"."""

    values: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            low, _, high = part.partition("-")
            values.extend(range(int(low), int(high) + 1))
        else:
            values.append(int(part))
    if not values:
        raise ValueError(f"{label} must list at least one value.")
    return tuple(sorted(set(values)))


def _format_duration(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"


def tournament_command(args: argparse.Namespace) -> int:
    run = RunDirectory(args.run)
    if not run.exists:
        raise FileNotFoundError(run.path)
    exact_iterations = (
        _parse_int_list(args.iterations, "--iterations")
        if args.iterations
        else None
    )
    roster = build_roster(
        run,
        from_iteration=args.from_iteration,
        to_iteration=args.to_iteration,
        stride=args.stride,
        include_latest=args.include_latest,
        include_heuristic=args.include_heuristic,
        iterations=exact_iterations,
    )
    protocol = TournamentProtocol(
        player_counts=_parse_int_list(args.player_counts, "--player-counts"),
        hand_sizes=_parse_int_list(args.hand_sizes, "--hand-sizes"),
        deals_per_match=int(args.deals_per_match),
        waves=int(args.waves),
        deal_seed=int(args.deal_seed),
        action_seed=int(args.action_seed),
        schedule_seed=int(args.schedule_seed),
        bootstrap_samples=int(args.bootstrap_samples),
        greedy=args.action_mode == "argmax",
    )
    plan = build_plan(roster, protocol)
    device = args.device or str(best_seq_device())

    checkpoints = [entry for entry in roster if not entry.is_heuristic]
    per_entry = plan.rounds_per_entry
    runs_present = sorted({entry.run for entry in checkpoints})
    selection_text = (
        "exact selection"
        if exact_iterations is not None
        else f"stride {args.stride}"
    )
    print(
        f"roster: {len(checkpoints)} checkpoints "
        f"({checkpoints[0].iteration}..{checkpoints[-1].iteration} "
        f"{selection_text}) "
        f"from {', '.join(runs_present)}"
        + (" + heuristic" if args.include_heuristic else ""),
        flush=True,
    )
    print(
        f"plan: {len(plan.matches)} tables | {plan.rounds} rounds | "
        f"{plan.decisions} decisions | "
        f"{min(per_entry.values())}-{max(per_entry.values())} rounds per entry | "
        f"batch {protocol.deals_per_match} | device {device}",
        flush=True,
    )
    if args.dry_run:
        if args.rate:
            seconds = plan.decisions / max(args.rate, 1e-9)
            basis = f"{args.rate:.0f} decisions/s"
        else:
            seconds = estimate_seconds(plan, device=device)
            basis = f"{plan.decisions / max(seconds, 1e-9):.0f} decisions/s blended"
        print(f"estimate: ~{_format_duration(seconds)} at {basis}", flush=True)
        return 0

    name = args.name or (
        f"iters-{checkpoints[0].iteration}-{checkpoints[-1].iteration}"
        f"-s{args.stride}-d{protocol.deals_per_match}w{protocol.waves}"
        f"-{args.action_mode}"
    )
    output = run.tournaments / f"{name}.json"
    csv_path = output.with_suffix(".csv")
    named_dashboard = output.with_suffix(".png")
    overall_started = time.perf_counter()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def publish(
        status: str,
        outcomes,
        progress: TournamentProgress | dict[str, Any],
        *,
        final: bool = False,
    ):
        # CIs are the only expensive part of aggregation.  During the run the
        # dashboard shows point estimates with collapsed intervals; the final
        # publication performs the configured bootstrap once.
        report = summarize(
            plan,
            outcomes,
            bootstrap_samples=None if final else 0,
        )
        payload = tournament_payload(
            run,
            plan,
            report,
            device=device,
            elapsed_sec=time.perf_counter() - overall_started,
            status=status,
            progress=progress,
            started_at=started_at,
        )
        atomic_write_json(output, payload)
        write_tournament_csv(csv_path, report)
        # The result-specific image is authoritative when multiple tournaments
        # run concurrently. The stable alias remains a convenient latest view.
        render_tournament_dashboard(output, named_dashboard)
        render_tournament_dashboard(output, run.tournament_dashboard)
        return report

    loading_progress = {
        "loaded_policies": 0,
        "total_policies": len(roster),
        "completed_matches": 0,
        "total_matches": len(plan.matches),
        "completed_decisions": 0,
        "total_decisions": plan.decisions,
        "completed_round_seats": 0,
        "total_round_seats": plan.round_seats,
        "fraction": 0.0,
    }
    publish("loading", (), loading_progress)
    print(f"live results: {output}", flush=True)
    print(f"live dashboard: {run.tournament_dashboard}", flush=True)

    def loaded(position: int, total: int, entry) -> None:
        if position == total or position % 10 == 0:
            print(f"loaded {position}/{total} policies", flush=True)
            publish(
                "loading",
                (),
                {**loading_progress, "loaded_policies": position},
            )

    started = time.perf_counter()
    policies = load_policies(
        roster,
        device=device,
        greedy=protocol.greedy,
        progress=loaded,
    )
    print(f"policies resident in {time.perf_counter() - started:.1f}s", flush=True)

    play_started = time.perf_counter()

    total_decisions = plan.decisions
    seen_decisions = {"count": 0}
    last_progress: dict[str, TournamentProgress | None] = {"value": None}

    def report_progress(shape, done, total, decisions, chunk_sec, played) -> None:
        elapsed = time.perf_counter() - play_started
        seen_decisions["count"] += decisions
        num_players, hand_size = shape
        chunk_rate = decisions / chunk_sec if chunk_sec > 0 else 0.0
        overall = seen_decisions["count"] / elapsed if elapsed > 0 else 0.0
        remaining = (total_decisions - seen_decisions["count"]) / max(overall, 1e-9)
        print(
            f"{num_players}p-{hand_size:>2}c | tables {done}/{total} | "
            f"{chunk_rate:.0f} dec/s here | {overall:.0f} dec/s overall | "
            f"{played} round-seats | eta {_format_duration(remaining)}",
            flush=True,
        )

    def publish_progress(progress: TournamentProgress, outcomes) -> None:
        last_progress["value"] = progress
        publish("running", outcomes, progress)

    outcomes = run_tournament(
        plan,
        policies,
        max_concurrent_rounds=int(args.max_concurrent_rounds),
        progress=report_progress,
        snapshot=publish_progress,
    )
    elapsed = time.perf_counter() - play_started
    complete_progress = last_progress["value"] or {
        "completed_matches": len(plan.matches),
        "total_matches": len(plan.matches),
        "completed_decisions": plan.decisions,
        "total_decisions": plan.decisions,
        "completed_round_seats": len(outcomes),
        "total_round_seats": plan.round_seats,
        "fraction": 1.0,
    }
    report = publish("complete", outcomes, complete_progress, final=True)
    print(
        f"played {plan.rounds} rounds in {_format_duration(elapsed)} "
        f"({plan.decisions / max(elapsed, 1e-9):.0f} decisions/s)",
        flush=True,
    )
    print(f"wrote {output}", flush=True)
    print(f"wrote {csv_path}", flush=True)
    print(f"wrote {named_dashboard}", flush=True)
    print(f"updated {run.tournament_dashboard}", flush=True)
    for entry in report.entries:
        print(
            f"  {entry.key:>14} n={entry.rounds:>6} "
            f"reward={entry.relative_reward:+.4f} "
            f"[{entry.relative_reward_ci_low:+.4f}, {entry.relative_reward_ci_high:+.4f}] "
            f"bid_hit={entry.bid_hit_rate:.4f}",
            flush=True,
        )
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
    eval_reports: dict[str, Any],
) -> dict[str, Any]:
    def measured_kl(value: float) -> float | str:
        # An omitted replay is not a measured zero. Empty cells keep the
        # dashboard and CSV consumers from reporting a fictitious KL value.
        return value if stats.policy_kl_checked else ""

    sample_report = eval_reports.get("sample")
    argmax_report = eval_reports.get("argmax")
    auxiliary_group = next(
        (
            group
            for group in trainer.optimizer.param_groups
            if not group.get("kl_sensitive", True)
        ),
        trainer.optimizer.param_groups[0],
    )
    return {
        "iteration": trainer.iteration,
        "optimizer_steps": trainer.optimizer_steps,
        "learning_rate": trainer.optimizer.param_groups[0]["lr"],
        "auxiliary_learning_rate": auxiliary_group["lr"],
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
        "distill_kl": stats.distill_kl,
        "distill_kl_bid": stats.distill_kl_bid,
        "distill_kl_play": stats.distill_kl_play,
        "distill_agree": stats.distill_agree,
        "distill_agree_bid": stats.distill_agree_bid,
        "distill_agree_play": stats.distill_agree_play,
        "distill_top1_mass_bid": stats.distill_top1_mass_bid,
        "distill_top1_mass_play": stats.distill_top1_mass_play,
        "distill_teacher_entropy_bid": stats.distill_teacher_entropy_bid,
        "distill_teacher_entropy_play": stats.distill_teacher_entropy_play,
        "distill_student_entropy_bid": stats.distill_student_entropy_bid,
        "distill_student_entropy_play": stats.distill_student_entropy_play,
        "distill_loss_policy": stats.distill_loss_policy,
        "distill_loss_value": stats.distill_loss_value,
        "distill_loss_aux": stats.distill_loss_aux,
        "distill_value_rmse": stats.distill_value_rmse,
        "distill_value_correlation": stats.distill_value_correlation,
        "distill_aux_kl_suit": stats.distill_aux_kl_suit,
        "distill_aux_kl_rank_boundary": stats.distill_aux_kl_rank_boundary,
        "distill_aux_kl_trick": stats.distill_aux_kl_trick,
        "distill_aux_kl_bid_hit": stats.distill_aux_kl_bid_hit,
        "distill_aux_kl_next_winner": stats.distill_aux_kl_next_winner,
        "distill_skipped_steps": stats.distill_skipped_steps,
        "distill_aux_entropy_floor": stats.distill_aux_entropy_floor,

        "loss_policy": stats.loss_policy,
        "loss_value": stats.loss_value,
        "loss_value_zero": stats.loss_value_zero,
        "value_rmse": stats.value_rmse,
        "value_zero_rmse": stats.value_zero_rmse,
        "value_correlation": stats.value_correlation,
        "value_prediction_std": stats.value_prediction_std,
        "value_rows": stats.value_rows,
        "loss_suit": stats.loss_suit,
        "loss_rank_boundary": stats.loss_rank_boundary,
        "loss_trick": stats.loss_trick,
        "loss_oracle_trick": stats.loss_oracle_trick,
        "loss_next_winner": stats.loss_next_winner,
        "loss_oracle_next_winner": stats.loss_oracle_next_winner,
        "loss_oracle_card_order": stats.loss_oracle_card_order,
        "loss_bid_hit": stats.loss_bid_hit,
        "suit_accuracy_10c_0": stats.suit_accuracy_10c_0,
        "suit_accuracy_10c_4": stats.suit_accuracy_10c_4,
        "suit_accuracy_10c_8": stats.suit_accuracy_10c_8,
        "suit_loss_10c_0": stats.suit_loss_10c_0,
        "suit_loss_10c_4": stats.suit_loss_10c_4,
        "suit_loss_10c_8": stats.suit_loss_10c_8,
        "rank_boundary_balanced_accuracy_10c_0": (
            stats.rank_boundary_balanced_accuracy_10c_0
        ),
        "rank_boundary_balanced_accuracy_10c_4": (
            stats.rank_boundary_balanced_accuracy_10c_4
        ),
        "rank_boundary_balanced_accuracy_10c_8": (
            stats.rank_boundary_balanced_accuracy_10c_8
        ),
        "rank_boundary_loss_10c_0": stats.rank_boundary_loss_10c_0,
        "rank_boundary_loss_10c_4": stats.rank_boundary_loss_10c_4,
        "rank_boundary_loss_10c_8": stats.rank_boundary_loss_10c_8,
        "trick_accuracy_10c_0": stats.trick_accuracy_10c_0,
        "trick_accuracy_10c_4": stats.trick_accuracy_10c_4,
        "trick_accuracy_10c_8": stats.trick_accuracy_10c_8,
        "oracle_trick_accuracy": stats.oracle_trick_accuracy,
        "next_winner_accuracy": stats.next_winner_accuracy,
        "oracle_next_winner_accuracy": stats.oracle_next_winner_accuracy,
        "oracle_card_order_top1_accuracy": (
            stats.oracle_card_order_top1_accuracy
        ),
        "entropy": stats.entropy,
        "entropy_bid_normalized": stats.entropy_bid_normalized,
        "entropy_play_normalized": stats.entropy_play_normalized,
        "entropy_target_bid": stats.entropy_target_bid,
        "entropy_target_play": stats.entropy_target_play,
        "entropy_alpha_bid": stats.entropy_alpha_bid,
        "entropy_alpha_play": stats.entropy_alpha_play,
        "ppo_ratio_clip_fraction": stats.ppo_ratio_clip_fraction,
        "ppo_behavior_replay_kl": stats.ppo_behavior_replay_kl,
        "advantage_mean": stats.advantage_mean,
        "advantage_std": stats.advantage_std,
        "policy_logit_shift": stats.policy_logit_shift,
        "policy_kl_checked": int(stats.policy_kl_checked),
        "policy_kl": measured_kl(stats.policy_kl),
        "policy_kl_p95": measured_kl(stats.policy_kl_p95),
        "policy_kl_p99": measured_kl(stats.policy_kl_p99),
        "policy_kl_max": measured_kl(stats.policy_kl_max),
        "proposed_policy_kl": measured_kl(stats.proposed_policy_kl),
        "proposed_policy_kl_p95": measured_kl(stats.proposed_policy_kl_p95),
        "proposed_policy_kl_p99": measured_kl(stats.proposed_policy_kl_p99),
        "proposed_policy_kl_max": measured_kl(stats.proposed_policy_kl_max),
        "proposed_mean_exceeded": measured_kl(int(stats.proposed_mean_exceeded)),
        "proposed_p99_exceeded": measured_kl(int(stats.proposed_p99_exceeded)),
        "backtracks": stats.backtracks,
        "step_scale": stats.step_scale,
        "rolled_back": int(stats.rolled_back),
        "core_grad_norm": stats.core_grad_norm,
        "embedding_grad_norm": stats.embedding_grad_norm,
        "auxiliary_grad_norm": stats.auxiliary_grad_norm,
        "policy_loss_grad_norm": stats.policy_loss_grad_norm,
        "auxiliary_loss_grad_norm": stats.auxiliary_loss_grad_norm,
        "auxiliary_grad_scale": stats.auxiliary_grad_scale,
        "critic_grad_norm": stats.critic_grad_norm,
        "critic_all_player_rmse": stats.critic_all_player_rmse,
        "oracle_value_loss": stats.oracle_value_loss,
        "actor_value_loss": stats.actor_value_loss,
        "actor_all_player_rmse": stats.actor_all_player_rmse,
        "actor_value_zero_rmse": stats.actor_value_zero_rmse,
        "critic_all_player_correlation": stats.critic_all_player_correlation,
        "critic_loss_first_epoch": stats.critic_loss_first_epoch,
        "critic_loss_last_epoch": stats.critic_loss_last_epoch,
        "critic_loss_reduction": stats.critic_loss_reduction,
        "peak_update_device_gb": stats.peak_update_device_bytes / (1024**3),
        # Legacy aliases intentionally follow sample mode: the curriculum gate
        # and best-checkpoint selection are both defined by sampled reward.
        "eval_reward_vs_heuristic": (
            "" if sample_report is None else sample_report.macro_relative_reward
        ),
        "eval_bid_hit": (
            "" if sample_report is None else sample_report.macro_bid_hit_rate
        ),
        "eval_reward_vs_heuristic_sample": (
            "" if sample_report is None else sample_report.macro_relative_reward
        ),
        "eval_bid_hit_sample": (
            "" if sample_report is None else sample_report.macro_bid_hit_rate
        ),
        "eval_reward_vs_heuristic_argmax": (
            "" if argmax_report is None else argmax_report.macro_relative_reward
        ),
        "eval_bid_hit_argmax": (
            "" if argmax_report is None else argmax_report.macro_bid_hit_rate
        ),
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
                "policy_kl_checked",
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
                "embedding_grad_norm",
                "auxiliary_grad_norm",
                "policy_loss_grad_norm",
                "auxiliary_loss_grad_norm",
                "auxiliary_grad_scale",
                "policy_logit_shift",
                "entropy_bid_normalized",
                "entropy_play_normalized",
                "entropy_target_bid",
                "entropy_target_play",
                "entropy_alpha_bid",
                "entropy_alpha_play",
                "ppo_ratio_clip_fraction",
                "ppo_behavior_replay_kl",
                "advantage_mean",
                "advantage_std",
                "critic_grad_norm",
                "critic_all_player_rmse",
                "oracle_value_loss",
                "actor_value_loss",
                "actor_all_player_rmse",
                "actor_value_zero_rmse",
                "critic_all_player_correlation",
                "critic_loss_first_epoch",
                "critic_loss_last_epoch",
                "critic_loss_reduction",
                "peak_update_device_gb",
                "loss_oracle_trick",
                "loss_next_winner",
                "loss_oracle_next_winner",
                "next_winner_accuracy",
                "oracle_next_winner_accuracy",
                "loss_oracle_card_order",
                "loss_rank_boundary",
                "suit_accuracy_10c_0",
                "suit_accuracy_10c_4",
                "suit_accuracy_10c_8",
                "suit_loss_10c_0",
                "suit_loss_10c_4",
                "suit_loss_10c_8",
                "rank_boundary_balanced_accuracy_10c_0",
                "rank_boundary_balanced_accuracy_10c_4",
                "rank_boundary_balanced_accuracy_10c_8",
                "rank_boundary_loss_10c_0",
                "rank_boundary_loss_10c_4",
                "rank_boundary_loss_10c_8",
                "trick_accuracy_10c_0",
                "trick_accuracy_10c_4",
                "trick_accuracy_10c_8",
                "oracle_trick_accuracy",
                "oracle_card_order_top1_accuracy",
                "eval_reward_vs_heuristic_sample",
                "eval_bid_hit_sample",
                "eval_reward_vs_heuristic_argmax",
                "eval_bid_hit_argmax",
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


def _last_metric_iteration(path: Path) -> int:
    if not path.is_file():
        return -1
    with path.open(newline="") as handle:
        last = None
        for last in csv.DictReader(handle):
            pass
    return -1 if last is None else int(float(last.get("iteration") or -1))


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


def _discard_reporting_after(
    run: RunDirectory, iteration: int
) -> tuple[int, int]:
    """Drop stale metric/evaluation rows when resuming an older checkpoint."""

    metric_rows = _truncate_metrics_after(run.metrics, iteration)
    evaluation_directories = 0
    for directory in run.evaluations.glob("iter_*"):
        match = re.fullmatch(r"iter_(\d+)", directory.name)
        if match is None or int(match.group(1)) <= iteration:
            continue
        shutil.rmtree(directory)
        evaluation_directories += 1
    # Rebuild this tiny derived state from the surviving evaluation rows. This
    # also makes selection/gate method changes take effect on the next monitor
    # pass rather than carrying forward stale sampled-reward semantics.
    run.evaluation_state.unlink(missing_ok=True)
    (run.evaluations / "monitor.json").unlink(missing_ok=True)
    try:
        best = json.loads((run.checkpoints / "best.json").read_text())
        best_is_future = int(best.get("iteration", -1)) > iteration
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        best_is_future = False
    if best_is_future:
        (run.checkpoints / "best.json").unlink(missing_ok=True)
        (run.checkpoints / "best.pt").unlink(missing_ok=True)
    return metric_rows, evaluation_directories


def _interval_iteration(checkpoint: Path) -> int:
    match = re.fullmatch(r"iter_(\d+)\.pt", checkpoint.name)
    if match is None:
        raise ValueError(f"Not an interval checkpoint: {checkpoint}")
    return int(match.group(1))


def _latest_manifest_iteration(run: RunDirectory) -> int:
    try:
        return int(
            json.loads((run.checkpoints / "latest.json").read_text())["iteration"]
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise FileNotFoundError(
            f"No valid latest checkpoint manifest in {run.checkpoints}"
        ) from None


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
