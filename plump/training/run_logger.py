"""Structured schema-v4 logging for Plump training runs."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from plump.evaluation import EvaluationReport
from plump.modeling import SCHEMA_VERSION
from plump.rounds import rules_fingerprint

from .ppo import (
    CollectionStats,
    PredictionStats,
    RolloutStats,
    TrainingConfig,
    UpdateStats,
)
from .search_distill import SearchRoutingStats, SearchUpdateStats


def _mixture_entropy(mixture: dict[str, float]) -> float:
    total = sum(value for value in mixture.values() if value > 0.0)
    if total <= 0.0:
        return 0.0
    return -sum(
        (value / total) * math.log(value / total)
        for value in mixture.values()
        if value > 0.0
    )


def _current_vs_mixture(
    payoffs: dict[tuple[str, str], float],
    mixture: dict[str, float],
) -> float:
    """Current policy's payoff against the learned league mixture."""

    weighted = [
        (weight, payoffs[("current", member_id)])
        for member_id, weight in mixture.items()
        if weight > 0.0 and ("current", member_id) in payoffs
    ]
    total = sum(weight for weight, _ in weighted)
    if total <= 0.0:
        return 0.0
    return sum(weight * reward for weight, reward in weighted) / total


METRIC_FIELDS = [
    "iteration",
    "timestamp_utc",
    "schema_version",
    "rules_fingerprint",
    "elapsed_sec",
    "iteration_sec",
    "collect_sec",
    "collect_current_forward_sec",
    "collect_historical_forward_sec",
    "collect_env_step_sec",
    "collect_finalize_sec",
    "collect_current_forward_calls",
    "collect_current_forward_rows",
    "collect_historical_forward_calls",
    "collect_historical_forward_rows",
    "collect_historical_policy_count",
    "collect_valid_event_tokens",
    "collect_processed_event_tokens",
    "collect_peak_device_memory_bytes",
    "collect_branch_root_hands",
    "collect_branch_roots_available",
    "collect_branch_roots_expanded",
    "collect_branch_bid_roots",
    "collect_branch_play_roots",
    "collect_branch_terminal_rollouts",
    "collect_branch_self_decisions",
    "collect_branch_historical_decisions",
    "collect_branch_forward_sec",
    "update_sec",
    "diagnostics_sec",
    "eval_sec",
    "checkpoint_sec",
    "configurations",
    "rollout_rounds",
    "samples",
    "bid_samples",
    "play_samples",
    "rollout_bid_hit_rate",
    "rollout_bid_abs_error_mean",
    "rollout_all_player_bid_hit_rate",
    "rollout_all_player_bid_abs_error_mean",
    "rollout_bid_hit_p3",
    "rollout_bid_hit_p4",
    "rollout_bid_hit_p5",
    "rollout_bid_hit_h3_5",
    "rollout_bid_hit_h6_8",
    "rollout_bid_hit_h9_10",
    "rollout_heuristic_relative_reward",
    "rollout_historical_relative_reward",
    "rollout_explore_self_relative_reward",
    "rollout_explore_historical_relative_reward",
    "rollout_tempered_self_relative_reward",
    "rollout_tempered_historical_relative_reward",
    "rollout_capped_self_relative_reward",
    "rollout_capped_historical_relative_reward",
    "rollout_epsilon_self_relative_reward",
    "rollout_epsilon_historical_relative_reward",
    "rollout_self_play_rounds",
    "rollout_heuristic_rounds",
    "rollout_mixed_rounds",
    "rollout_historical_rounds",
    "rollout_explore_self_rounds",
    "rollout_explore_historical_rounds",
    "rollout_tempered_self_rounds",
    "rollout_tempered_historical_rounds",
    "rollout_capped_self_rounds",
    "rollout_capped_historical_rounds",
    "rollout_epsilon_self_rounds",
    "rollout_epsilon_historical_rounds",
    "league_meta_entropy",
    "league_current_vs_mixture",
    "loss_total",
    "loss_policy",
    "loss_branch_policy",
    "branch_kl",
    "branch_target_entropy",
    "branch_samples",
    "loss_value",
    "loss_oracle_value",
    "magnet_kl",
    "loss_auxiliary",
    "loss_trick",
    "loss_owner",
    "loss_owner_ce",
    "loss_owner_capacity",
    "loss_suit_presence",
    "entropy_update",
    "approx_kl",
    "clip_fraction",
    "skipped_steps",
    "epochs_run",
    "return_mean",
    "return_std",
    "old_value_mean",
    "advantage_std",
    "bid_advantage_mean",
    "bid_advantage_std",
    "bid_advantage_abs_p50",
    "bid_advantage_abs_p90",
    "play_advantage_mean",
    "play_advantage_std",
    "play_advantage_abs_p50",
    "play_advantage_abs_p90",
    "old_bid_entropy_mean",
    "old_play_entropy_mean",
    "pred_samples",
    "pred_value_mae",
    "pred_value_mse",
    "pred_value_explained_variance",
    "pred_oracle_value_explained_variance",
    "pred_bid_value_explained_variance",
    "pred_play_value_explained_variance",
    "pred_trick_implied_value_explained_variance",
    "pred_bid_trick_implied_value_explained_variance",
    "pred_play_trick_implied_value_explained_variance",
    "pred_trick_count_accuracy",
    "pred_trick_count_true_prob",
    "pred_trick_accuracy_bidtime",
    "pred_trick_accuracy_early",
    "pred_trick_accuracy_mid",
    "pred_trick_accuracy_late",
    "pred_trick_accuracy_p3",
    "pred_trick_accuracy_p4",
    "pred_trick_accuracy_p5",
    "pred_trick_accuracy_h3_5",
    "pred_trick_accuracy_h6_8",
    "pred_trick_accuracy_h9_10",
    "pred_suit_presence_accuracy",
    "pred_suit_presence_brier",
    "pred_suit_presence_loss_early",
    "pred_suit_presence_loss_mid",
    "pred_suit_presence_loss_late",
    "pred_suit_presence_accuracy_early",
    "pred_suit_presence_accuracy_mid",
    "pred_suit_presence_accuracy_late",
    "pred_owner_accuracy",
    "pred_owner_true_prob",
    "pred_owner_brier",
    "pred_owner_opponent_accuracy",
    "pred_owner_opponent_true_prob",
    "pred_owner_capacity_mae",
    "pred_owner_capacity_max_error",
    "pred_owner_raw_capacity_mae",
    "pred_hit_prob_brier",
    "pred_bid_entropy",
    "pred_play_entropy",
    "pred_bid_max_prob",
    "pred_play_max_prob",
    "eval_rounds",
    "eval_macro_relative_reward",
    "eval_ci_low",
    "eval_ci_high",
    "eval_macro_bid_hit_rate",
    "eval_mean_forward_passes",
    "eval_elo_delta",
    "search_bid_eligible",
    "search_bid_probed",
    "search_bid_accepted_rate",
    "search_bid_agreement",
    "search_bid_median_js",
    "search_bid_ci_low",
    "search_bid_sampler_infeasible_rejection_rate",
    "search_bid_sampler_failed_draw_rate",
    "search_bid_gate_passed",
    "search_bid_routed",
    "search_play_eligible",
    "search_play_probed",
    "search_play_accepted_rate",
    "search_play_agreement",
    "search_play_median_js",
    "search_play_ci_low",
    "search_play_sampler_infeasible_rejection_rate",
    "search_play_sampler_failed_draw_rate",
    "search_play_gate_passed",
    "search_play_routed",
    "search_bid_update_samples",
    "search_bid_update_loss",
    "search_bid_cross_entropy_loss",
    "search_bid_entropy_floor_loss",
    "search_bid_policy_entropy",
    "search_bid_target_entropy_floor",
    "search_bid_update_kl",
    "search_bid_max_stratum_kl",
    "search_bid_kl_cap",
    "search_bid_regret_fraction",
    "search_bid_backtracks",
    "search_bid_update_applied",
    "search_play_update_samples",
    "search_play_update_loss",
    "search_play_cross_entropy_loss",
    "search_play_entropy_floor_loss",
    "search_play_policy_entropy",
    "search_play_target_entropy_floor",
    "search_play_update_kl",
    "search_play_max_stratum_kl",
    "search_play_kl_cap",
    "search_play_regret_fraction",
    "search_play_backtracks",
    "search_play_update_applied",
    "checkpoint_path",
]


class TrainingRunLogger:
    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.log_dir / "metrics.csv"
        self.events_path = self.log_dir / "events.jsonl"
        self.config_path = self.log_dir / "run_config.json"
        self.latest_path = self.log_dir / "latest.json"
        if not self.metrics_path.exists():
            with self.metrics_path.open("w", newline="") as file:
                csv.DictWriter(file, fieldnames=METRIC_FIELDS).writeheader()
        else:
            self._upgrade_metrics_header()

    def _upgrade_metrics_header(self) -> None:
        with self.metrics_path.open(newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames == METRIC_FIELDS:
                return
            rows = list(reader)
        temporary_path = self.metrics_path.with_suffix(".csv.tmp")
        with temporary_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        field: row.get(field, "")
                        for field in METRIC_FIELDS
                    }
                )
        temporary_path.replace(self.metrics_path)

    def write_config(self, payload: dict[str, Any]) -> None:
        data = {
            "schema_version": SCHEMA_VERSION,
            "rules_fingerprint": rules_fingerprint(),
            **payload,
        }
        self.config_path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n")

    def log_iteration(
        self,
        *,
        iteration: int,
        elapsed_sec: float,
        timings: dict[str, float],
        update: UpdateStats,
        rollout: RolloutStats,
        collection: CollectionStats | None,
        prediction: PredictionStats | None,
        evaluation: EvaluationReport | None,
        checkpoint_path: Path | None,
        search_routing: list[SearchRoutingStats] | None = None,
        search_updates: list[SearchUpdateStats] | None = None,
        league: dict[str, float] | None = None,
        league_mixture: dict[str, float] | None = None,
        league_payoffs: dict[tuple[str, str], float] | None = None,
    ) -> None:
        row = self._row(
            iteration,
            elapsed_sec,
            timings,
            update,
            rollout,
            collection,
            prediction,
            evaluation,
            checkpoint_path,
            search_routing,
            search_updates,
        )
        row["league_meta_entropy"] = _mixture_entropy(league_mixture or {})
        row["league_current_vs_mixture"] = _current_vs_mixture(
            league_payoffs or {},
            league_mixture or {},
        )
        with self.metrics_path.open("a", newline="") as file:
            csv.DictWriter(file, fieldnames=METRIC_FIELDS).writerow(row)
        event = {
            "type": "iteration",
            "iteration": iteration,
            "timestamp_utc": row["timestamp_utc"],
            "schema_version": SCHEMA_VERSION,
            "rules_fingerprint": rules_fingerprint(),
            "elapsed_sec": elapsed_sec,
            "timings": timings,
            "update": asdict(update),
            "rollout": asdict(rollout),
            "collection": (
                asdict(collection) if collection is not None else None
            ),
            "prediction": asdict(prediction) if prediction is not None else None,
            "evaluation": asdict(evaluation) if evaluation is not None else None,
            "search_routing": [
                asdict(stats)
                for stats in (search_routing or [])
            ],
            "search_updates": [
                asdict(stats)
                for stats in (search_updates or [])
            ],
            "league_reward_ema": dict(league or {}),
            "league_meta_mixture": dict(league_mixture or {}),
            "league_current_payoff_row": {
                table_id: value
                for (focal_id, table_id), value in (league_payoffs or {}).items()
                if focal_id == "current"
            },
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
        }
        with self.events_path.open("a") as file:
            file.write(json.dumps(_jsonable(event), sort_keys=True) + "\n")
        self.latest_path.write_text(json.dumps(_jsonable(event), indent=2, sort_keys=True) + "\n")

    def _row(
        self,
        iteration: int,
        elapsed_sec: float,
        timings: dict[str, float],
        update: UpdateStats,
        rollout: RolloutStats,
        collection: CollectionStats | None,
        prediction: PredictionStats | None,
        evaluation: EvaluationReport | None,
        checkpoint_path: Path | None,
        search_routing: list[SearchRoutingStats] | None,
        search_updates: list[SearchUpdateStats] | None,
    ) -> dict[str, Any]:
        row = {
            "iteration": iteration,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "schema_version": SCHEMA_VERSION,
            "rules_fingerprint": rules_fingerprint(),
            "elapsed_sec": elapsed_sec,
            **{name: timings.get(name, 0.0) for name in (
                "iteration_sec",
                "collect_sec",
                "update_sec",
                "diagnostics_sec",
                "eval_sec",
                "checkpoint_sec",
            )},
            "configurations": update.configurations,
            "rollout_rounds": rollout.rounds,
            "samples": update.samples,
            "bid_samples": rollout.bid_samples,
            "play_samples": rollout.play_samples,
            "rollout_bid_hit_rate": rollout.bid_hit_rate,
            "rollout_bid_abs_error_mean": rollout.bid_abs_error_mean,
            "rollout_all_player_bid_hit_rate": rollout.all_player_bid_hit_rate,
            "rollout_all_player_bid_abs_error_mean": (
                rollout.all_player_bid_abs_error_mean
            ),
            "rollout_bid_hit_p3": rollout.bid_hit_rate_by_players.get(3, ""),
            "rollout_bid_hit_p4": rollout.bid_hit_rate_by_players.get(4, ""),
            "rollout_bid_hit_p5": rollout.bid_hit_rate_by_players.get(5, ""),
            "rollout_bid_hit_h3_5": rollout.bid_hit_rate_by_hand_bucket.get("3_5", ""),
            "rollout_bid_hit_h6_8": rollout.bid_hit_rate_by_hand_bucket.get("6_8", ""),
            "rollout_bid_hit_h9_10": rollout.bid_hit_rate_by_hand_bucket.get("9_10", ""),
            "rollout_heuristic_relative_reward": rollout.heuristic_relative_reward,
            "rollout_historical_relative_reward": rollout.historical_relative_reward,
            "rollout_explore_self_relative_reward": (
                rollout.explore_self_relative_reward
            ),
            "rollout_explore_historical_relative_reward": (
                rollout.explore_historical_relative_reward
            ),
            "rollout_tempered_self_relative_reward": (
                rollout.tempered_self_relative_reward
            ),
            "rollout_tempered_historical_relative_reward": (
                rollout.tempered_historical_relative_reward
            ),
            "rollout_capped_self_relative_reward": (
                rollout.capped_self_relative_reward
            ),
            "rollout_capped_historical_relative_reward": (
                rollout.capped_historical_relative_reward
            ),
            "rollout_epsilon_self_relative_reward": (
                rollout.epsilon_self_relative_reward
            ),
            "rollout_epsilon_historical_relative_reward": (
                rollout.epsilon_historical_relative_reward
            ),
            "rollout_self_play_rounds": rollout.self_play_rounds,
            "rollout_heuristic_rounds": rollout.heuristic_rounds,
            "rollout_mixed_rounds": rollout.mixed_rounds,
            "rollout_historical_rounds": rollout.historical_rounds,
            "rollout_explore_self_rounds": rollout.explore_self_rounds,
            "rollout_explore_historical_rounds": rollout.explore_historical_rounds,
            "rollout_tempered_self_rounds": rollout.tempered_self_rounds,
            "rollout_tempered_historical_rounds": (
                rollout.tempered_historical_rounds
            ),
            "rollout_capped_self_rounds": rollout.capped_self_rounds,
            "rollout_capped_historical_rounds": rollout.capped_historical_rounds,
            "rollout_epsilon_self_rounds": rollout.epsilon_self_rounds,
            "rollout_epsilon_historical_rounds": (
                rollout.epsilon_historical_rounds
            ),
            "loss_total": update.total_loss,
            "loss_policy": update.policy_loss,
            "loss_branch_policy": update.branch_policy_loss,
            "branch_kl": update.branch_kl,
            "branch_target_entropy": update.branch_target_entropy,
            "branch_samples": update.branch_samples,
            "loss_value": update.value_loss,
            "loss_oracle_value": update.oracle_value_loss,
            "magnet_kl": update.magnet_kl,
            "loss_auxiliary": update.auxiliary_loss,
            "loss_trick": update.trick_loss,
            "loss_owner": update.owner_loss,
            "loss_owner_ce": update.owner_ce_loss,
            "loss_owner_capacity": update.owner_capacity_loss,
            "loss_suit_presence": update.suit_presence_loss,
            "entropy_update": update.entropy,
            "approx_kl": update.approx_kl,
            "clip_fraction": update.clip_fraction,
            "skipped_steps": update.skipped_steps,
            "epochs_run": update.epochs_run,
            "return_mean": rollout.return_mean,
            "return_std": rollout.return_std,
            "old_value_mean": rollout.old_value_mean,
            "advantage_std": rollout.advantage_std,
            "bid_advantage_mean": rollout.bid_advantage_mean,
            "bid_advantage_std": rollout.bid_advantage_std,
            "bid_advantage_abs_p50": rollout.bid_advantage_abs_p50,
            "bid_advantage_abs_p90": rollout.bid_advantage_abs_p90,
            "play_advantage_mean": rollout.play_advantage_mean,
            "play_advantage_std": rollout.play_advantage_std,
            "play_advantage_abs_p50": rollout.play_advantage_abs_p50,
            "play_advantage_abs_p90": rollout.play_advantage_abs_p90,
            "old_bid_entropy_mean": rollout.old_bid_entropy_mean,
            "old_play_entropy_mean": rollout.old_play_entropy_mean,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
        }
        if collection is not None:
            row.update(
                {
                    "collect_current_forward_sec": collection.current_forward_sec,
                    "collect_historical_forward_sec": collection.historical_forward_sec,
                    "collect_env_step_sec": collection.env_step_sec,
                    "collect_finalize_sec": collection.finalize_sec,
                    "collect_current_forward_calls": collection.current_forward_calls,
                    "collect_current_forward_rows": collection.current_forward_rows,
                    "collect_historical_forward_calls": collection.historical_forward_calls,
                    "collect_historical_forward_rows": collection.historical_forward_rows,
                    "collect_historical_policy_count": collection.historical_policy_count,
                    "collect_valid_event_tokens": collection.valid_event_tokens,
                    "collect_processed_event_tokens": collection.processed_event_tokens,
                    "collect_peak_device_memory_bytes": collection.peak_device_memory_bytes,
                    "collect_branch_root_hands": collection.branch_root_hands,
                    "collect_branch_roots_available": collection.branch_roots_available,
                    "collect_branch_roots_expanded": collection.branch_roots_expanded,
                    "collect_branch_bid_roots": collection.branch_bid_roots,
                    "collect_branch_play_roots": collection.branch_play_roots,
                    "collect_branch_terminal_rollouts": collection.branch_terminal_rollouts,
                    "collect_branch_self_decisions": collection.branch_self_decisions,
                    "collect_branch_historical_decisions": collection.branch_historical_decisions,
                    "collect_branch_forward_sec": collection.branch_forward_sec,
                }
            )
        if prediction is not None:
            row.update(
                {
                    "pred_samples": prediction.samples,
                    "pred_value_mae": prediction.value_mae,
                    "pred_value_mse": prediction.value_mse,
                    "pred_value_explained_variance": prediction.value_explained_variance,
                    "pred_oracle_value_explained_variance": (
                        prediction.oracle_value_explained_variance
                    ),
                    "pred_bid_value_explained_variance": (
                        prediction.bid_value_explained_variance
                    ),
                    "pred_play_value_explained_variance": (
                        prediction.play_value_explained_variance
                    ),
                    "pred_trick_implied_value_explained_variance": (
                        prediction.trick_implied_value_explained_variance
                    ),
                    "pred_bid_trick_implied_value_explained_variance": (
                        prediction.bid_trick_implied_value_explained_variance
                    ),
                    "pred_play_trick_implied_value_explained_variance": (
                        prediction.play_trick_implied_value_explained_variance
                    ),
                    "pred_trick_count_accuracy": prediction.trick_count_accuracy,
                    "pred_trick_count_true_prob": prediction.trick_count_true_prob,
                    "pred_trick_accuracy_bidtime": (
                        prediction.trick_count_accuracy_bidtime
                    ),
                    "pred_trick_accuracy_early": (
                        prediction.trick_count_accuracy_early
                    ),
                    "pred_trick_accuracy_mid": prediction.trick_count_accuracy_mid,
                    "pred_trick_accuracy_late": prediction.trick_count_accuracy_late,
                    "pred_trick_accuracy_p3": (
                        prediction.trick_count_accuracy_by_players.get(3, "")
                    ),
                    "pred_trick_accuracy_p4": (
                        prediction.trick_count_accuracy_by_players.get(4, "")
                    ),
                    "pred_trick_accuracy_p5": (
                        prediction.trick_count_accuracy_by_players.get(5, "")
                    ),
                    "pred_trick_accuracy_h3_5": (
                        prediction.trick_count_accuracy_by_hand_bucket.get("3_5", "")
                    ),
                    "pred_trick_accuracy_h6_8": (
                        prediction.trick_count_accuracy_by_hand_bucket.get("6_8", "")
                    ),
                    "pred_trick_accuracy_h9_10": (
                        prediction.trick_count_accuracy_by_hand_bucket.get("9_10", "")
                    ),
                    "pred_suit_presence_accuracy": prediction.suit_presence_accuracy,
                    "pred_suit_presence_brier": prediction.suit_presence_brier,
                    "pred_suit_presence_loss_early": prediction.suit_presence_loss_early,
                    "pred_suit_presence_loss_mid": prediction.suit_presence_loss_mid,
                    "pred_suit_presence_loss_late": prediction.suit_presence_loss_late,
                    "pred_suit_presence_accuracy_early": (
                        prediction.suit_presence_accuracy_early
                    ),
                    "pred_suit_presence_accuracy_mid": (
                        prediction.suit_presence_accuracy_mid
                    ),
                    "pred_suit_presence_accuracy_late": (
                        prediction.suit_presence_accuracy_late
                    ),
                    "pred_owner_accuracy": prediction.owner_accuracy,
                    "pred_owner_true_prob": prediction.owner_true_prob,
                    "pred_owner_brier": prediction.owner_brier,
                    "pred_owner_opponent_accuracy": (
                        prediction.owner_opponent_accuracy
                    ),
                    "pred_owner_opponent_true_prob": (
                        prediction.owner_opponent_true_prob
                    ),
                    "pred_owner_capacity_mae": (
                        prediction.owner_capacity_mae
                    ),
                    "pred_owner_capacity_max_error": (
                        prediction.owner_capacity_max_error
                    ),
                    "pred_owner_raw_capacity_mae": (
                        prediction.owner_raw_capacity_mae
                    ),
                    "pred_hit_prob_brier": prediction.hit_prob_brier,
                    "pred_bid_entropy": prediction.bid_entropy,
                    "pred_play_entropy": prediction.play_entropy,
                    "pred_bid_max_prob": prediction.bid_max_prob,
                    "pred_play_max_prob": prediction.play_max_prob,
                }
            )
        if evaluation is not None:
            row.update(
                {
                    "eval_rounds": evaluation.rounds,
                    "eval_macro_relative_reward": evaluation.macro_relative_reward,
                    "eval_ci_low": evaluation.relative_reward_ci_low,
                    "eval_ci_high": evaluation.relative_reward_ci_high,
                    "eval_macro_bid_hit_rate": evaluation.macro_bid_hit_rate,
                    "eval_mean_forward_passes": evaluation.mean_forward_passes,
                    "eval_elo_delta": evaluation.elo_delta,
                }
            )
        routing_by_phase = {
            stats.phase: stats
            for stats in (search_routing or [])
        }
        updates_by_phase = {
            stats.phase: stats
            for stats in (search_updates or [])
        }
        for phase in ("bid", "play"):
            routing = routing_by_phase.get(phase)
            if routing is not None:
                prefix = f"search_{phase}"
                row.update(
                    {
                        f"{prefix}_eligible": routing.eligible,
                        f"{prefix}_probed": routing.probed,
                        f"{prefix}_accepted_rate": routing.accepted_rate,
                        f"{prefix}_agreement": routing.argmax_agreement,
                        f"{prefix}_median_js": routing.median_target_js,
                        f"{prefix}_ci_low": routing.paired_ci_low,
                        f"{prefix}_sampler_infeasible_rejection_rate": (
                            routing.sampler_infeasible_rejection_rate
                        ),
                        f"{prefix}_sampler_failed_draw_rate": (
                            routing.sampler_failed_draw_rate
                        ),
                        f"{prefix}_gate_passed": routing.gate_passed,
                        f"{prefix}_routed": routing.routed,
                    }
                )
            search_update = updates_by_phase.get(phase)
            if search_update is not None:
                prefix = f"search_{phase}"
                row.update(
                    {
                        f"{prefix}_update_samples": search_update.samples,
                        f"{prefix}_update_loss": search_update.loss,
                        f"{prefix}_cross_entropy_loss": (
                            search_update.cross_entropy_loss
                        ),
                        f"{prefix}_entropy_floor_loss": (
                            search_update.entropy_floor_loss
                        ),
                        f"{prefix}_policy_entropy": (
                            search_update.policy_entropy
                        ),
                        f"{prefix}_target_entropy_floor": (
                            search_update.target_entropy_floor
                        ),
                        f"{prefix}_update_kl": search_update.kl,
                        f"{prefix}_max_stratum_kl": (
                            search_update.maximum_stratum_kl
                        ),
                        f"{prefix}_kl_cap": search_update.kl_cap,
                        f"{prefix}_regret_fraction": (
                            search_update.regret_matching_fraction
                        ),
                        f"{prefix}_backtracks": search_update.backtracks,
                        f"{prefix}_update_applied": search_update.applied,
                    }
                )
        return {field: row.get(field, "") for field in METRIC_FIELDS}


def training_config_snapshot(config: TrainingConfig) -> dict[str, Any]:
    return _jsonable(asdict(config))


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
