"""Versioned training configuration loading for the schema-v6 CLI."""

from __future__ import annotations

import copy
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plump.seq.config import (
    BranchBudgetConfig,
    BranchRuleConfig,
    GameScheduleCell,
    RolloutOptions,
    SeqModelConfig,
    SeqTrainingConfig,
    build_branch_rate_table,
    build_position_balanced_schedule,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKOUT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train.toml"
PACKAGED_CONFIG_PATH = Path(__file__).with_name("train.toml")
DEFAULT_CONFIG_PATH = (
    CHECKOUT_CONFIG_PATH
    if CHECKOUT_CONFIG_PATH.is_file()
    else PACKAGED_CONFIG_PATH
)


@dataclass(frozen=True)
class ResolvedTraining:
    raw: dict[str, Any]
    model: SeqModelConfig
    training: SeqTrainingConfig

    @property
    def run(self) -> dict[str, Any]:
        return self.raw["run"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.raw["evaluation"]


def load_training_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    overrides: list[str] | None = None,
) -> ResolvedTraining:
    """Load TOML and apply typed ``section.key=value`` overrides."""

    path = Path(path)
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    raw = copy.deepcopy(raw)
    for override in overrides or []:
        _apply_override(raw, override)
    return resolve_training_config(raw)


def resolve_training_config(raw: dict[str, Any]) -> ResolvedTraining:
    required = {"run", "model", "training", "rollout", "evaluation"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"Configuration is missing sections: {sorted(missing)}")

    model_raw = raw["model"]
    model = SeqModelConfig(
        d_model=int(model_raw["d_model"]),
        n_layers=int(model_raw["n_layers"]),
        n_heads=int(model_raw["n_heads"]),
        n_kv_heads=int(model_raw["n_kv_heads"]),
        d_ff=int(model_raw["d_ff"]),
        trick_win_token=bool(model_raw["trick_win_token"]),
        turn_token=str(model_raw["turn_token"]),
    )

    training_raw = raw["training"]
    hand_sizes = tuple(int(value) for value in training_raw["hand_sizes"])
    player_counts = tuple(int(value) for value in training_raw["player_counts"])
    player_weights = tuple(
        float(value) for value in training_raw["player_count_weights"]
    )
    games_per_cell = int(training_raw.get("games_per_cell", 0))
    if games_per_cell > 0:
        cells = tuple(
            GameScheduleCell(hand_size=hand_size)
            for hand_size in hand_sizes
            for _ in range(games_per_cell)
        )
    else:
        deals_per_shape = int(training_raw.get("deals_per_shape", 0))
        cells = build_position_balanced_schedule(
            hand_sizes=hand_sizes,
            player_counts=player_counts,
            repeats=int(training_raw["schedule_repeats"]),
            deals_per_shape=deals_per_shape or None,
        )

    rate_table = build_branch_rate_table(
        float(training_raw["reference_rate"]),
        exhaustive_until=int(training_raw["exhaustive_until"]),
        hand_sizes=hand_sizes,
        player_counts=player_counts,
        player_exponent=float(
            training_raw["branch_rate_player_exponent"]
        ),
    )
    rollout_raw = raw["rollout"]
    run_raw = raw["run"]
    evaluation_raw = raw["evaluation"]
    training = SeqTrainingConfig(
        schedule_cells=cells,
        player_counts=player_counts,
        player_count_weights=player_weights,
        branch_rule=BranchRuleConfig(
            bid_mode=str(training_raw.get("bid_mode", "top_k")),
            bid_top_k=int(training_raw["bid_top_k"]),
            play_mode=str(training_raw["play_mode"]),
            play_top_k=int(training_raw["play_top_k"]),
        ),
        branch_budget=BranchBudgetConfig(branch_rate_by_shape=rate_table),
        rollout=RolloutOptions(
            auto_deals_per_batch=bool(
                rollout_raw["auto_deals_per_batch"]
            ),
            deals_per_batch=int(rollout_raw.get("deals_per_batch", 1)),
            cache_budget_gb=float(rollout_raw["cache_budget_gb"]),
            max_cache_rows=int(rollout_raw["max_cache_rows"]),
            historical_arm=str(rollout_raw["historical_arm"]),
            bid_position_mode=str(
                rollout_raw.get("bid_position_mode", "cycle")
            ),
        ),
        learning_rate=float(training_raw["learning_rate"]),
        lr_warmup_updates=int(training_raw["lr_warmup_updates"]),
        epochs=int(training_raw["epochs"]),
        microbatch_positions=int(training_raw["microbatch_positions"]),
        policy_coef=float(training_raw["policy_coef"]),
        policy_kl_cap=float(training_raw["policy_kl_cap"]),
        neurd_advantage_clip=float(
            training_raw["neurd_advantage_clip"]
        ),
        neurd_inclusion_exponent=float(
            training_raw["neurd_inclusion_exponent"]
        ),
        value_coef=float(training_raw["value_coef"]),
        suit_coef=float(training_raw["suit_coef"]),
        bid_hit_coef=float(training_raw["bid_hit_coef"]),
        trick_coef=float(training_raw["trick_coef"]),
        entropy_coef=float(training_raw["entropy_coef"]),
        kv_dtype=str(rollout_raw["kv_dtype"]),
        snapshot_every=int(run_raw["checkpoint_every"]),
        league_max_snapshots=int(training_raw["league_max_snapshots"]),
        league_min_iteration=int(training_raw["league_min_iteration"]),
        eval_every=int(evaluation_raw["every"]),
        checkpoint_every=int(run_raw["checkpoint_every"]),
        seed=int(training_raw["seed"]),
    )
    model.validate()
    training.validate()
    return ResolvedTraining(raw=raw, model=model, training=training)


def config_diff(
    recorded: dict[str, Any],
    requested: dict[str, Any],
) -> list[str]:
    left = _flatten(recorded)
    right = _flatten(requested)
    lines = []
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            lines.append(
                f"{key}: recorded={left.get(key)!r} requested={right.get(key)!r}"
            )
    return lines


def dump_toml(raw: dict[str, Any]) -> str:
    """Serialize the simple scalar/list configuration shape deterministically."""

    lines: list[str] = []
    for section, values in raw.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines)


def _apply_override(raw: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Override must be section.key=value: {override}")
    dotted, source = override.split("=", 1)
    parts = dotted.split(".")
    if len(parts) != 2 or parts[0] not in raw or parts[1] not in raw[parts[0]]:
        raise ValueError(f"Unknown configuration key: {dotted}")
    try:
        value = tomllib.loads(f"value = {source}")["value"]
    except tomllib.TOMLDecodeError:
        value = source
    raw[parts[0]][parts[1]] = value


def _flatten(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{section}.{key}": value
        for section, values in raw.items()
        for key, value in values.items()
    }


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value: {value!r}")
