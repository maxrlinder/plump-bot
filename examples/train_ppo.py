"""Train the schema-v4 position-robust Plump policy."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_training_metrics import render_metrics_plot
from plump.evaluation import DealBank, evaluate_policy
from plump.modeling import ModelConfig
from plump.modeling.torch_model import PlumpTransformerModel
from plump.policies import HeuristicPolicy, ModelPolicy, RandomPolicy
from plump.search import SearchConfig
from plump.training import (
    CounterfactualSearchRouter,
    PPOTrainer,
    SearchTrustRegionUpdater,
    StratifiedReplayWindow,
    TrainingConfig,
    TrainingRunLogger,
    format_update_stats,
    training_config_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the schema-v4 balanced Plump agent.")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--player-counts", default="3,4,5")
    parser.add_argument("--hand-sizes", default="3,4,5,6,7,8,9,10")
    parser.add_argument("--rounds-per-configuration", type=int, default=16)
    parser.add_argument("--num-envs", type=int, default=384)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument(
        "--pipeline-rollouts",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--env-workers", type=int, default=0)
    parser.add_argument("--event-length-buckets", default="")
    parser.add_argument(
        "--batch-packing",
        choices=("torch", "numpy"),
        default="torch",
    )
    parser.add_argument(
        "--lean-rollout-forward",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--minibatch-size", type=int, default=1440)
    parser.add_argument("--microbatch-size", type=int, default=576)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--ppo-clip-eps", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument(
        "--branch-rollouts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Collect bounded recursive focal-action trees; every admitted "
            "branch finishes at a terminal round result."
        ),
    )
    parser.add_argument(
        "--branch-decision-budget-per-arm",
        type=int,
        default=0,
        help=(
            "Non-forced neural-decision budget for each of self/history; "
            "a player's deterministic final card is played for free."
        ),
    )
    parser.add_argument(
        "--branch-update-decisions-per-arm",
        type=int,
        default=2400,
        help="Maximum retained focal update rows per arm after tree reduction.",
    )
    parser.add_argument("--branch-max-replicates", type=int, default=4)
    parser.add_argument("--branch-max-active", type=int, default=768)
    parser.add_argument(
        "--branch-bid-max-actions",
        type=int,
        default=4,
        help=(
            "Maximum bid branches, including the sampled bid plus the "
            "highest-probability alternatives; 0 means exhaustive."
        ),
    )
    parser.add_argument("--branch-support-floor", type=float, default=0.0)
    parser.add_argument("--branch-target-temperature", type=float, default=1.0)
    parser.add_argument("--branch-advantage-clip", type=float, default=4.0)
    parser.add_argument("--branch-policy-coef", type=float, default=1.0)
    parser.add_argument(
        "--branch-policy-objective",
        choices=("ppo", "neurd"),
        default="ppo",
        help=(
            "Direct branch actor objective. 'neurd' applies signed "
            "counterfactual advantages to centered logits."
        ),
    )
    parser.add_argument("--branch-neurd-regret-coef", type=float, default=1.0)
    parser.add_argument("--branch-neurd-kl-coef", type=float, default=1.0)
    parser.add_argument("--branch-kl-cap", type=float, default=0.005)
    parser.add_argument(
        "--branch-exhaustive-hand-schedule",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Deal one matched self/history root for every configured hand "
            "size, sample player count and focal seat, and expand every "
            "focal action without rollout or update-row truncation."
        ),
    )
    parser.add_argument(
        "--branch-tree-decision-budget-per-arm",
        type=int,
        default=0,
        help=(
            "Maximum terminal rollout decisions spent per arm and matched "
            "deal; 0 leaves only the global per-arm cap."
        ),
    )
    parser.add_argument(
        "--branch-tree-update-decisions-per-arm",
        type=int,
        default=0,
        help=(
            "Maximum directly trained branch states per arm and matched "
            "deal; 0 leaves only the global update-row cap."
        ),
    )
    parser.add_argument("--trick-coef", type=float, default=0.1)
    parser.add_argument(
        "--belief-head-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Detach suit/final-trick belief supervision from the shared trunk "
            "and place the belief stack in its own optimizer group."
        ),
    )
    parser.add_argument("--owner-coef", type=float, default=0.05)
    # Head-only warmup after owner activation: iterations during which owner
    # gradients are detached from the trunk (persists across resumes).
    parser.add_argument("--owner-warmup-iters", type=int, default=0)
    parser.add_argument("--owner-capacity-coef", type=float, default=0.1)
    parser.add_argument(
        "--suit-presence-head",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--suit-coef", type=float, default=0.1)
    parser.add_argument("--position-baseline-decay", type=float, default=0.98)
    parser.add_argument("--explore-eps-bid", type=float, default=0.0)
    parser.add_argument("--explore-eps-play", type=float, default=0.0)
    # Per-arm exploration overrides as "bid,play" (e.g. "0.25,0.10"); arms
    # without an override use the global --explore-eps values.
    parser.add_argument("--explore-eps-self", default=None)
    parser.add_argument("--explore-eps-heuristic", default=None)
    parser.add_argument("--explore-eps-mixed", default=None)
    parser.add_argument("--explore-eps-historical", default=None)
    parser.add_argument("--explore-eps-explore-self", default=None)
    parser.add_argument("--explore-eps-explore-historical", default=None)
    parser.add_argument("--explore-eps-tempered-self", default=None)
    parser.add_argument("--explore-eps-tempered-historical", default=None)
    parser.add_argument("--explore-eps-capped-self", default=None)
    parser.add_argument("--explore-eps-capped-historical", default=None)
    parser.add_argument("--explore-eps-epsilon-self", default=None)
    parser.add_argument("--explore-eps-epsilon-historical", default=None)
    parser.add_argument(
        "--explore-argmax-cap",
        type=float,
        default=1.0,
        help=(
            "On capped arms, maximum probability allowed for the highest-logit "
            "legal action; probabilities already below it stay unchanged."
        ),
    )
    parser.add_argument(
        "--explore-argmax-rest-temp",
        type=float,
        default=1.0,
        help=(
            "Temperature for redistributing capped probability across legal "
            "non-argmax actions."
        ),
    )
    # Trajectory-diversity temperatures: this fraction of rounds in
    # --explore-temp-arms sample the current policy from softmax(logits/T)
    # (behavior-policy only; PPO stays importance-correct via the recorded
    # mixture logprob).
    parser.add_argument("--explore-temp-fraction", type=float, default=0.0)
    parser.add_argument("--explore-temp-bid", type=float, default=1.0)
    parser.add_argument("--explore-temp-play", type=float, default=1.0)
    parser.add_argument("--explore-temp-arms", default="self,mixed")
    # At most one policy-shaped non-argmax focal action per explore round,
    # injected with this probability; placement is uniform over decisions.
    parser.add_argument("--explore-uniform-round-prob", type=float, default=0.0)
    parser.add_argument(
        "--explore-single-deviation-temp",
        type=float,
        default=3.0,
        help=(
            "Temperature over legal non-argmax logits for the at-most-one "
            "deliberate deviation selected by --explore-uniform-round-prob."
        ),
    )
    # Shrink explore temperature + uniform probability on longer rounds by
    # (min_hand+1)/(hand+1) so per-round distortion stays roughly constant.
    parser.add_argument(
        "--explore-noise-hand-normalized",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    # Per-value sampling weights aligned with --player-counts / --hand-sizes
    # (normalized internally; empty = uniform cells).
    parser.add_argument("--player-count-weights", default="")
    parser.add_argument("--hand-size-weights", default="")
    parser.add_argument("--self-play-fraction", type=float, default=0.3)
    parser.add_argument("--heuristic-fraction", type=float, default=0.3)
    parser.add_argument("--mixed-fraction", type=float, default=0.3)
    parser.add_argument("--historical-fraction", type=float, default=0.1)
    # Explore arms: noised focal seat vs frozen raw opponents (current
    # weights / league snapshots); only the focal seat produces samples.
    parser.add_argument("--explore-self-fraction", type=float, default=0.0)
    parser.add_argument("--explore-historical-fraction", type=float, default=0.0)
    parser.add_argument("--tempered-self-fraction", type=float, default=0.0)
    parser.add_argument("--tempered-historical-fraction", type=float, default=0.0)
    parser.add_argument("--capped-self-fraction", type=float, default=0.0)
    parser.add_argument("--capped-historical-fraction", type=float, default=0.0)
    parser.add_argument("--epsilon-self-fraction", type=float, default=0.0)
    parser.add_argument("--epsilon-historical-fraction", type=float, default=0.0)
    parser.add_argument("--historical-checkpoint", action="append", default=[])
    parser.add_argument("--historical-max-snapshots", type=int, default=4)
    parser.add_argument("--league-temperature", type=float, default=2.0)
    parser.add_argument("--league-reward-decay", type=float, default=0.95)
    parser.add_argument(
        "--batched-league-sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--league-probe-fraction", type=float, default=0.10)
    parser.add_argument(
        "--league-meta-solver",
        choices=("softmax_ema", "regret_matching", "uniform"),
        default="regret_matching",
    )
    parser.add_argument(
        "--league-uniform-min-iteration",
        type=int,
        default=None,
        help=(
            "Uniform league: on every checkpoint save, resample the whole "
            "historical pool uniformly from saved checkpoints at or after "
            "this iteration (pool size = --historical-max-snapshots)."
        ),
    )
    parser.add_argument(
        "--league-archive-dir",
        type=Path,
        default=None,
        help=(
            "Optional immutable checkpoint archive included in uniform-league "
            "draws. New checkpoints still come from --checkpoint-dir."
        ),
    )
    parser.add_argument(
        "--league-resample-every",
        type=int,
        default=0,
        help=(
            "When positive, refresh a uniform league every N training "
            "iterations instead of coupling refreshes to checkpoint writes."
        ),
    )
    parser.add_argument("--league-eval-every", type=int, default=50)
    parser.add_argument("--league-eval-deals-per-configuration", type=int, default=2)
    parser.add_argument(
        "--oracle-critic",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--oracle-value-coef", type=float, default=0.5)
    parser.add_argument(
        "--mmd-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--mmd-coef", type=float, default=0.05)
    parser.add_argument("--mmd-magnet-decay", type=float, default=0.995)
    parser.add_argument(
        "--historical-current-snapshots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--warm-start-v2", type=Path, default=None)
    parser.add_argument("--warm-start-v3", type=Path, default=None)
    parser.add_argument("--trick-baseline", action="store_true")
    parser.add_argument("--training-mode", choices=("round", "game"), default="round")
    parser.add_argument("--game-schedule", default="")
    parser.add_argument("--min-cards", type=int, default=3)
    parser.add_argument("--max-cards", type=int, default=10)
    parser.add_argument("--games-per-player-seat", type=int, default=4)
    parser.add_argument(
        "--counterfactual-search",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--search-min-iteration", type=int, default=250)
    parser.add_argument("--search-ev-threshold", type=float, default=0.30)
    parser.add_argument("--search-states-per-phase", type=int, default=24)
    parser.add_argument("--search-bid-states", type=int, default=None)
    parser.add_argument("--search-play-states", type=int, default=None)
    parser.add_argument(
        "--search-opponent-arms",
        default="",
        help="Optional comma-separated rollout arms eligible for search probes.",
    )
    parser.add_argument(
        "--search-assume-value-ready",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Seed the value-readiness history for a mature resumed policy; "
            "search's own stability and improvement gates still apply."
        ),
    )
    parser.add_argument(
        "--search-every",
        type=int,
        default=1,
        help="Run search and its distillation update every N iterations.",
    )
    parser.add_argument("--search-min-determinizations", type=int, default=4)
    parser.add_argument("--search-max-determinizations", type=int, default=32)
    parser.add_argument("--search-forward-pass-budget", type=int, default=2_000)
    parser.add_argument("--search-exact-tricks-remaining", type=int, default=3)
    parser.add_argument(
        "--search-use-owner-beliefs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--search-replay-capacity", type=int, default=50_000)
    parser.add_argument("--search-replay-max-age", type=int, default=250)
    parser.add_argument("--search-lr", type=float, default=1e-4)
    parser.add_argument("--search-minibatch-size", type=int, default=256)
    parser.add_argument("--search-entropy-floor-coef", type=float, default=0.002)
    parser.add_argument("--search-kl-cap-start", type=float, default=0.005)
    parser.add_argument("--search-kl-cap-end", type=float, default=0.003)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-deals-per-configuration", type=int, default=4)
    parser.add_argument("--eval-bootstrap-samples", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=384)
    parser.add_argument("--eval-opponent", choices=("heuristic", "random"), default="heuristic")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--save-every-minutes", type=float, default=30.0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/v4"))
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=704)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=11)
    parser.add_argument("--d-ff", type=int, default=2560)
    parser.add_argument("--context-hidden-dim", type=int, default=512)
    parser.add_argument("--owner-sinkhorn-iterations", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--diag-every", type=int, default=5)
    parser.add_argument("--diag-samples", type=int, default=2048)
    parser.add_argument("--diag-batch-size", type=int, default=256)
    parser.add_argument("--plot-every", type=int, default=5)
    return parser.parse_args()


def _uniform_league_draw(
    checkpoint_dir: Path,
    min_iteration: int,
    pool_size: int,
    seed: int,
    archive_dir: Path | None = None,
) -> list[Path]:
    """Uniform draw over archived and newly saved eligible checkpoints.

    Seeded by (run seed, current iteration) so a resumed run redraws the same
    pool at the same point, but every refresh is an independent sample.
    """

    candidates_by_iteration: dict[int, Path] = {}
    for directory in (archive_dir, checkpoint_dir):
        if directory is None:
            continue
        for path in directory.glob("plump_v4_iter_*.pt"):
            iteration = int(path.stem.rsplit("_", 1)[1])
            if iteration >= min_iteration:
                # Prefer a newly written run checkpoint when an archive
                # contains the same iteration.
                candidates_by_iteration[iteration] = path
    candidates = [
        candidates_by_iteration[iteration]
        for iteration in sorted(candidates_by_iteration)
    ]
    if len(candidates) <= pool_size:
        return candidates
    return random.Random(seed).sample(candidates, pool_size)


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    args = parse_args()
    player_counts = _csv_ints(args.player_counts)
    hand_sizes = _csv_ints(args.hand_sizes)
    game_schedule = _csv_ints(args.game_schedule)
    event_length_buckets = _csv_ints(args.event_length_buckets)
    initialization_paths = [
        args.resume_from,
        args.warm_start_v2,
        args.warm_start_v3,
    ]
    if sum(path is not None for path in initialization_paths) > 1:
        raise ValueError(
            "--resume-from, --warm-start-v2, and --warm-start-v3 "
            "are mutually exclusive."
        )
    search_enabled = (
        args.counterfactual_search
        and args.training_mode == "round"
    )
    if args.search_every < 1:
        raise ValueError("--search-every must be positive.")
    if args.search_min_determinizations < 1:
        raise ValueError("--search-min-determinizations must be positive.")
    if args.search_max_determinizations < args.search_min_determinizations:
        raise ValueError(
            "--search-max-determinizations must be at least the minimum."
        )
    if args.search_forward_pass_budget < 1:
        raise ValueError("--search-forward-pass-budget must be positive.")
    model_config = ModelConfig(
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        context_hidden_dim=args.context_hidden_dim,
        owner_sinkhorn_iterations=args.owner_sinkhorn_iterations,
        dropout=args.dropout,
        oracle_critic=args.oracle_critic,
        suit_presence_head=args.suit_presence_head,
    )
    train_config = TrainingConfig(
        player_counts=player_counts,
        hand_sizes=hand_sizes,
        rounds_per_configuration=args.rounds_per_configuration,
        games_per_player_seat=args.games_per_player_seat,
        num_envs=args.num_envs,
        ppo_epochs=args.ppo_epochs,
        target_kl=args.target_kl,
        pipeline_rollouts=args.pipeline_rollouts,
        env_workers=args.env_workers,
        event_length_buckets=event_length_buckets,
        batch_packing=args.batch_packing,
        lean_rollout_forward=args.lean_rollout_forward,
        minibatch_size=args.minibatch_size,
        microbatch_size=args.microbatch_size or None,
        learning_rate=args.lr,
        ppo_clip_eps=args.ppo_clip_eps,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        branch_rollouts=args.branch_rollouts,
        branch_decision_budget_per_arm=(
            args.branch_decision_budget_per_arm
        ),
        branch_update_decision_budget_per_arm=(
            args.branch_update_decisions_per_arm
        ),
        branch_max_replicates=args.branch_max_replicates,
        branch_max_active=args.branch_max_active,
        branch_bid_max_actions=args.branch_bid_max_actions,
        branch_support_floor=args.branch_support_floor,
        branch_target_temperature=args.branch_target_temperature,
        branch_advantage_clip=args.branch_advantage_clip,
        branch_policy_coef=args.branch_policy_coef,
        branch_policy_objective=args.branch_policy_objective,
        branch_neurd_regret_coef=args.branch_neurd_regret_coef,
        branch_neurd_kl_coef=args.branch_neurd_kl_coef,
        branch_kl_cap=args.branch_kl_cap,
        branch_exhaustive_hand_schedule=(
            args.branch_exhaustive_hand_schedule
        ),
        branch_tree_decision_budget_per_arm=(
            args.branch_tree_decision_budget_per_arm
        ),
        branch_tree_update_decisions_per_arm=(
            args.branch_tree_update_decisions_per_arm
        ),
        trick_coef=args.trick_coef,
        belief_head_only=args.belief_head_only,
        owner_coef=args.owner_coef,
        owner_warmup_iterations=args.owner_warmup_iters,
        owner_capacity_coef=args.owner_capacity_coef,
        suit_coef=args.suit_coef,
        max_grad_norm=args.max_grad_norm,
        explore_eps_bid=args.explore_eps_bid,
        explore_eps_play=args.explore_eps_play,
        explore_eps_by_arm={
            arm: tuple(float(part) for part in override.split(","))
            for arm, override in (
                ("self", args.explore_eps_self),
                ("heuristic", args.explore_eps_heuristic),
                ("mixed", args.explore_eps_mixed),
                ("historical", args.explore_eps_historical),
                ("explore_self", args.explore_eps_explore_self),
                ("explore_historical", args.explore_eps_explore_historical),
                ("tempered_self", args.explore_eps_tempered_self),
                ("tempered_historical", args.explore_eps_tempered_historical),
                ("capped_self", args.explore_eps_capped_self),
                ("capped_historical", args.explore_eps_capped_historical),
                ("epsilon_self", args.explore_eps_epsilon_self),
                ("epsilon_historical", args.explore_eps_epsilon_historical),
            )
            if override
        },
        explore_argmax_cap=args.explore_argmax_cap,
        explore_argmax_rest_temperature=args.explore_argmax_rest_temp,
        explore_temperature_fraction=args.explore_temp_fraction,
        explore_temperature_bid=args.explore_temp_bid,
        explore_temperature_play=args.explore_temp_play,
        explore_temperature_arms=tuple(
            arm.strip()
            for arm in args.explore_temp_arms.split(",")
            if arm.strip()
        ),
        explore_uniform_round_probability=args.explore_uniform_round_prob,
        explore_single_deviation_temperature=(
            args.explore_single_deviation_temp
        ),
        explore_noise_hand_normalized=args.explore_noise_hand_normalized,
        player_count_weights=_csv_floats(args.player_count_weights),
        hand_size_weights=_csv_floats(args.hand_size_weights),
        position_baseline_decay=args.position_baseline_decay,
        self_play_fraction=args.self_play_fraction,
        heuristic_fraction=args.heuristic_fraction,
        mixed_fraction=args.mixed_fraction,
        historical_fraction=args.historical_fraction,
        explore_self_fraction=args.explore_self_fraction,
        explore_historical_fraction=args.explore_historical_fraction,
        tempered_self_fraction=args.tempered_self_fraction,
        tempered_historical_fraction=args.tempered_historical_fraction,
        capped_self_fraction=args.capped_self_fraction,
        capped_historical_fraction=args.capped_historical_fraction,
        epsilon_self_fraction=args.epsilon_self_fraction,
        epsilon_historical_fraction=args.epsilon_historical_fraction,
        historical_checkpoint_paths=tuple(args.historical_checkpoint),
        historical_max_snapshots=args.historical_max_snapshots,
        league_temperature=args.league_temperature,
        league_reward_decay=args.league_reward_decay,
        batched_league_sampling=args.batched_league_sampling,
        league_probe_fraction=args.league_probe_fraction,
        league_meta_solver=args.league_meta_solver,
        league_eval_every=args.league_eval_every,
        league_eval_deals_per_configuration=args.league_eval_deals_per_configuration,
        oracle_critic=args.oracle_critic,
        oracle_value_coef=args.oracle_value_coef,
        mmd_enabled=args.mmd_enabled,
        mmd_coef=args.mmd_coef,
        mmd_magnet_decay=args.mmd_magnet_decay,
        trick_baseline=args.trick_baseline,
        training_mode=args.training_mode,
        game_schedule=game_schedule,
        min_cards=args.min_cards,
        max_cards=args.max_cards,
        precision=args.precision,
        seed=args.seed,
        device=args.device,
        model_config=model_config,
    )
    trainer = PPOTrainer(PlumpTransformerModel(model_config), train_config)
    resume_info = (
        trainer.load_checkpoint(args.resume_from, load_optimizer=args.resume_optimizer)
        if args.resume_from
        else None
    )
    warm_start_info = (
        trainer.warm_start_v2(args.warm_start_v2)
        if args.warm_start_v2
        else (
            trainer.warm_start_v3(args.warm_start_v3)
            if args.warm_start_v3
            else None
        )
    )
    start_iteration = int(resume_info["iteration"] or 0) if resume_info else 0
    if start_iteration >= args.iterations:
        raise ValueError(
            f"Checkpoint iteration {start_iteration} has already reached "
            f"the requested total of {args.iterations} iterations."
        )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if (
        args.league_archive_dir is not None
        and not args.league_archive_dir.is_dir()
    ):
        raise ValueError(
            f"League archive directory does not exist: "
            f"{args.league_archive_dir}"
        )
    if args.league_resample_every < 0:
        raise ValueError("--league-resample-every must be nonnegative.")
    if args.league_uniform_min_iteration is not None:
        # Replace whatever pool the checkpoint restored (e.g. an old
        # sliding-window league) with a uniform draw before the first
        # collection, so the new sampling regime applies from iteration one.
        initial_pool = _uniform_league_draw(
            args.checkpoint_dir,
            args.league_uniform_min_iteration,
            args.historical_max_snapshots,
            args.seed * 1_000_003 + start_iteration,
            args.league_archive_dir,
        )
        if initial_pool:
            trainer.replace_historical_snapshots(initial_pool)
    log_dir = args.log_dir or args.checkpoint_dir
    logger = TrainingRunLogger(log_dir)
    search_router = (
        CounterfactualSearchRouter(
            trainer.model,
            device=trainer.device,
            precision=args.precision,
            minimum_iteration=args.search_min_iteration,
            explained_variance_threshold=args.search_ev_threshold,
            states_per_phase=args.search_states_per_phase,
            bid_states_per_iteration=args.search_bid_states,
            play_states_per_iteration=args.search_play_states,
            replay_capacity=args.search_replay_capacity,
            replay_max_age=args.search_replay_max_age,
            search_config=SearchConfig(
                min_determinizations=args.search_min_determinizations,
                max_determinizations=args.search_max_determinizations,
                forward_pass_budget=args.search_forward_pass_budget,
                exact_tricks_remaining=args.search_exact_tricks_remaining,
                use_owner_beliefs=args.search_use_owner_beliefs,
                seed=args.seed + 45_000,
            ),
            opponent_arms=tuple(
                arm.strip()
                for arm in args.search_opponent_arms.split(",")
                if arm.strip()
            ),
            assume_value_ready=args.search_assume_value_ready,
            event_length_buckets=event_length_buckets,
            batch_packing=args.batch_packing,
            lean_action_forward=args.lean_rollout_forward,
            seed=args.seed + 40_000,
        )
        if search_enabled
        else None
    )
    search_updater = (
        SearchTrustRegionUpdater(
            trainer.model,
            device=trainer.device,
            learning_rate=args.search_lr,
            minibatch_size=args.search_minibatch_size,
            max_grad_norm=args.max_grad_norm,
            entropy_floor_coef=args.search_entropy_floor_coef,
            kl_cap_start=args.search_kl_cap_start,
            kl_cap_end=args.search_kl_cap_end,
            seed=args.seed + 50_000,
        )
        if search_enabled
        else None
    )
    search_replay_path = args.checkpoint_dir / "search_replay.pt"
    if search_router is not None and search_replay_path.is_file():
        search_router.replay = StratifiedReplayWindow.load(search_replay_path)
        print(
            f"loaded search replay rows={len(search_router.replay)} "
            f"strata={search_router.replay.strata}"
        )
    if args.env_workers > 0 and args.training_mode == "round":
        from plump.training.env_workers import EnvWorkerPool

        trainer.env_pool = EnvWorkerPool(
            num_workers=args.env_workers,
            model_config=model_config,
            include_game_context=train_config.include_game_context,
            num_envs=args.num_envs,
        )
    logger.write_config(
        {
            "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
            "args": vars(args),
            "device": str(trainer.device),
            "parameter_count": sum(parameter.numel() for parameter in trainer.model.parameters()),
            "resume": resume_info,
            "warm_start": warm_start_info,
            "counterfactual_search_enabled": search_enabled,
            "mps_environment": {
                key: os.environ.get(key)
                for key in (
                    "PYTORCH_MPS_FAST_MATH",
                    "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
                    "PYTORCH_MPS_LOW_WATERMARK_RATIO",
                    "PYTORCH_MPS_PREFER_METAL",
                )
            },
            "training_config": training_config_snapshot(train_config),
        }
    )
    print(
        f"device={trainer.device} configurations={len(train_config.specs)} "
        f"rounds_per_batch={train_config.rounds_per_batch} "
        f"precision={train_config.precision} start_iteration={start_iteration} "
        f"search={search_enabled} log_dir={log_dir}"
    )

    eval_bank = DealBank.generate(
        player_counts=player_counts,
        hand_sizes=hand_sizes,
        deals_per_configuration=args.eval_deals_per_configuration,
        seed=args.seed + 20_000,
    )
    eval_opponent = HeuristicPolicy() if args.eval_opponent == "heuristic" else RandomPolicy(args.seed + 30_000)
    best_eval = float("-inf")
    started_at = time.perf_counter()
    last_checkpoint_at = started_at
    collector = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="rollout-collector")
        if args.pipeline_rollouts
        else None
    )
    pending_collection: Future | None = None
    buffer = None

    for iteration in range(start_iteration + 1, args.iterations + 1):
        iteration_start = time.perf_counter()
        if collector is None or buffer is None:
            # Sequential mode, or priming the pipeline on the first iteration.
            buffer = trainer.collect_rollouts(iteration=iteration)
        collect_sec = trainer.last_collect_sec
        if collector is not None and iteration < args.iterations:
            # Collect the next batch from the frozen rollout snapshot while
            # update() trains the live model; data is one policy-version stale
            # and target-kl budgets the divergence from the collecting policy.
            pending_collection = collector.submit(
                trainer.collect_rollouts,
                iteration=iteration + 1,
            )
        rollout_stats = trainer.summarize_rollout(buffer)

        search_due = (
            search_router is not None
            and (iteration - start_iteration) % args.search_every == 0
        )
        search_routing = (
            search_router.route(buffer, iteration=iteration)
            if search_due and search_router is not None
            else []
        )
        update_start = time.perf_counter()
        update_stats = trainer.update(buffer)
        search_updates = []
        if search_due and search_router is not None and search_updater is not None:
            enabled_search_phases = (
                phase
                for phase in ("bid", "play")
                if search_router.states_per_phase[phase] > 0
            )
            for phase in enabled_search_phases:
                search_updates.append(
                    search_updater.update(
                        search_router.replay,
                        phase=phase,
                        iteration=iteration,
                        regret_matching_fraction=(
                            search_router.regret_matching_fraction(phase)
                        ),
                    )
                )
        update_sec = time.perf_counter() - update_start

        diagnostics_sec = 0.0
        prediction_stats = None
        if args.diag_every > 0 and iteration % args.diag_every == 0:
            diagnostics_start = time.perf_counter()
            if trainer.device.type == "mps":
                torch.mps.empty_cache()
            prediction_stats = trainer.compute_prediction_stats(
                buffer,
                max_samples=args.diag_samples,
                minibatch_size=args.diag_batch_size,
            )
            if search_router is not None:
                search_router.update_diagnostics(
                    bid_explained_variance=(
                        prediction_stats.bid_value_explained_variance
                    ),
                    play_explained_variance=(
                        prediction_stats.play_value_explained_variance
                    ),
                )
            diagnostics_sec = time.perf_counter() - diagnostics_start

        branch_collection = ""
        if args.branch_rollouts:
            collection = trainer.last_collection_stats
            branch_collection = (
                f" roots={collection.branch_root_hands} "
                f"leaves={collection.branch_terminal_rollouts} "
                f"branch_decisions="
                f"{collection.branch_self_decisions}/"
                f"{collection.branch_historical_decisions} "
                f"collect_s={collect_sec:.2f}"
            )
        print(
            f"iter={iteration} bid_hit={rollout_stats.bid_hit_rate:.4f} "
            f"{format_update_stats(update_stats)}{branch_collection}"
        )
        for routing in search_routing:
            if routing.eligible or routing.gate_passed:
                print(
                    f"search phase={routing.phase} probed={routing.probed} "
                    f"accepted={routing.accepted_rate:.3f} "
                    f"ci_low={routing.paired_ci_low:.4f} "
                    f"sampler_reject={routing.sampler_infeasible_rejection_rate:.4f} "
                    f"gate={routing.gate_passed} routed={routing.routed}"
                )

        eval_sec = 0.0
        evaluation = None
        if args.eval_every > 0 and iteration % args.eval_every == 0:
            eval_start = time.perf_counter()
            if trainer.device.type == "mps":
                torch.mps.empty_cache()
            candidate = ModelPolicy(
                trainer.model,
                device=trainer.device,
                greedy=True,
                precision=args.precision,
                event_length_buckets=train_config.event_length_buckets,
                batch_packing=train_config.batch_packing,
                lean_action_forward=train_config.lean_rollout_forward,
                name=f"iteration-{iteration}",
            )
            evaluation = evaluate_policy(
                candidate,
                eval_opponent,
                eval_bank,
                bootstrap_samples=args.eval_bootstrap_samples,
                seed=args.seed + iteration,
                batch_size=args.eval_batch_size,
            )
            eval_sec = time.perf_counter() - eval_start
            print(
                f"eval rounds={evaluation.rounds} macro_rel={evaluation.macro_relative_reward:.4f} "
                f"ci=[{evaluation.relative_reward_ci_low:.4f},{evaluation.relative_reward_ci_high:.4f}] "
                f"bid_hit={evaluation.macro_bid_hit_rate:.4f}"
            )
            if evaluation.macro_relative_reward > best_eval:
                best_eval = evaluation.macro_relative_reward
                best_path = args.checkpoint_dir / "best.pt"
                trainer.save_checkpoint(best_path, iteration=iteration, extra={"evaluation": evaluation})

        checkpoint_sec = 0.0
        checkpoint_path = None
        initial_checkpoint_due = iteration == start_iteration + 1
        iteration_checkpoint_due = args.save_every > 0 and iteration % args.save_every == 0
        wall_checkpoint_due = (
            args.save_every_minutes > 0
            and time.perf_counter() - last_checkpoint_at >= args.save_every_minutes * 60.0
        )
        if initial_checkpoint_due or iteration_checkpoint_due or wall_checkpoint_due:
            checkpoint_start = time.perf_counter()
            checkpoint_path = args.checkpoint_dir / f"plump_v4_iter_{iteration:05d}.pt"
            trainer.save_checkpoint(checkpoint_path, iteration=iteration, extra={"evaluation": evaluation})
            if search_router is not None:
                search_router.replay.save(
                    search_replay_path,
                    gate_report_path=logger.events_path,
                )
            checkpoint_sec = time.perf_counter() - checkpoint_start
            last_checkpoint_at = time.perf_counter()

        # Pipeline sync point: league membership, payoff refreshes, and the
        # rollout snapshot must only change while no collection is in flight.
        next_buffer = None
        if pending_collection is not None:
            next_buffer = pending_collection.result()
            pending_collection = None
        uniform_resample_due = (
            args.league_uniform_min_iteration is not None
            and (
                (
                    args.league_resample_every <= 0
                    and checkpoint_path is not None
                )
                or (
                    args.league_resample_every > 0
                    and (iteration - start_iteration)
                    % args.league_resample_every
                    == 0
                )
            )
        )
        if uniform_resample_due:
            pool = _uniform_league_draw(
                args.checkpoint_dir,
                args.league_uniform_min_iteration,
                args.historical_max_snapshots,
                args.seed * 1_000_003 + iteration,
                args.league_archive_dir,
            )
            if pool:
                trainer.replace_historical_snapshots(pool)
        elif (
            checkpoint_path is not None
            and args.league_uniform_min_iteration is None
            and args.historical_current_snapshots
        ):
            trainer.add_historical_checkpoint(
                checkpoint_path,
                mirrors_current=True,
            )
        trainer.refresh_league_payoffs(iteration=iteration)
        trainer.sync_rollout_model()

        timings = {
            "collect_sec": collect_sec,
            "update_sec": update_sec,
            "diagnostics_sec": diagnostics_sec,
            "eval_sec": eval_sec,
            "checkpoint_sec": checkpoint_sec,
            "iteration_sec": time.perf_counter() - iteration_start,
        }
        logger.log_iteration(
            iteration=iteration,
            elapsed_sec=time.perf_counter() - started_at,
            timings=timings,
            update=update_stats,
            rollout=rollout_stats,
            collection=trainer.last_collection_stats,
            prediction=prediction_stats,
            evaluation=evaluation,
            checkpoint_path=checkpoint_path,
            search_routing=search_routing,
            search_updates=search_updates,
            league=trainer.league_reward_ema,
            league_mixture=trainer.league_meta_mixture,
            league_payoffs=trainer.league_payoffs,
        )

        if args.plot_every > 0 and iteration % args.plot_every == 0:
            try:
                render_metrics_plot(
                    metrics_path=logger.metrics_path,
                    output_path=log_dir / "metrics.png",
                    smooth=50,
                    diagnostic_smooth=50,
                )
            except Exception as error:  # pragma: no cover - plotting is best-effort
                print(f"metrics plot failed: {error}")

        buffer = next_buffer

    if collector is not None:
        collector.shutdown(wait=True)
    if trainer.env_pool is not None:
        trainer.env_pool.close()


if __name__ == "__main__":
    main()
