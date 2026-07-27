"""Train the schema-v6 autoregressive sequence model (KV-cache pipeline).

Example (Mac bring-up):
    .venv/bin/python examples/train_seq.py \
        --iterations 2000 --reference-rate 0.5 \
        --checkpoint-dir checkpoints/seq_v6_run1 --log-dir logs/seq_v6_run1

The default schedule covers every (player count, hand size, bidding position)
once per update. Branching is placed across the whole game, exhaustively up to
``--exhaustive-until`` cards (where it is nearly free) and tapering to
``--reference-rate`` at 10 cards, since the rate compounds over game length.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

from plump.evaluation import DealBank, evaluate_policy
from plump.policies import HeuristicPolicy
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
from plump.seq.model import SeqPlumpModel
from plump.seq.policy import SeqModelPolicy, best_seq_device
from plump.seq.trainer import SeqTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-kv-heads", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=768)
    parser.add_argument(
        "--no-trick-win-token",
        action="store_true",
        help="drop the per-trick winner token; the winner is derivable from "
        "the plays and is already named by the last play's next-actor slot",
    )
    parser.add_argument(
        "--turn-token",
        choices=["off", "bid", "all"],
        default="bid",
        help="pause token before actions. Default 'bid': the bid is the one "
        "decision worth extra serial compute, and the token names the actor "
        "via its player embedding. 'all' also puts one before every play, "
        "which costs 5x more than it sounds -- the sequence carries all P "
        "players' actions, so at 5p/10c it is +55 tokens, not +11.",
    )

    parser.add_argument("--hand-sizes", default="3,4,5,6,7,8,9,10")
    parser.add_argument("--player-counts", default="3,4,5")
    parser.add_argument("--player-count-weights", default="2,3,4")
    parser.add_argument(
        "--schedule-repeats",
        type=int,
        default=1,
        help="passes over (players x cards x bidding position) per update",
    )
    parser.add_argument(
        "--games-per-cell",
        type=int,
        default=None,
        help="legacy flat schedule: N deals per hand size, players sampled",
    )

    parser.add_argument(
        "--reference-rate",
        type=float,
        default=0.5,
        help="branch rate at 10 cards; shorter games taper up to 1.0",
    )
    parser.add_argument("--branch-rate-player-exponent", type=float, default=0.0)
    parser.add_argument(
        "--exhaustive-until",
        type=int,
        default=7,
        help="branch every eligible decision at or below this hand size",
    )
    parser.add_argument("--bid-top-k", type=int, default=4)
    parser.add_argument(
        "--play-mode",
        choices=[
            "all_legal",
            "top_k",
            "top_k_plus_random",
            "sample_k",
            "sample_k_plus_uniform",
            "none",
        ],
        default="sample_k_plus_uniform",
    )
    parser.add_argument("--play-top-k", type=int, default=3)
    parser.add_argument(
        "--max-cache-rows",
        type=int,
        default=65536,
        help="hard KV row ceiling; the backstop behind the branch rate",
    )
    parser.add_argument("--cache-budget-gb", type=float, default=10.0)
    parser.add_argument("--auto-deals-per-batch", action="store_true", default=True)

    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--microbatch-positions", type=int, default=16384)
    parser.add_argument(
        "--branch-objective", choices=["neurd", "ppo"], default="neurd"
    )
    parser.add_argument("--branch-policy-coef", type=float, default=1.0)
    parser.add_argument("--spine-policy-coef", type=float, default=1.0)
    parser.add_argument("--branch-kl-cap", type=float, default=0.005)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--owner-coef", type=float, default=0.25)
    parser.add_argument("--suit-coef", type=float, default=0.1)
    parser.add_argument("--trick-coef", type=float, default=0.25)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--kv-dtype", choices=["fp32", "fp16"], default="fp32")

    parser.add_argument("--snapshot-every", type=int, default=200)
    parser.add_argument("--league-max-snapshots", type=int, default=8)
    parser.add_argument("--league-min-iteration", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-deals", type=int, default=6)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--checkpoint-dir", default="checkpoints/seq_v6")
    parser.add_argument("--log-dir", default="logs/seq_v6")
    parser.add_argument("--resume-from", default=None)
    return parser.parse_args()


def _csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part)


def _csv_floats(text: str) -> tuple[float, ...]:
    return tuple(float(part) for part in text.split(",") if part)


LOG_COLUMNS = [
    "iteration",
    "collect_sec",
    "update_sec",
    "trees",
    "leaves",
    "decisions",
    "spine_rows",
    "branch_rows",
    "positions",
    "bid_hit_rate",
    "reward_self",
    "reward_historical",
    "spine_entropy",
    "loss_spine",
    "loss_branch",
    "loss_value",
    "loss_owner",
    "loss_suit",
    "loss_trick",
    "entropy",
    "branch_kl",
    "rolled_back",
    "eval_reward_vs_heuristic",
    "eval_bid_hit",
]


def main() -> None:
    args = parse_args()
    device = args.device or str(best_seq_device())
    model_config = SeqModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        d_ff=args.d_ff,
        trick_win_token=not args.no_trick_win_token,
        turn_token=args.turn_token,
    )
    hand_sizes = _csv_ints(args.hand_sizes)
    player_counts = _csv_ints(args.player_counts)
    if args.games_per_cell is not None:
        cells = tuple(
            GameScheduleCell(hand_size=hand_size)
            for hand_size in hand_sizes
            for _ in range(args.games_per_cell)
        )
    else:
        cells = build_position_balanced_schedule(
            hand_sizes=hand_sizes,
            player_counts=player_counts,
            repeats=args.schedule_repeats,
        )
    rate_table = build_branch_rate_table(
        args.reference_rate,
        exhaustive_until=args.exhaustive_until,
        hand_sizes=hand_sizes,
        player_counts=player_counts,
        player_exponent=args.branch_rate_player_exponent,
    )
    train_config = SeqTrainingConfig(
        schedule_cells=cells,
        player_counts=_csv_ints(args.player_counts),
        player_count_weights=_csv_floats(args.player_count_weights),
        branch_rule=BranchRuleConfig(
            bid_top_k=args.bid_top_k,
            play_mode=args.play_mode,
            play_top_k=args.play_top_k,
        ),
        branch_budget=BranchBudgetConfig(branch_rate_by_shape=rate_table),
        rollout=RolloutOptions(
            auto_deals_per_batch=args.auto_deals_per_batch,
            cache_budget_gb=args.cache_budget_gb,
            max_cache_rows=args.max_cache_rows,
            historical_arm="paired",
        ),
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        microbatch_positions=args.microbatch_positions,
        branch_policy_objective=args.branch_objective,
        branch_policy_coef=args.branch_policy_coef,
        spine_policy_coef=args.spine_policy_coef,
        branch_kl_cap=args.branch_kl_cap,
        value_coef=args.value_coef,
        owner_coef=args.owner_coef,
        suit_coef=args.suit_coef,
        trick_coef=args.trick_coef,
        entropy_coef=args.entropy_coef,
        kv_dtype=args.kv_dtype,
        snapshot_every=args.snapshot_every,
        league_max_snapshots=args.league_max_snapshots,
        league_min_iteration=args.league_min_iteration,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
    )

    trainer = SeqTrainer(SeqPlumpModel(model_config), train_config, device=device)
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
        print(f"Resumed from {args.resume_from} at iteration {trainer.iteration}.")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / "metrics.csv"
    if not metrics_path.exists():
        with metrics_path.open("w", newline="") as handle:
            csv.writer(handle).writerow(LOG_COLUMNS)
    (log_dir / "config.json").write_text(
        json.dumps(
            {
                "model_config": asdict(model_config),
                "training_config": asdict(train_config),
                "device": device,
            },
            indent=2,
            default=str,
        )
    )

    eval_bank = DealBank.generate(
        player_counts=train_config.player_counts,
        hand_sizes=tuple(sorted({cell.hand_size for cell in cells})),
        deals_per_configuration=args.eval_deals,
        seed=1234,
    )
    heuristic = HeuristicPolicy()

    print(
        f"Training seq v6 on {device}: dims {args.d_model}x{args.n_layers}, "
        f"{sum(cell.games for cell in cells)} deals/update over {len(cells)} "
        f"shapes, branch rate {args.reference_rate} at 10 cards "
        f"cache cap {args.max_cache_rows} rows | schema: trick_win="
        f"{model_config.trick_win_token} turn={model_config.turn_token} "
        f"(max_seq_len {model_config.max_seq_len})"
    )
    for iteration in range(trainer.iteration + 1, args.iterations + 1):
        trainer.iteration = iteration
        started = time.perf_counter()
        trees, summary = trainer.collect()
        stats = trainer.update(trees)

        eval_reward = ""
        eval_bid_hit = ""
        if args.eval_every > 0 and iteration % args.eval_every == 0:
            policy = SeqModelPolicy(
                trainer.model, device=device, greedy=True, name="candidate"
            )
            report = evaluate_policy(policy, heuristic, eval_bank, batch_size=256)
            eval_reward = f"{report.macro_relative_reward:.4f}"
            eval_bid_hit = f"{report.macro_bid_hit_rate:.4f}"
            trainer.model.eval()

        if iteration % args.checkpoint_every == 0:
            trainer.save_checkpoint(
                Path(args.checkpoint_dir) / "seq_v6_latest.pt"
            )
        trainer.maybe_snapshot(args.checkpoint_dir)

        row = [
            iteration,
            f"{summary.collect_sec:.2f}",
            f"{stats.update_sec:.2f}",
            summary.trees,
            summary.leaves,
            summary.decisions,
            stats.spine_rows,
            stats.branch_rows,
            stats.positions,
            f"{summary.bid_hit_rate:.4f}",
            f"{summary.reward_self:.4f}",
            f"{summary.reward_historical:.4f}",
            f"{summary.spine_entropy:.4f}",
            f"{stats.loss_spine:.5f}",
            f"{stats.loss_branch:.5f}",
            f"{stats.loss_value:.5f}",
            f"{stats.loss_owner:.5f}",
            f"{stats.loss_suit:.5f}",
            f"{stats.loss_trick:.5f}",
            f"{stats.entropy:.5f}",
            f"{stats.branch_kl:.6f}",
            int(stats.rolled_back),
            eval_reward,
            eval_bid_hit,
        ]
        with metrics_path.open("a", newline="") as handle:
            csv.writer(handle).writerow(row)
        total = time.perf_counter() - started
        message = (
            f"iter {iteration:5d} | {total:6.1f}s "
            f"(collect {summary.collect_sec:.1f} update {stats.update_sec:.1f}) | "
            f"leaves {summary.leaves:5d} rows {stats.positions:6d} | "
            f"bid_hit {summary.bid_hit_rate:.3f} ent {summary.spine_entropy:.3f} | "
            f"kl {stats.branch_kl:.5f}{' ROLLBACK' if stats.rolled_back else ''}"
        )
        if eval_reward:
            message += f" | eval vs heuristic {eval_reward} (bid_hit {eval_bid_hit})"
        print(message, flush=True)


if __name__ == "__main__":
    main()
