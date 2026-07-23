#!/usr/bin/env python3
"""Render complementary geometry analyses for 52 learned card representations.

The input representation is the card-dependent part of a PLAY event token:
``card_emb + rank_emb + suit_emb``. The action representation is a row in
``model.card_head.weight``: the learned direction in shared state space used to
score that card. All geometric analyses use L2-normalized vectors and cosine
distance. Scalar output biases are deliberately excluded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import MDS
from umap import UMAP

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plump.cards import RANK_LABELS, SUIT_SYMBOLS, Suit
from plump.policies import ModelPolicy
from plot_card_embedding_pca import (
    DEFAULT_CHECKPOINT_DIR,
    LABEL_OFFSETS,
    SUIT_STYLE,
    action_head_card_vectors,
    cards,
    checkpoint_iteration,
    input_card_vectors,
    latest_checkpoint,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze learned card input or action-head embeddings.",
    )
    parser.add_argument(
        "--source",
        choices=("action-head", "input"),
        default="action-head",
        help="Representation to analyze (default: action-head).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to inspect. Defaults to the latest in --checkpoint-dir.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory searched numerically for plump_v4_iter_*.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the four PNG outputs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--permutations",
        type=int,
        default=1000,
        help="Shuffled-label baselines for the probe plot.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def card_labels() -> list[str]:
    return [f"{RANK_LABELS[card.rank]}{SUIT_SYMBOLS[card.suit]}" for card in cards()]


def normalized_card_vectors(model, source: str) -> np.ndarray:
    if source == "input":
        vectors = input_card_vectors(model)
    else:
        vectors = action_head_card_vectors(model)
    vectors = vectors.numpy().astype(np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, np.finfo(vectors.dtype).eps)


def cosine_distances(vectors: np.ndarray) -> np.ndarray:
    distances = 1.0 - vectors @ vectors.T
    distances = np.clip(distances, 0.0, 2.0)
    np.fill_diagonal(distances, 0.0)
    return distances


def save_figure(fig, output: Path, dpi: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    fig.savefig(temporary, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    temporary.replace(output)


def scatter_cards(ax, coordinates: np.ndarray) -> None:
    deck = cards()
    for suit in SUIT_STYLE:
        color, marker = SUIT_STYLE[suit]
        indices = [index for index, card in enumerate(deck) if card.suit == suit]
        ax.scatter(
            coordinates[indices, 0],
            coordinates[indices, 1],
            s=42,
            color=color,
            marker=marker,
            alpha=0.9,
            label=suit.value.title(),
            zorder=3,
        )
    for index, card in enumerate(deck):
        offset = LABEL_OFFSETS[card.suit]
        ax.annotate(
            card_labels()[index],
            coordinates[index],
            xytext=offset,
            textcoords="offset points",
            ha="right" if offset[0] < 0 else "left",
            va="top" if offset[1] < 0 else "bottom",
            fontsize=7.5,
            color=SUIT_STYLE[card.suit][0],
            zorder=4,
        )
    ax.axhline(0.0, color="#888888", linewidth=0.7, alpha=0.35)
    ax.axvline(0.0, color="#888888", linewidth=0.7, alpha=0.35)
    ax.grid(alpha=0.16)
    ax.margins(0.14)
    ax.legend(fontsize=8, loc="best")


def plot_mds(
    distances: np.ndarray,
    representation_title: str,
    iteration_label: str,
    output: Path,
    seed: int,
    dpi: int,
) -> float:
    estimator = MDS(
        n_components=2,
        metric_mds=True,
        metric="precomputed",
        normalized_stress=True,
        n_init=8,
        init="random",
        max_iter=1000,
        eps=1e-9,
        random_state=seed,
        n_jobs=1,
    )
    coordinates = estimator.fit_transform(distances)
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    scatter_cards(ax, coordinates)
    ax.set_title(
        f"{representation_title} cosine MDS — {iteration_label}\n"
        f"normalized stress {estimator.stress_:.3f}",
        fontsize=14,
    )
    ax.set_xlabel("MDS dimension 1")
    ax.set_ylabel("MDS dimension 2")
    save_figure(fig, output, dpi)
    return float(estimator.stress_)


def plot_umap(
    vectors: np.ndarray,
    representation_title: str,
    iteration_label: str,
    output: Path,
    seed: int,
    dpi: int,
) -> None:
    estimator = UMAP(
        n_components=2,
        n_neighbors=8,
        min_dist=0.15,
        metric="cosine",
        init="spectral",
        random_state=seed,
        transform_seed=seed,
        n_jobs=1,
    )
    coordinates = estimator.fit_transform(vectors)
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    scatter_cards(ax, coordinates)
    ax.set_title(
        f"{representation_title} cosine UMAP — {iteration_label}\n"
        f"8 neighbors · min_dist 0.15 · seed {seed}",
        fontsize=14,
    )
    ax.set_xlabel("UMAP dimension 1")
    ax.set_ylabel("UMAP dimension 2")
    save_figure(fig, output, dpi)


def plot_similarity_heatmap(
    vectors: np.ndarray,
    representation_title: str,
    iteration_label: str,
    output: Path,
    dpi: int,
) -> None:
    deck = cards()
    suit_order = (Suit.HEARTS, Suit.SPADES, Suit.DIAMONDS, Suit.CLUBS)
    suit_position = {suit: index for index, suit in enumerate(suit_order)}
    suit_major = sorted(
        range(len(deck)),
        key=lambda index: (
            suit_position[deck[index].suit],
            int(deck[index].rank),
        ),
    )
    rank_major = sorted(
        range(len(deck)),
        key=lambda index: (
            int(deck[index].rank),
            suit_position[deck[index].suit],
        ),
    )
    similarities = vectors @ vectors.T
    labels = card_labels()

    fig, axes = plt.subplots(1, 2, figsize=(24, 11), constrained_layout=True)
    image = None
    panel_specs = (
        (axes[0], suit_major, "Suit-major: ♥, ♠, ♦, ♣ · ranks 2→A", (13, 26, 39)),
        (
            axes[1],
            rank_major,
            "Rank-major: 2→A · suits ♥, ♠, ♦, ♣",
            tuple(range(4, 52, 4)),
        ),
    )
    for ax, order, title, boundaries in panel_specs:
        ordered = similarities[np.ix_(order, order)]
        image = ax.imshow(
            ordered,
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
            aspect="equal",
        )
        tick_labels = [labels[index] for index in order]
        positions = np.arange(len(order))
        ax.set_xticks(positions, tick_labels, rotation=90, fontsize=6.5)
        ax.set_yticks(positions, tick_labels, fontsize=6.5)
        for tick, index in zip(ax.get_xticklabels(), order):
            tick.set_color(SUIT_STYLE[deck[index].suit][0])
        for tick, index in zip(ax.get_yticklabels(), order):
            tick.set_color(SUIT_STYLE[deck[index].suit][0])
        for boundary in boundaries:
            position = boundary - 0.5
            ax.axhline(position, color="#555555", linewidth=0.7, alpha=0.65)
            ax.axvline(position, color="#555555", linewidth=0.7, alpha=0.65)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Card")
        ax.set_ylabel("Card")
    fig.suptitle(
        f"{representation_title} cosine similarity — {iteration_label}",
        fontsize=14,
    )
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.015)
    colorbar.set_label("Cosine similarity")
    save_figure(fig, output, dpi)


def ridge_fold_operator(
    x_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    mean = x_train.mean(axis=0, keepdims=True)
    train = x_train - mean
    test = x_test - mean
    gram = train @ train.T
    gram.flat[:: gram.shape[0] + 1] += alpha
    operator = test @ train.T @ np.linalg.inv(gram)
    return operator


def ridge_cv_operators(
    vectors: np.ndarray,
    groups: np.ndarray,
    alpha: float = 0.1,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    operators: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for group in np.unique(groups):
        test_mask = groups == group
        train_mask = ~test_mask
        operator = ridge_fold_operator(
            vectors[train_mask],
            vectors[test_mask],
            alpha,
        )
        operators.append((test_mask, train_mask, operator))
    return operators


def cross_validated_ridge_predictions(
    targets: np.ndarray,
    operators: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    target_matrix = targets if targets.ndim == 2 else targets[:, None]
    predictions = np.empty((len(targets), target_matrix.shape[1]), dtype=np.float64)
    for test_mask, train_mask, operator in operators:
        train_targets = target_matrix[train_mask]
        target_mean = train_targets.mean(axis=0, keepdims=True)
        predictions[test_mask] = (
            operator @ (train_targets - target_mean) + target_mean
        )
    return predictions if targets.ndim == 2 else predictions[:, 0]


def probe_metrics(
    vectors: np.ndarray,
    permutations: int,
    seed: int,
) -> tuple[float, np.ndarray, float, np.ndarray]:
    deck = cards()
    suit_names = list(SUIT_STYLE)
    suit_targets = np.asarray([suit_names.index(card.suit) for card in deck])
    one_hot_suits = np.eye(len(suit_names), dtype=np.float64)[suit_targets]
    ranks = np.asarray([int(card.rank) for card in deck], dtype=np.float64)

    # Holding out a complete rank tests whether suit generalizes to an unseen
    # rank. Holding out a complete suit tests whether rank generalizes to an
    # unseen suit instead of memorizing individual card identities.
    suit_groups = ranks.astype(int)
    rank_groups = suit_targets
    suit_operators = ridge_cv_operators(vectors, suit_groups)
    rank_operators = ridge_cv_operators(vectors, rank_groups)
    suit_scores = cross_validated_ridge_predictions(one_hot_suits, suit_operators)
    observed_suit_accuracy = float(
        np.mean(suit_scores.argmax(axis=1) == suit_targets)
    )
    rank_predictions = cross_validated_ridge_predictions(ranks, rank_operators)
    observed_rank_mae = float(np.mean(np.abs(rank_predictions - ranks)))

    rng = np.random.default_rng(seed)
    baseline_suit = np.empty(permutations, dtype=np.float64)
    baseline_rank = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        shuffled_suits = one_hot_suits[rng.permutation(len(deck))]
        shuffled_suit_labels = shuffled_suits.argmax(axis=1)
        scores = cross_validated_ridge_predictions(shuffled_suits, suit_operators)
        baseline_suit[index] = np.mean(
            scores.argmax(axis=1) == shuffled_suit_labels
        )

        shuffled_ranks = ranks[rng.permutation(len(deck))]
        predictions = cross_validated_ridge_predictions(shuffled_ranks, rank_operators)
        baseline_rank[index] = np.mean(np.abs(predictions - shuffled_ranks))

    return observed_suit_accuracy, baseline_suit, observed_rank_mae, baseline_rank


def plot_probe_baselines(
    vectors: np.ndarray,
    representation_title: str,
    iteration_label: str,
    output: Path,
    permutations: int,
    seed: int,
    dpi: int,
) -> tuple[float, float, float, float]:
    suit_accuracy, suit_baseline, rank_mae, rank_baseline = probe_metrics(
        vectors, permutations, seed
    )
    suit_p = (1 + np.count_nonzero(suit_baseline >= suit_accuracy)) / (
        permutations + 1
    )
    rank_p = (1 + np.count_nonzero(rank_baseline <= rank_mae)) / (
        permutations + 1
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    axes[0].hist(suit_baseline, bins=20, color="#9e9e9e", alpha=0.8)
    axes[0].axvline(suit_accuracy, color="#d62728", linewidth=2.2)
    axes[0].set_title(
        f"Suit linear probe: {100 * suit_accuracy:.1f}% accuracy\n"
        f"leave-one-rank-out · permutation p={suit_p:.4f}"
    )
    axes[0].set_xlabel("Accuracy (higher is better)")
    axes[0].set_ylabel("Shuffled-label permutations")

    axes[1].hist(rank_baseline, bins=20, color="#9e9e9e", alpha=0.8)
    axes[1].axvline(rank_mae, color="#1f77b4", linewidth=2.2)
    axes[1].set_title(
        f"Rank linear probe: {rank_mae:.2f} MAE\n"
        f"leave-one-suit-out · permutation p={rank_p:.4f}"
    )
    axes[1].set_xlabel("Rank MAE (lower is better)")
    axes[1].set_ylabel("Shuffled-label permutations")
    fig.suptitle(
        f"{representation_title} linear probes — {iteration_label}",
        fontsize=14,
    )
    save_figure(fig, output, dpi)
    return suit_accuracy, suit_p, rank_mae, rank_p


def main() -> None:
    args = parse_args()
    if args.permutations < 1:
        raise ValueError("--permutations must be at least 1")
    if args.checkpoint is None:
        iteration, checkpoint = latest_checkpoint(args.checkpoint_dir)
    else:
        checkpoint = args.checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        iteration = checkpoint_iteration(checkpoint)

    policy = ModelPolicy.from_checkpoint(checkpoint, device="cpu", greedy=True)
    if args.source == "input":
        required = ("card_emb", "rank_emb", "suit_emb")
        if not all(hasattr(policy.model, name) for name in required):
            raise TypeError("Checkpoint model does not expose card input embeddings.")
        representation_title = "Effective card input embedding"
        output_prefix = "card_input_embedding"
    else:
        if not hasattr(policy.model, "card_head"):
            raise TypeError("Checkpoint model does not expose a card action head.")
        representation_title = "Card action-head"
        output_prefix = "card_action_head"
    vectors = normalized_card_vectors(policy.model, args.source)
    distances = cosine_distances(vectors)
    label = f"iteration {iteration}" if iteration is not None else checkpoint.name
    output_dir = args.output_dir.expanduser().resolve()

    outputs = {
        "mds": output_dir / f"{output_prefix}_mds.png",
        "umap": output_dir / f"{output_prefix}_umap.png",
        "heatmap": output_dir / f"{output_prefix}_cosine_heatmap.png",
        "probes": output_dir / f"{output_prefix}_probes.png",
    }
    stress = plot_mds(
        distances,
        representation_title,
        label,
        outputs["mds"],
        args.seed,
        args.dpi,
    )
    plot_umap(
        vectors,
        representation_title,
        label,
        outputs["umap"],
        args.seed,
        args.dpi,
    )
    plot_similarity_heatmap(
        vectors,
        representation_title,
        label,
        outputs["heatmap"],
        args.dpi,
    )
    suit_accuracy, suit_p, rank_mae, rank_p = plot_probe_baselines(
        vectors,
        representation_title,
        label,
        outputs["probes"],
        args.permutations,
        args.seed,
        args.dpi,
    )

    print(f"checkpoint={checkpoint}")
    print(f"iteration={iteration if iteration is not None else 'unknown'}")
    print(f"source={args.source}")
    for name, output in outputs.items():
        print(f"{name}_output={output}")
    print(f"mds_normalized_stress={stress:.6f}")
    print(f"suit_probe_accuracy={suit_accuracy:.6f} p={suit_p:.6f}")
    print(f"rank_probe_mae={rank_mae:.6f} p={rank_p:.6f}")


if __name__ == "__main__":
    main()
