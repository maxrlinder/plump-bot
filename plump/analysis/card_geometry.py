#!/usr/bin/env python3
"""Schema-v6 card-representation geometry and probe report.

The input representation is the card-dependent part of a PLAY event token:
exact-card + rank + suit slot embeddings. The action representation is its
effective exact-card + rank + suit output row. Scalar biases are excluded.
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import MDS
from umap import UMAP

from plump.cards import RANK_LABELS, SUIT_SYMBOLS, Suit
from plump.runs import atomic_write_json, atomic_write_text
from plump.seq.policy import SeqModelPolicy

from .card_pca import (
    LABEL_OFFSETS,
    SUIT_STYLE,
    action_head_card_vectors,
    cards,
    input_card_vectors,
    pca,
    save_pca,
)


def card_labels() -> list[str]:
    return [f"{RANK_LABELS[card.rank]}{SUIT_SYMBOLS[card.suit]}" for card in cards()]


def normalized_card_vectors(model, source: str) -> np.ndarray:
    if source == "input":
        vectors = input_card_vectors(model)
    elif source == "action-head":
        vectors = action_head_card_vectors(model)
    else:
        raise ValueError(f"Unknown card representation: {source}")
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


def fit_mds(distances: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    """Fit the deterministic report MDS and return coordinates plus stress."""

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
    return coordinates, float(estimator.stress_)


def plot_mds(
    distances: np.ndarray,
    representation_title: str,
    iteration_label: str,
    output: Path,
    seed: int,
    dpi: int,
) -> float:
    coordinates, stress = fit_mds(distances, seed)
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    scatter_cards(ax, coordinates)
    ax.set_title(
        f"{representation_title} cosine MDS — {iteration_label}\n"
        f"normalized stress {stress:.3f}",
        fontsize=14,
    )
    ax.set_xlabel("MDS dimension 1")
    ax.set_ylabel("MDS dimension 2")
    save_figure(fig, output, dpi)
    return stress


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


def probe_p_values(
    suit_accuracy: float,
    suit_baseline: np.ndarray,
    rank_mae: float,
    rank_baseline: np.ndarray,
) -> tuple[float, float]:
    """Plus-one permutation p-values for the two directional probes."""

    permutations = len(suit_baseline)
    if len(rank_baseline) != permutations or permutations < 1:
        raise ValueError("Probe baselines must have equal positive length.")
    suit_p = (1 + np.count_nonzero(suit_baseline >= suit_accuracy)) / (
        permutations + 1
    )
    rank_p = (1 + np.count_nonzero(rank_baseline <= rank_mae)) / (
        permutations + 1
    )
    return float(suit_p), float(rank_p)


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
    suit_p, rank_p = probe_p_values(
        suit_accuracy,
        suit_baseline,
        rank_mae,
        rank_baseline,
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


def analyze_checkpoint(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    permutations: int = 1000,
    dpi: int = 180,
) -> dict:
    """Write the complete card-geometry suite and a quantitative manifest."""

    if permutations < 1:
        raise ValueError("permutations must be at least 1")
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    iteration = int(payload.get("iteration", 0))
    label = f"iteration {iteration}" if iteration else checkpoint.name
    policy = SeqModelPolicy.from_checkpoint(
        checkpoint,
        device="cpu",
        greedy=True,
    )
    report: dict = {
        "checkpoint": str(checkpoint),
        "iteration": iteration,
        "outputs": [],
        "representations": {},
    }

    for source, title, prefix in (
        ("input", "Effective card input embedding", "card_input_embedding"),
        (
            "action-head",
            "Effective card action output embedding",
            "card_action_head",
        ),
    ):
        pca_output = output_dir / f"{prefix}_pca.png"
        projection = save_pca(
            policy.model,
            source,
            pca_output,
            checkpoint_label=label,
            dpi=dpi,
        )
        vectors = normalized_card_vectors(policy.model, source)
        distances = cosine_distances(vectors)
        outputs = {
            "pca": pca_output,
            "mds": output_dir / f"{prefix}_mds.png",
            "umap": output_dir / f"{prefix}_umap.png",
            "heatmap": output_dir / f"{prefix}_cosine_heatmap.png",
            "probes": output_dir / f"{prefix}_probes.png",
        }
        stress = plot_mds(
            distances,
            title,
            label,
            outputs["mds"],
            seed,
            dpi,
        )
        plot_umap(vectors, title, label, outputs["umap"], seed, dpi)
        plot_similarity_heatmap(
            vectors,
            title,
            label,
            outputs["heatmap"],
            dpi,
        )
        suit_accuracy, suit_p, rank_mae, rank_p = plot_probe_baselines(
            vectors,
            title,
            label,
            outputs["probes"],
            permutations,
            seed,
            dpi,
        )
        report["outputs"].extend(str(path) for path in outputs.values())
        report["representations"][source] = {
            "pca_explained_variance": projection.explained_variance,
            "nearest_same_suit": projection.nearest_suit_fraction,
            "mean_nearest_rank_gap": projection.nearest_rank_gap,
            "mds_normalized_stress": stress,
            "suit_probe_accuracy": suit_accuracy,
            "suit_probe_p": suit_p,
            "rank_probe_mae": rank_mae,
            "rank_probe_p": rank_p,
            "cosine_heatmap_representation": "exact-card+rank+suit",
        }

    atomic_write_json(output_dir / "report.json", report)
    return report


HISTORY_METRIC_FIELDS = (
    "mds_normalized_stress",
    "pca_first_component_explained",
    "pca_first_four_explained",
    "suit_probe_accuracy",
    "suit_probe_p",
    "rank_probe_mae",
    "rank_probe_p",
    "nearest_same_suit",
    "mean_nearest_rank_gap",
    "mean_cosine_distance",
    "mean_same_suit_cosine_distance",
    "mean_same_rank_cosine_distance",
    "mean_nearest_cosine_distance",
)


def _history_checkpoint_signature(checkpoint: Path) -> dict[str, int | str]:
    stat = checkpoint.stat()
    return {
        "name": checkpoint.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _card_distance_metrics(distances: np.ndarray) -> dict[str, float]:
    deck = cards()
    count = len(deck)
    upper = np.triu(np.ones((count, count), dtype=bool), k=1)
    suits = np.asarray([card.suit for card in deck], dtype=object)
    ranks = np.asarray([int(card.rank) for card in deck], dtype=np.int64)
    same_suit = upper & (suits[:, None] == suits[None, :])
    same_rank = upper & (ranks[:, None] == ranks[None, :])
    nearest = distances.copy()
    np.fill_diagonal(nearest, np.inf)
    return {
        "mean_cosine_distance": float(distances[upper].mean()),
        "mean_same_suit_cosine_distance": float(distances[same_suit].mean()),
        "mean_same_rank_cosine_distance": float(distances[same_rank].mean()),
        "mean_nearest_cosine_distance": float(nearest.min(axis=1).mean()),
    }


def _representation_history_metrics(
    model,
    source: str,
    *,
    permutations: int,
    seed: int,
) -> dict[str, float]:
    if source == "input":
        raw_vectors = input_card_vectors(model)
    elif source == "action-head":
        raw_vectors = action_head_card_vectors(model)
    else:
        raise ValueError(f"Unknown card representation: {source}")
    projection = pca(raw_vectors)
    vectors = normalized_card_vectors(model, source)
    distances = cosine_distances(vectors)
    _, stress = fit_mds(distances, seed)
    suit_accuracy, suit_baseline, rank_mae, rank_baseline = probe_metrics(
        vectors,
        permutations,
        seed,
    )
    suit_p, rank_p = probe_p_values(
        suit_accuracy,
        suit_baseline,
        rank_mae,
        rank_baseline,
    )
    explained = tuple(float(value) for value in projection.explained_variance)
    metrics = {
        "mds_normalized_stress": stress,
        "pca_first_component_explained": explained[0],
        "pca_first_four_explained": float(sum(explained[:4])),
        "suit_probe_accuracy": suit_accuracy,
        "suit_probe_p": suit_p,
        "rank_probe_mae": rank_mae,
        "rank_probe_p": rank_p,
        "nearest_same_suit": projection.nearest_suit_fraction,
        "mean_nearest_rank_gap": projection.nearest_rank_gap,
    }
    metrics.update(_card_distance_metrics(distances))
    return metrics


def _checkpoint_history_metrics(
    checkpoint: Path,
    *,
    permutations: int,
    seed: int,
) -> dict:
    policy = SeqModelPolicy.from_checkpoint(
        checkpoint,
        device="cpu",
        greedy=True,
    )
    match = re.fullmatch(r"iter_(\d+)\.pt", checkpoint.name)
    if match is None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        iteration = int(payload.get("iteration", 0))
    else:
        iteration = int(match.group(1))
    representations = {
        source: _representation_history_metrics(
            policy.model,
            source,
            permutations=permutations,
            seed=seed,
        )
        for source in ("input", "action-head")
    }
    return {
        "iteration": iteration,
        "checkpoint": str(checkpoint),
        "signature": _history_checkpoint_signature(checkpoint),
        "representations": representations,
    }


def _plot_history(
    checkpoints: list[dict],
    output: Path,
    *,
    permutations: int,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(
        5,
        2,
        figsize=(16, 19),
        sharex="col",
        constrained_layout=True,
    )
    sources = (
        ("input", "Effective card input embedding"),
        ("action-head", "Effective card action output embedding"),
    )
    for column, (source, title) in enumerate(sources):
        iterations = np.asarray(
            [row["iteration"] for row in checkpoints], dtype=np.float64
        )

        def values(field: str) -> np.ndarray:
            return np.asarray(
                [row["representations"][source][field] for row in checkpoints],
                dtype=np.float64,
            )

        ax = axes[0, column]
        ax.plot(
            iterations,
            values("mds_normalized_stress"),
            marker="o",
            markersize=3,
            label="MDS normalized stress ↓",
        )
        ax.plot(
            iterations,
            values("pca_first_four_explained"),
            marker="o",
            markersize=3,
            label="PCA variance, first 4 ↑",
        )
        ax.set_title(title, fontsize=13)
        ax.set_ylabel("Geometry summary")
        ax.set_ylim(bottom=0)

        ax = axes[1, column]
        ax.plot(
            iterations,
            values("suit_probe_accuracy"),
            marker="o",
            markersize=3,
            label="Suit probe accuracy ↑",
        )
        ax.plot(
            iterations,
            values("nearest_same_suit"),
            marker="o",
            markersize=3,
            label="Nearest card has same suit ↑",
        )
        ax.axhline(
            0.25,
            color="#777777",
            linestyle=":",
            linewidth=1,
            label="Suit chance (25%)",
        )
        ax.set_ylabel("Suit structure")
        ax.set_ylim(0, 1.02)

        ax = axes[2, column]
        ax.plot(
            iterations,
            values("rank_probe_mae"),
            marker="o",
            markersize=3,
            label="Rank probe MAE ↓",
        )
        ax.plot(
            iterations,
            values("mean_nearest_rank_gap"),
            marker="o",
            markersize=3,
            label="Nearest-card rank gap ↓",
        )
        ax.set_ylabel("Rank difference")
        ax.set_ylim(bottom=0)

        ax = axes[3, column]
        ax.plot(
            iterations,
            values("suit_probe_p"),
            marker="o",
            markersize=3,
            label="Suit permutation p ↓",
        )
        ax.plot(
            iterations,
            values("rank_probe_p"),
            marker="o",
            markersize=3,
            label="Rank permutation p ↓",
        )
        ax.axhline(
            0.05,
            color="#777777",
            linestyle=":",
            linewidth=1,
            label="p = 0.05",
        )
        ax.set_yscale("log")
        ax.set_ylabel("Permutation p-value")

        ax = axes[4, column]
        distance_fields = (
            ("mean_cosine_distance", "All card pairs"),
            ("mean_same_suit_cosine_distance", "Same-suit pairs"),
            ("mean_same_rank_cosine_distance", "Same-rank pairs"),
            ("mean_nearest_cosine_distance", "Nearest card"),
        )
        for field, label in distance_fields:
            ax.plot(
                iterations,
                values(field),
                marker="o",
                markersize=3,
                label=label,
            )
        ax.set_ylabel("Cosine distance ↓")
        ax.set_xlabel("Training iteration")
        ax.set_ylim(bottom=0)

        for row in range(5):
            axes[row, column].grid(alpha=0.2)
            axes[row, column].legend(fontsize=8, loc="best")

    figure.suptitle(
        "Card-representation geometry across checkpoints\n"
        f"{len(checkpoints)} checkpoints · {permutations} probe permutations",
        fontsize=15,
    )
    save_figure(figure, output, dpi)


def _history_csv(checkpoints: list[dict]) -> str:
    buffer = io.StringIO()
    fieldnames = ("iteration", "representation", *HISTORY_METRIC_FIELDS)
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for checkpoint in checkpoints:
        for source in ("input", "action-head"):
            writer.writerow(
                {
                    "iteration": checkpoint["iteration"],
                    "representation": source,
                    **checkpoint["representations"][source],
                }
            )
    return buffer.getvalue()


def analyze_checkpoint_history(
    checkpoints: list[str | Path],
    output_dir: str | Path,
    *,
    seed: int = 42,
    permutations: int = 1000,
    dpi: int = 180,
    force: bool = False,
) -> dict:
    """Compute and plot card-geometry scalars across interval checkpoints.

    The JSON report doubles as an incremental cache. A later invocation only
    evaluates new or replaced checkpoints when seed and permutation settings
    match, which keeps this useful while a run is still producing snapshots.
    """

    if permutations < 1:
        raise ValueError("permutations must be at least 1")
    paths = sorted(
        (Path(path).expanduser().resolve() for path in checkpoints),
        key=lambda path: path.name,
    )
    if not paths:
        raise ValueError("History analysis requires at least one checkpoint.")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_output = output_dir / "card_geometry_history.json"
    csv_output = output_dir / "card_geometry_history.csv"
    plot_output = output_dir / "card_geometry_history.png"

    cached: dict[tuple[str, int, int], dict] = {}
    if not force and json_output.is_file():
        try:
            previous = json.loads(json_output.read_text())
            if (
                int(previous.get("seed", -1)) == seed
                and int(previous.get("permutations", -1)) == permutations
            ):
                for row in previous.get("checkpoints", []):
                    signature = row["signature"]
                    key = (
                        str(signature["name"]),
                        int(signature["size"]),
                        int(signature["mtime_ns"]),
                    )
                    cached[key] = row
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            cached = {}

    rows: list[dict] = []
    for checkpoint in paths:
        signature = _history_checkpoint_signature(checkpoint)
        key = (
            str(signature["name"]),
            int(signature["size"]),
            int(signature["mtime_ns"]),
        )
        row = cached.get(key)
        if row is None:
            row = _checkpoint_history_metrics(
                checkpoint,
                permutations=permutations,
                seed=seed,
            )
        rows.append(row)
    rows.sort(key=lambda row: int(row["iteration"]))

    _plot_history(rows, plot_output, permutations=permutations, dpi=dpi)
    atomic_write_text(csv_output, _history_csv(rows))
    report = {
        "format_version": 1,
        "seed": seed,
        "permutations": permutations,
        "checkpoints": rows,
        "outputs": [str(plot_output), str(csv_output), str(json_output)],
    }
    atomic_write_json(json_output, report)
    return report
