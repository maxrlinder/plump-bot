#!/usr/bin/env python3
"""Plot all six PC1-PC4 views of learned representations for the 52 cards.

By default the newest durable checkpoint in the active local training run is
used. The input view uses the effective card-dependent part of a PLAY event
token (exact-card + rank + suit embeddings). The action-head view uses the 52
row vectors of the linear card action head. Each view renders every pair among
the first four principal components.

The event-type embedding is constant across all cards, so including it would
only translate every vector and cannot change centered PCA coordinates.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from plump.cards import RANK_LABELS, SUIT_SYMBOLS, Card, Suit
from plump.modeling import card_id
from plump.modeling.encoding import RANKS, SUITS
from plump.policies import ModelPolicy


CHECKPOINT_PATTERN = re.compile(r"plump_v4_iter_(\d+)\.pt$")
DEFAULT_CHECKPOINT_DIR = (
    REPO_ROOT / "checkpoints" / "local" / "v9_8m_laptop_seed1"
)
DEFAULT_OUTPUT = REPO_ROOT / "analysis" / "card_embedding_pca.png"
DEFAULT_ACTION_HEAD_OUTPUT = REPO_ROOT / "analysis" / "card_action_head_pca.png"

SUIT_STYLE = {
    Suit.SPADES: ("#252525", "^"),
    Suit.HEARTS: ("#d62728", "o"),
    Suit.DIAMONDS: ("#1f77b4", "D"),
    Suit.CLUBS: ("#2ca02c", "s"),
}
LABEL_OFFSETS = {
    Suit.SPADES: (-4, 5),
    Suit.HEARTS: (4, 5),
    Suit.DIAMONDS: (4, -10),
    Suit.CLUBS: (-4, -10),
}


@dataclass(frozen=True)
class Projection:
    coordinates: torch.Tensor
    explained_variance: tuple[float, ...]
    nearest_suit_fraction: float
    nearest_rank_gap: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize all 52 learned card embeddings with PCA.",
    )
    parser.add_argument(
        "--source",
        choices=("input", "action-head"),
        default="input",
        help="Card representation to project (default: input).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to inspect. Defaults to the latest file in --checkpoint-dir.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory searched numerically for plump_v4_iter_*.pt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults according to --source.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def latest_checkpoint(directory: Path) -> tuple[int, Path]:
    candidates: list[tuple[int, Path]] = []
    if directory.is_dir():
        for path in directory.iterdir():
            match = CHECKPOINT_PATTERN.fullmatch(path.name)
            if match:
                candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No plump_v4_iter_*.pt checkpoints in {directory}")
    return max(candidates, key=lambda item: item[0])


def checkpoint_iteration(path: Path) -> int | None:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else None


def cards() -> list[Card]:
    return [Card(suit, rank) for suit in SUITS for rank in RANKS]


def input_card_vectors(model) -> torch.Tensor:
    deck = cards()
    card_indices = torch.tensor([card_id(card) for card in deck], dtype=torch.long)
    exact = (
        model.card_emb.weight.detach().float().cpu().index_select(0, card_indices)
    )
    rank_indices = torch.tensor(
        [RANKS.index(card.rank) for card in deck],
        dtype=torch.long,
    )
    suit_indices = torch.tensor(
        [SUITS.index(card.suit) for card in deck],
        dtype=torch.long,
    )
    rank = model.rank_emb.weight.detach().float().cpu().index_select(0, rank_indices)
    suit = model.suit_emb.weight.detach().float().cpu().index_select(0, suit_indices)
    return exact + rank + suit


def action_head_card_vectors(model) -> torch.Tensor:
    """Return one output-side scoring direction per card.

    ``card_head`` maps the shared state vector to 52 logits. Its weight rows are
    therefore the learned output embeddings/directions for the card actions.
    The scalar biases are intentionally omitted from this geometric view.
    """
    weight = model.card_head.weight.detach().float().cpu()
    if weight.ndim != 2 or weight.shape[0] != 52:
        raise ValueError(f"Expected card_head.weight shape (52, d), got {weight.shape}")
    return weight


def pca(vectors: torch.Tensor, dimensions: int = 4) -> Projection:
    centered = vectors - vectors.mean(dim=0, keepdim=True)
    _, singular_values, right = torch.linalg.svd(centered, full_matrices=False)
    components = right[:dimensions].clone()
    # SVD component signs are arbitrary. Fix them for stable rerenders.
    for index in range(dimensions):
        pivot = components[index].abs().argmax()
        if components[index, pivot] < 0:
            components[index].neg_()
    coordinates = centered @ components.T
    variance = singular_values.square()
    ratios = variance / variance.sum().clamp_min(torch.finfo(variance.dtype).eps)

    distances = torch.cdist(vectors, vectors)
    distances.fill_diagonal_(torch.inf)
    nearest = distances.argmin(dim=1).tolist()
    deck = cards()
    suit_fraction = sum(
        deck[index].suit == deck[neighbor].suit
        for index, neighbor in enumerate(nearest)
    ) / len(deck)
    rank_gap = sum(
        abs(int(deck[index].rank) - int(deck[neighbor].rank))
        for index, neighbor in enumerate(nearest)
    ) / len(deck)
    return Projection(
        coordinates=coordinates,
        explained_variance=tuple(float(value) for value in ratios[:dimensions]),
        nearest_suit_fraction=suit_fraction,
        nearest_rank_gap=rank_gap,
    )


def plot_projection(
    ax,
    projection: Projection,
    x_component: int,
    y_component: int,
) -> None:
    deck = cards()
    coordinates = projection.coordinates.numpy()
    for suit in SUITS:
        color, marker = SUIT_STYLE[suit]
        indices = [index for index, card in enumerate(deck) if card.suit == suit]
        ax.scatter(
            coordinates[indices, x_component],
            coordinates[indices, y_component],
            s=27,
            color=color,
            marker=marker,
            alpha=0.9,
            zorder=3,
        )
    for index, card in enumerate(deck):
        offset = LABEL_OFFSETS[card.suit]
        horizontal = "right" if offset[0] < 0 else "left"
        vertical = "top" if offset[1] < 0 else "bottom"
        ax.annotate(
            f"{RANK_LABELS[card.rank]}{SUIT_SYMBOLS[card.suit]}",
            (coordinates[index, x_component], coordinates[index, y_component]),
            xytext=offset,
            textcoords="offset points",
            ha=horizontal,
            va=vertical,
            fontsize=6.5,
            color=SUIT_STYLE[card.suit][0],
            zorder=4,
        )
    x_variance = projection.explained_variance[x_component]
    y_variance = projection.explained_variance[y_component]
    ax.set_title(f"PC{x_component + 1} × PC{y_component + 1}", fontsize=11)
    ax.set_xlabel(f"PC{x_component + 1} ({100 * x_variance:.1f}%)")
    ax.set_ylabel(f"PC{y_component + 1} ({100 * y_variance:.1f}%)")
    ax.axhline(0.0, color="#888888", linewidth=0.7, alpha=0.35)
    ax.axvline(0.0, color="#888888", linewidth=0.7, alpha=0.35)
    ax.grid(alpha=0.16)
    ax.margins(0.13)


def main() -> None:
    args = parse_args()
    if args.checkpoint is None:
        iteration, checkpoint = latest_checkpoint(args.checkpoint_dir)
    else:
        checkpoint = args.checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        iteration = checkpoint_iteration(checkpoint)

    policy = ModelPolicy.from_checkpoint(checkpoint, device="cpu", greedy=True)
    model = policy.model
    if args.source == "input":
        if not all(
            hasattr(model, name) for name in ("card_emb", "rank_emb", "suit_emb")
        ):
            raise TypeError(
                "Checkpoint model does not expose schema-v4 card input embeddings."
            )
        vectors = input_card_vectors(model)
        figure_title = "Effective PLAY-event card input PCA"
        output = args.output or DEFAULT_OUTPUT
    else:
        if not hasattr(model, "card_head"):
            raise TypeError("Checkpoint model does not expose a card action head.")
        vectors = action_head_card_vectors(model)
        figure_title = "Card action-head output embedding PCA"
        output = args.output or DEFAULT_ACTION_HEAD_OUTPUT

    projection = pca(vectors, dimensions=4)

    component_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
    for ax, (x_component, y_component) in zip(axes.flat, component_pairs):
        plot_projection(ax, projection, x_component, y_component)

    legend_handles = [
        plt.Line2D(
            [],
            [],
            color=color,
            marker=marker,
            linestyle="None",
            markersize=6,
            label=suit.value.title(),
        )
        for suit, (color, marker) in SUIT_STYLE.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    checkpoint_label = f"iteration {iteration}" if iteration is not None else checkpoint.name
    fig.suptitle(
        f"{figure_title} — {checkpoint_label}\n"
        f"1-NN same suit {100 * projection.nearest_suit_fraction:.0f}% · "
        f"mean nearest rank gap {projection.nearest_rank_gap:.1f}",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    fig.savefig(temporary, dpi=args.dpi)
    plt.close(fig)
    temporary.replace(output)

    print(f"checkpoint={checkpoint}")
    print(f"iteration={iteration if iteration is not None else 'unknown'}")
    print(f"source={args.source}")
    print(f"output={output}")
    explained = " ".join(
        f"pc{index + 1}={100 * value:.2f}%"
        for index, value in enumerate(projection.explained_variance)
    )
    print(
        f"{args.source}: {explained} "
        f"nearest_same_suit={100 * projection.nearest_suit_fraction:.1f}% "
        f"mean_nearest_rank_gap={projection.nearest_rank_gap:.2f}"
    )


if __name__ == "__main__":
    main()
