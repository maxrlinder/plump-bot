#!/usr/bin/env python3
"""PCA views of schema-v6 card input and action representations.

The event-type embedding is constant across all cards, so including it would
only translate every vector and cannot change centered PCA coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from plump.cards import RANK_LABELS, SUIT_SYMBOLS, Card, Suit
from plump.seq.config import NUM_CARDS, SLOT_CARD, SLOT_RANK, SLOT_SUIT
from plump.seq.tokens import RANKS, SUITS, card_id


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


def cards() -> list[Card]:
    return [Card(suit, rank) for suit in SUITS for rank in RANKS]


def input_card_vectors(model) -> torch.Tensor:
    """Return the card-dependent part of a schema-v6 token embedding."""

    deck = cards()
    offsets = model.slot_offsets.detach().cpu()
    card_indices = torch.tensor(
        [int(offsets[SLOT_CARD]) + card_id(card) for card in deck],
        dtype=torch.long,
    )
    rank_indices = torch.tensor(
        [
            int(offsets[SLOT_RANK]) + RANKS.index(card.rank)
            for card in deck
        ],
        dtype=torch.long,
    )
    suit_indices = torch.tensor(
        [
            int(offsets[SLOT_SUIT]) + SUITS.index(card.suit)
            for card in deck
        ],
        dtype=torch.long,
    )
    embedding = model.slot_embedding.weight.detach().float().cpu()
    return (
        embedding.index_select(0, card_indices)
        + embedding.index_select(0, rank_indices)
        + embedding.index_select(0, suit_indices)
    )


def exact_card_input_vectors(model) -> torch.Tensor:
    """Return only the 52 exact-card slot embeddings.

    Unlike ``input_card_vectors``, this excludes the explicitly shared rank and
    suit slot embeddings, so any geometry here was learned by the individual
    card identities rather than supplied by the token schema.
    """

    offsets = model.slot_offsets.detach().cpu()
    indices = torch.arange(NUM_CARDS, dtype=torch.long) + int(offsets[SLOT_CARD])
    return model.slot_embedding.weight.detach().float().cpu().index_select(
        0,
        indices,
    )


def action_head_card_vectors(model) -> torch.Tensor:
    """Return one output-side scoring direction per card.

    Each direction is the effective exact-card + rank + suit row used for the
    card logits. The scalar exact-card biases are intentionally omitted from
    this geometric view.
    """
    weight = model.effective_card_output_weight().detach().float().cpu()
    if weight.ndim != 2 or weight.shape[0] != 52:
        raise ValueError(
            f"Expected effective card output shape (52, d), got {weight.shape}"
        )
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


def save_pca(
    model,
    source: str,
    output: Path,
    *,
    checkpoint_label: str,
    dpi: int = 180,
) -> Projection:
    if source == "input":
        vectors = input_card_vectors(model)
        figure_title = "Effective PLAY-event card input PCA"
    elif source == "action-head":
        vectors = action_head_card_vectors(model)
        figure_title = "Effective card action output embedding PCA"
    else:
        raise ValueError(f"Unknown card representation: {source}")

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
    fig.suptitle(
        f"{figure_title} — {checkpoint_label}\n"
        f"1-NN same suit {100 * projection.nearest_suit_fraction:.0f}% · "
        f"mean nearest rank gap {projection.nearest_rank_gap:.1f}",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    fig.savefig(temporary, dpi=dpi)
    plt.close(fig)
    temporary.replace(output)
    return projection
