"""Reward transforms shared by training and evaluation."""

from __future__ import annotations


def compute_relative_rewards(scores: dict[int, int]) -> dict[int, float]:
    """Return each player's score relative to the mean opponent score."""

    if len(scores) < 2:
        raise ValueError("Relative rewards require at least two players.")
    total = sum(scores.values())
    opponents = len(scores) - 1
    return {
        player: float(score - ((total - score) / opponents))
        for player, score in scores.items()
    }
