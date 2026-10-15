"""The world: the true outcome, the clues, the news schedule, and the benchmark P*.

Clue ids run from 0 to M - 1. The first K are the private pool handed out at setup; the
rest are released as scheduled news, in id order, at evenly spaced steps (PLAN.md
Section 2.4).
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from kalshi_model.config import Config


@dataclass(frozen=True, eq=False)
class World:
    """One draw of the world. Arrays are read-only."""

    y: bool
    """True outcome: True means the contract resolves YES."""
    points_yes: np.ndarray
    """`points_yes[i]` is True if clue i points to YES. Length M."""
    n_private: int
    """K: clues 0..K-1 are the private pool; clues K..M-1 are news."""
    news_steps: np.ndarray
    """`news_steps[j]` is the step at which news clue K + j is released. Non-decreasing."""
    clue_accuracy: float
    """q: the chance each clue points to the truth."""

    @property
    def n_clues(self) -> int:
        return len(self.points_yes)

    def news_released_by(self, step: int) -> np.ndarray:
        """Ids of news clues released at or before `step`."""
        n_released = int(np.searchsorted(self.news_steps, step, side="right"))
        return np.arange(self.n_private, self.n_private + n_released)

    def p_star(self, known_clues: Iterable[int]) -> float:
        """The Bayesian posterior that Y is YES from the real clues in `known_clues`.

        Each clue counts once, however often it appears.
        """
        return p_star(self.points_yes, known_clues, self.clue_accuracy)

    def p_star_at(self, step: int, private_known: Iterable[int]) -> float:
        """P* at `step`: the held private clues plus all news released so far."""
        private = {int(i) for i in private_known}
        if any(not 0 <= i < self.n_private for i in private):
            raise ValueError("private_known must contain only private-pool clue ids")
        return self.p_star(private.union(self.news_released_by(step).tolist()))


def make_world(config: Config, rng: np.random.Generator) -> World:
    """Draw Y by coin flip and each clue's direction, correct with probability q."""
    y = bool(rng.random() < 0.5)
    correct = rng.random(config.n_clues_total) < config.clue_accuracy
    points_yes = correct if y else ~correct
    points_yes.flags.writeable = False
    news_steps = news_schedule(config.n_clues_total - config.n_private, config.n_steps)
    return World(
        y=y,
        points_yes=points_yes,
        n_private=config.n_private,
        news_steps=news_steps,
        clue_accuracy=config.clue_accuracy,
    )


def news_schedule(n_news: int, n_steps: int) -> np.ndarray:
    """Evenly spaced release steps in 1..n_steps: news j comes out at (j + 1) T / (n + 1)."""
    steps = (np.arange(1, n_news + 1) * n_steps) // (n_news + 1)
    steps = np.maximum(steps, 1)
    steps.flags.writeable = False
    return steps


def deal_private_clues(
    n_clues: Iterable[int], n_private: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Give agent a `n_clues[a]` distinct clues drawn at random from the private pool.

    Requests above the pool size K are capped at K.
    """
    hands = []
    for count in n_clues:
        count = int(count)
        if count < 0:
            raise ValueError(f"n_clues must be non-negative, got {count}")
        hand = rng.choice(n_private, size=min(count, n_private), replace=False)
        hand.sort()
        hands.append(hand)
    return hands


def clue_log_odds(clue_accuracy: float) -> float:
    """Log-odds weight of one clue: ln(q / (1 - q))."""
    return math.log(clue_accuracy / (1 - clue_accuracy))


def p_star(points_yes: np.ndarray, known_clues: Iterable[int], clue_accuracy: float) -> float:
    """logit(P*) = (# YES - # NO known clues) x ln(q / (1 - q)), from a fair-coin prior."""
    ids = np.unique(np.fromiter(known_clues, dtype=np.int64))
    if ids.size and (ids[0] < 0 or ids[-1] >= len(points_yes)):
        raise ValueError("known_clues contains an id outside the clue list")
    n_yes = int(np.count_nonzero(points_yes[ids]))
    net = 2 * n_yes - ids.size
    return sigmoid(net * clue_log_odds(clue_accuracy))


def sigmoid(x: float) -> float:
    """1 / (1 + e^-x) without overflow for large |x|."""
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)
