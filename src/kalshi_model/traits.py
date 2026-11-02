"""Agent trait population (PLAN.md Section 2.2).

Each trait is drawn from its own child of the run's "traits" stream, so changing one
trait's distribution never changes another trait's draws. Bounded traits use bounded
distributions, so nothing is clipped. The only links between traits are the four to
informedness: `w_hearsay` and `noise` fall and `patience` and `cash` rise with `n_clues`. The noise
link scales each agent's drawn noise, so it uses no extra draws. The cash
link reassigns the drawn cash amounts among agents (a Gaussian copula with correlation
`cash_rho`), so the cash distribution itself does not change, and it draws from a
separate child stream spawned after the traits' own, so every other trait is unchanged.

`fixed` pins chosen traits to exact values (custom configurations, `kalshi_model.custom`).
Every trait is still drawn, so pinning one never changes another's draws, but a pinned
`n_clues` feeds the four links, and a pinned trait replaces the linked value.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from statistics import NormalDist

import numpy as np

from kalshi_model.config import Config

# Child index of each trait in the traits stream. Append-only, like rng.STREAM_NAMES.
TRAIT_NAMES = (
    "n_clues",
    "w_hearsay",
    "market_trust",
    "w_depth",
    "noise",
    "min_edge",
    "patience",
    "kelly_frac",
    "g",
    "cash",
    "activity",
)


@dataclass(frozen=True, eq=False)
class Traits:
    """One array per trait, indexed by agent. Arrays are read-only."""

    n_clues: np.ndarray
    """Private clues requested at setup (int, >= 0; capped at K when dealt)."""
    w_hearsay: np.ndarray
    """Weight on relayed fakes (unverifiable hearsay), in (0, 1)."""
    market_trust: np.ndarray
    """How many clues the agent thinks the price is worth, > 0."""
    w_depth: np.ndarray
    """Log-odds nudge per unit of book imbalance, in (0, 1)."""
    noise: np.ndarray
    """Standard deviation of belief error per turn, in log-odds, >= 0."""
    min_edge: np.ndarray
    """Smallest edge worth acting on, in cents, >= 0."""
    patience: np.ndarray
    """Chance of posting a resting order instead of waiting, in (0, 1)."""
    kelly_frac: np.ndarray
    """Fraction of the full Kelly bet placed, in (0, 1)."""
    g: np.ndarray
    """Probability weighting: 1 is none, below 1 overweights longshots."""
    cash: np.ndarray
    """Starting cash in integer cents, >= 1."""
    activity: np.ndarray
    """Relative chance of getting a turn, > 0."""

    def __post_init__(self) -> None:
        sizes = {len(getattr(self, f.name)) for f in fields(self)}
        if len(sizes) != 1:
            raise ValueError("every trait array must have one entry per agent")
        for f in fields(self):
            getattr(self, f.name).flags.writeable = False

    def __len__(self) -> int:
        return len(self.n_clues)


# Allowed values of each trait when pinned: (low, high, low_inclusive, integer). The
# upper bound is inclusive.
TRAIT_RANGES: dict[str, tuple[float, float, bool, bool]] = {
    "n_clues": (0, math.inf, True, True),
    "w_hearsay": (0, 1, True, False),
    "market_trust": (0, math.inf, True, False),
    "w_depth": (0, 1, True, False),
    "noise": (0, math.inf, True, False),
    "min_edge": (0, math.inf, True, False),
    "patience": (0, 1, True, False),
    "kelly_frac": (0, 1, True, False),
    "g": (0, 1, False, False),
    "cash": (1, math.inf, True, True),
    "activity": (0, math.inf, False, False),
}


def fixed_values(name: str, value: object, n_agents: int) -> np.ndarray:
    """Check a pinned trait and expand it to one value per agent.

    `value` is one number for every agent or a sequence with one number per agent.
    Raises ValueError for an unknown trait, a wrong length, or a value out of range.
    """
    if name not in TRAIT_RANGES:
        raise ValueError(f"unknown trait {name!r}; traits are {', '.join(TRAIT_NAMES)}")
    low, high, low_inclusive, integer = TRAIT_RANGES[name]
    values = list(value) if isinstance(value, Sequence) and not isinstance(value, str) else None
    if values is None:
        values = [value] * n_agents
    elif len(values) != n_agents:
        raise ValueError(f"trait {name!r} has {len(values)} values for {n_agents} agents")
    for v in values:
        is_number = isinstance(v, int | float) and not isinstance(v, bool)
        if not is_number or (integer and not isinstance(v, int)):
            kind = "an integer" if integer else "a number"
            raise ValueError(f"trait {name!r} must be {kind}, got {v!r}")
        if not (low <= v if low_inclusive else low < v) or not v <= high:
            raise ValueError(f"trait {name!r} is out of range: {v!r}")
    return np.array(values, dtype=np.int64 if integer else np.float64)


def draw_traits(
    config: Config,
    rng: np.random.Generator,
    fixed: Mapping[str, object] | None = None,
) -> Traits:
    """Draw the trait vector of every agent from the config's distributions.

    `fixed` maps trait names to pinned values (see `fixed_values`).
    """
    n = config.n_agents
    pins = {name: fixed_values(name, value, n) for name, value in (fixed or {}).items()}
    streams = dict(zip(TRAIT_NAMES, rng.spawn(len(TRAIT_NAMES)), strict=True))

    # Negative binomial with mean m and dispersion r: most agents know little, a few a lot.
    r = config.n_clues_dispersion
    n_clues = streams["n_clues"].negative_binomial(r, r / (r + config.n_clues_mean), n)
    n_clues = pins.get("n_clues", n_clues)

    w_hearsay = _logit_normal(
        streams["w_hearsay"],
        config.hearsay_mu - config.hearsay_gamma * n_clues,
        config.hearsay_sigma,
    )
    market_trust = streams["market_trust"].gamma(
        config.market_trust_shape, config.market_trust_scale, n
    )
    w_depth = streams["w_depth"].beta(config.depth_alpha, config.depth_beta, n)
    noise = np.abs(streams["noise"].normal(0.0, config.noise_scale, n)) * np.exp(
        -config.noise_delta * n_clues
    )
    min_edge = np.abs(streams["min_edge"].normal(0.0, config.min_edge_scale, n))
    patience = _logit_normal(
        streams["patience"],
        config.patience_mu + config.patience_delta * n_clues,
        config.patience_sigma,
    )
    kelly_frac = streams["kelly_frac"].beta(config.kelly_alpha, config.kelly_beta, n)
    g = config.g_low + (config.g_high - config.g_low) * streams["g"].beta(
        config.g_alpha, config.g_beta, n
    )
    cash = np.maximum(
        1, np.rint(streams["cash"].lognormal(np.log(config.cash_median), config.cash_sigma, n))
    ).astype(np.int64)
    if config.cash_rho:
        (link,) = rng.spawn(1)  # child after the TRAIT_NAMES children
        cash = link_to_informedness(cash, n_clues, config.cash_rho, link)
    activity = streams["activity"].lognormal(0.0, config.activity_sigma, n)

    drawn = {
        "n_clues": n_clues.astype(np.int64),
        "w_hearsay": w_hearsay,
        "market_trust": market_trust,
        "w_depth": w_depth,
        "noise": noise,
        "min_edge": min_edge,
        "patience": patience,
        "kelly_frac": kelly_frac,
        "g": g,
        "cash": cash,
        "activity": activity,
    }
    return Traits(**(drawn | pins))


def link_to_informedness(
    values: np.ndarray, n_clues: np.ndarray, rho: float, rng: np.random.Generator
) -> np.ndarray:
    """Reassign `values` among agents so their ranks follow informedness with Gaussian
    copula correlation `rho`. Ties in `n_clues` are broken at random."""
    n = len(values)
    tie_break = rng.random(n)
    ranks = np.lexsort((tie_break, n_clues)).argsort()
    normal = NormalDist()
    scores = np.array([normal.inv_cdf((r + 0.5) / n) for r in ranks.tolist()])
    latent = rho * scores + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    return np.sort(values)[latent.argsort().argsort()]


def _logit_normal(rng: np.random.Generator, mu: np.ndarray, sigma: float) -> np.ndarray:
    z = rng.normal(0.0, 1.0, np.shape(mu))
    return 1.0 / (1.0 + np.exp(-(mu + sigma * z)))
