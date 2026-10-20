"""Agent trait population (PLAN.md Section 2.2).

Each trait is drawn from its own child of the run's "traits" stream, so changing one
trait's distribution never changes another trait's draws. Bounded traits use bounded
distributions, so nothing is clipped. The only links between traits are the four to
informedness: `w_hearsay` and `noise` fall and `patience` and `cash` rise with `n_clues`. The noise
link scales each agent's drawn noise, so it uses no extra draws. The cash
link reassigns the drawn cash amounts among agents (a Gaussian copula with correlation
`cash_rho`), so the cash distribution itself does not change, and it draws from a
separate child stream spawned after the traits' own, so every other trait is unchanged.
"""

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
    """Weight on relayed clues, in (0, 1)."""
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


def draw_traits(config: Config, rng: np.random.Generator) -> Traits:
    """Draw the trait vector of every agent from the config's distributions."""
    n = config.n_agents
    streams = dict(zip(TRAIT_NAMES, rng.spawn(len(TRAIT_NAMES)), strict=True))

    # Negative binomial with mean m and dispersion r: most agents know little, a few a lot.
    r = config.n_clues_dispersion
    n_clues = streams["n_clues"].negative_binomial(r, r / (r + config.n_clues_mean), n)

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

    return Traits(
        n_clues=n_clues.astype(np.int64),
        w_hearsay=w_hearsay,
        market_trust=market_trust,
        w_depth=w_depth,
        noise=noise,
        min_edge=min_edge,
        patience=patience,
        kelly_frac=kelly_frac,
        g=g,
        cash=cash,
        activity=activity,
    )


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
