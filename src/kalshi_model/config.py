"""The single configuration object for a model run.

Every parameter lives here (PLAN.md Section 4). Values are starting defaults until
calibration (PLAN.md Section 4, Step 7) freezes them. Later build steps add their
parameters to this class.
"""

from dataclasses import dataclass, fields

MECHANISMS = ("amm", "clob")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


@dataclass(frozen=True, kw_only=True)
class Config:
    """Parameters of one run. Experiments vary one field at a time around the defaults."""

    n_agents: int = 300
    n_steps: int = 2000
    turns_per_step: int = 20
    """Agents drawn each step, without replacement, with chance weighted by `activity`."""

    # Market rules. E1: "amm" with no fees; "clob" with no fees; "clob" with Kalshi fees.
    mechanism: str = "clob"
    lmsr_b: float = 50.0
    """LMSR liquidity, in contracts: one contract moves a 50c price by 25 / b cents."""

    # Fee rates in basis points (700 = 0.07). Kalshi's default market charges takers
    # 0.07 and makers nothing. The AMM has no fees, so it requires both to be 0.
    taker_fee_bp: int = 700
    maker_fee_bp: int = 0

    # Order book reading and upkeep (PLAN.md Section 2.2; starting guesses).
    spread_cap: int = 10
    """Widest spread, in cents, at which agents read the midpoint instead of the last trade."""
    depth_window: int = 2
    """Cents from each best price counted in book imbalance (must cover the spoofer's k)."""
    stale_ticks: int = 3
    """A resting order this many cents or more behind the best quote on its side is stale."""
    stale_steps: int = 20
    """Steps a resting order may sit stale before its owner cancels it to reprice."""

    # Market-maker component in the CLOB (PLAN.md Section 2.1; starting guesses).
    mm_cash: int = 1_000_000
    """Market maker's starting cash, in cents."""
    mm_size: int = 10
    """Contracts quoted on each side."""
    mm_half_spread: float = 2.0
    """Base half-spread, in cents."""
    mm_max_half_spread: float = 10.0
    mm_alpha: float = 0.1
    """Weight of each new trade price in the market maker's moving price estimate."""
    mm_skew: float = 0.05
    """Cents the quotes move against each contract of inventory."""
    mm_widen: float = 0.5
    """Extra half-spread, in cents, per cent of markout loss per quoted contract."""
    mm_decay: float = 0.9
    """Share of the extra half-spread kept from one step to the next."""
    mm_max_inventory: int = 200
    """Beyond this many contracts long or short, the market maker quotes only to unwind."""

    # World (PLAN.md Section 2.4): M clues, each pointing to the truth with probability
    # q. The first K are the private pool; the other M - K are released as news.
    n_clues_total: int = 80
    n_private: int = 60
    clue_accuracy: float = 0.53
    n_news_seeds: int = 5
    """Agents who receive each news clue directly when it is released."""

    # Trait distributions (PLAN.md Section 2.2), starting guesses until calibration.
    # Informedness is heavy-tailed: with mean 20 and dispersion 0.2, about 40% of agents
    # hold no private clue and the top 10% hold most of the pool. Beliefs are averages, so
    # the price is roughly a weighted average of agents' posteriors; how much of the pool
    # the informed hold sets how far it can move toward P*.
    n_clues_mean: float = 20.0
    n_clues_dispersion: float = 0.2
    hearsay_mu: float = 0.0
    hearsay_gamma: float = 0.05
    hearsay_sigma: float = 1.0
    market_trust_shape: float = 2.0
    market_trust_scale: float = 1.5
    depth_alpha: float = 1.0
    depth_beta: float = 9.0
    noise_scale: float = 0.1
    min_edge_scale: float = 2.0
    """Half-normal scale of `min_edge`, in cents."""
    patience_mu: float = 0.0
    patience_delta: float = 0.05
    patience_sigma: float = 1.0
    kelly_alpha: float = 2.0
    kelly_beta: float = 8.0
    g_low: float = 0.6
    g_high: float = 1.0
    g_alpha: float = 2.0
    g_beta: float = 2.0
    cash_median: int = 10_000
    """Median starting cash, in cents."""
    cash_sigma: float = 1.0
    activity_sigma: float = 1.0

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if f.type is int and not _is_int(value):
                raise ValueError(f"{f.name} must be an integer, got {value!r}")
            if f.type is float and not _is_real(value):
                raise ValueError(f"{f.name} must be a number, got {value!r}")

        positive = (
            "n_agents n_steps turns_per_step n_clues_total lmsr_b n_clues_mean "
            "n_clues_dispersion market_trust_shape market_trust_scale depth_alpha "
            "depth_beta kelly_alpha kelly_beta g_alpha g_beta cash_median spread_cap "
            "stale_ticks stale_steps mm_size mm_half_spread mm_max_half_spread mm_alpha "
            "mm_max_inventory"
        )
        non_negative = (
            "taker_fee_bp maker_fee_bp n_private n_news_seeds hearsay_gamma hearsay_sigma "
            "noise_scale min_edge_scale patience_delta patience_sigma cash_sigma activity_sigma "
            "depth_window mm_cash mm_skew mm_widen mm_decay"
        )
        for name in positive.split():
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)!r}")
        for name in non_negative.split():
            if not getattr(self, name) >= 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)!r}")

        if self.mechanism not in MECHANISMS:
            raise ValueError(f"mechanism must be one of {MECHANISMS}, got {self.mechanism!r}")
        if self.mechanism == "amm" and (self.taker_fee_bp or self.maker_fee_bp):
            raise ValueError("the AMM has no fees: set taker_fee_bp and maker_fee_bp to 0")
        if self.turns_per_step > self.n_agents:
            raise ValueError("turns_per_step cannot exceed n_agents")
        if self.n_news_seeds > self.n_agents:
            raise ValueError("n_news_seeds cannot exceed n_agents")
        if self.n_private > self.n_clues_total:
            raise ValueError(
                f"n_private ({self.n_private}) cannot exceed n_clues_total ({self.n_clues_total})"
            )
        if self.n_clues_total - self.n_private > self.n_steps:
            raise ValueError("there must be at least one step per news clue")
        if not isinstance(self.clue_accuracy, float) or not 0.5 < self.clue_accuracy < 1:
            raise ValueError(
                "clue_accuracy must be a float strictly between 0.5 and 1, "
                f"got {self.clue_accuracy!r}"
            )
        if self.mm_alpha > 1 or self.mm_decay >= 1:
            raise ValueError("mm_alpha must be at most 1 and mm_decay below 1")
        if self.mm_half_spread > self.mm_max_half_spread:
            raise ValueError("mm_half_spread cannot exceed mm_max_half_spread")
        if not 0 < self.g_low <= self.g_high <= 1:
            raise ValueError("g must satisfy 0 < g_low <= g_high <= 1")


def clob_config(**overrides: object) -> Config:
    """The E1 fee-free CLOB: Kalshi's order book without fees, other defaults unchanged."""
    return Config(**{"mechanism": "clob", "taker_fee_bp": 0, "maker_fee_bp": 0, **overrides})


def amm_config(**overrides: object) -> Config:
    """The E1 AMM market: the LMSR with no fees, other defaults unchanged."""
    return Config(**{"mechanism": "amm", "taker_fee_bp": 0, "maker_fee_bp": 0, **overrides})
