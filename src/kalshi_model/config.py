"""The single configuration object for a model run.

Every parameter lives here (PLAN.md Section 4). The trait settings and the market maker
were calibrated in Steps 7, 8b, and 8c against real Kalshi patterns
(`experiments/calibrate.py`, `results/calibration.csv`) and are frozen for all
experiments. Since Step 8c only informedness and cash vary across agents; each other
trait is one population value. `g` (set from lab estimates), informedness (set in Step
3), `w_hearsay` and `w_depth` (no data anchor), and the network, legacy AMM, and
order-upkeep settings are assumptions or starting guesses, and the manipulator's
settings (E3) are the experimenter's choice.
"""

from dataclasses import dataclass, fields

MECHANISMS = ("amm", "clob")  # "amm" is a legacy feature, unused by the experiments
MANIPULATORS = ("none", "twin", "pump", "spoof")


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
    """Agents drawn each step, without replacement (equal chances unless `activity` is
    pinned)."""

    # Market rules. E1: "clob" with no fees vs "clob" with Kalshi fees. "amm" (the LMSR, no
    # fees) is a legacy feature, no longer used by any experiment.
    mechanism: str = "clob"
    lmsr_b: float = 50.0
    """LMSR liquidity, in contracts: one contract moves a 50c price by 25 / b cents.
    Legacy: used only by the AMM."""

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

    # Market-maker component in the CLOB (PLAN.md Section 2.1). mm_size and
    # mm_half_spread were calibrated (Steps 7, 8b, and 8c); the rest are starting guesses.
    mm_cash: int = 1_000_000
    """Market maker's starting cash, in cents."""
    mm_size: int = 36
    """Contracts quoted on each side; 0 turns the market maker off."""
    mm_half_spread: float = 2.72
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
    """Hard cap on the market maker's inventory, in contracts long or short."""

    # World (PLAN.md Section 2.4): M clues, each pointing to the truth with probability
    # q. The first K are the private pool; the other M - K are released as news.
    n_clues_total: int = 80
    """M; 0 gives a world with no information at all (P* stays at 50%)."""
    n_private: int = 60
    clue_accuracy: float = 0.53
    n_news_seeds: int = 5
    """Agents who receive each news (or fake) clue directly when it is released."""

    # Information layer (PLAN.md Section 2.4): a Holme-Kim network. E2 varies
    # `disinfo_ratio`.
    mean_degree: int = 8
    """Must be even: each new node makes `mean_degree / 2` links.

    0 means no network at all: news stays with its seed agents (single-agent tests).
    """
    hk_triad: float = 0.5
    """Holme-Kim chance of closing a triangle after each preferential link."""
    beta: float = 0.5
    """Chance per step that a spreader tells each neighbor."""
    disinfo_ratio: float = 0.0
    """Fake clues per true news clue (0, 0.05, 0.1, 0.2, 0.5 give 0, 1, 2, 4, 10 fakes)."""

    # Manipulator (PLAN.md Section 2.3; E3). "none" in E1 and E2; "twin" adds the
    # manipulator's agent trading honestly throughout (E3's baseline, the honest twin);
    # "pump" and "spoof" add its manipulation layer from `manip_start`. Order book only.
    # Every setting is the experimenter's choice (starting guesses, PLAN.md Section 2.3).
    manipulator: str = "none"
    manip_n_clues: int = 2
    """Private clues the manipulator holds: the population's median informedness."""
    manip_cash_share: float = 0.05
    """Manipulator's starting cash as a share of all agents' starting cash on the seed."""
    manip_start: int = 1000
    """Step on which the manipulation layer starts; before it, and after it ends, the
    manipulator follows the honest rules."""
    manip_chunk: int = 2000
    """Most contracts the pump buys, or the dump sells, per turn."""
    pump_push: int = 5
    """Cents above the start price at which the pump stops buying."""
    pump_max_steps: int = 100
    """Longest the pump buys, and the dump sells, before moving on."""
    pump_hold: int = 50
    """Steps the pump holds its inventory before dumping it."""
    spoof_size: int = 20_000
    """Contracts in each fake bid."""
    spoof_ticks: int = 2
    """Cents behind the best bid at which the fake bid rests (at most `depth_window`)."""
    spoof_qty: int = 2000
    """Contracts the spoofer offers for real at the best ask, reposted whenever its
    offer has filled or the best ask has moved away from it."""
    spoof_steps: int = 500
    """Steps the spoofing lasts."""

    # Agent traits (PLAN.md Section 2.2). Only informedness and cash vary across agents;
    # every other trait is one value for the whole population. Informedness is
    # heavy-tailed: with mean 20 and dispersion 0.2, about 40% of agents hold no private
    # clue and the top 10% hold most of the pool. The price is roughly a wealth-weighted
    # average of beliefs, so how much of the cash the informed hold (cash_rho) sets how
    # far it moves toward P*. Calibrated in Step 8c except informedness, w_hearsay,
    # w_depth, and g.
    n_clues_mean: float = 20.0
    n_clues_dispersion: float = 0.2
    cash_median: int = 233_400
    """Median starting cash, in cents (lognormal)."""
    cash_sigma: float = 1.30
    """Log-scale spread of starting cash."""
    cash_rho: float = 0.625
    """Rank correlation (Gaussian copula) of starting cash with informedness: market
    selection leaves more of the money with the well-informed (PLAN.md Section 2.2)."""
    w_hearsay: float = 0.39
    """Weight on fakes heard from neighbors, 0 to 1 (no data anchor)."""
    market_trust: float = 35.3
    """How many clues each agent thinks the price is worth."""
    w_depth: float = 0.1
    """Log-odds nudge per unit of book imbalance, 0 to 1 (no data anchor)."""
    noise: float = 0.0506
    """Belief noise per turn, in log-odds."""
    min_edge: float = 2.22
    """Smallest edge worth acting on, in cents."""
    patience: float = 0.902
    """Chance of posting a resting order instead of waiting."""
    kelly_frac: float = 0.274
    """Fraction of the full Kelly bet placed."""
    g: float = 0.8
    """Probability weighting: 1 is none, below 1 overweights longshots (lab estimates)."""

    @property
    def n_fakes(self) -> int:
        """Fake clues the disinformation source releases: disinfo_ratio x true news."""
        return round(self.disinfo_ratio * (self.n_clues_total - self.n_private))

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if f.type is int and not _is_int(value):
                raise ValueError(f"{f.name} must be an integer, got {value!r}")
            if f.type is float and not _is_real(value):
                raise ValueError(f"{f.name} must be a number, got {value!r}")

        positive = (
            "n_agents n_steps turns_per_step lmsr_b n_clues_mean "
            "n_clues_dispersion cash_median spread_cap "
            "stale_ticks stale_steps mm_half_spread mm_max_half_spread mm_alpha "
            "mm_max_inventory manip_cash_share manip_start manip_chunk pump_push "
            "pump_max_steps spoof_size spoof_ticks spoof_qty spoof_steps"
        )
        non_negative = (
            "taker_fee_bp maker_fee_bp n_clues_total n_private n_news_seeds market_trust "
            "noise min_edge cash_sigma depth_window mm_cash mm_size mm_skew "
            "mm_widen mm_decay manip_n_clues pump_hold"
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
        if self.mean_degree % 2 or not 0 <= self.mean_degree < self.n_agents:
            raise ValueError("mean_degree must be even, non-negative, and below n_agents")
        for name in ("hk_triad", "beta"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.disinfo_ratio < 0:
            raise ValueError("disinfo_ratio must be non-negative")
        if self.mm_alpha > 1 or self.mm_decay >= 1:
            raise ValueError("mm_alpha must be at most 1 and mm_decay below 1")
        if self.mm_half_spread > self.mm_max_half_spread:
            raise ValueError("mm_half_spread cannot exceed mm_max_half_spread")
        if not -1 < self.cash_rho < 1:
            raise ValueError("cash_rho must be strictly between -1 and 1")
        for name in ("w_hearsay", "w_depth", "patience", "kelly_frac"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if not 0 < self.g <= 1:
            raise ValueError("g must satisfy 0 < g <= 1")
        if self.manipulator not in MANIPULATORS:
            raise ValueError(f"manipulator must be one of {MANIPULATORS}, got {self.manipulator!r}")
        if self.manipulator != "none" and self.mechanism != "clob":
            raise ValueError("the manipulator trades only in the order book (mechanism 'clob')")
        if self.manipulator != "none" and self.manip_start > self.n_steps:
            raise ValueError("manip_start cannot be after the last step")
        if self.manipulator != "none" and self.manip_n_clues > self.n_private:
            raise ValueError("manip_n_clues cannot exceed n_private")
        if self.manip_cash_share >= 1:
            raise ValueError("manip_cash_share must be below 1")
        if self.pump_push > 98:
            raise ValueError("pump_push cannot exceed 98 cents")
        if self.spoof_ticks > self.depth_window:
            raise ValueError(
                "spoof_ticks cannot exceed depth_window: agents would never see the fake bid"
            )


def clob_config(**overrides: object) -> Config:
    """The E1 fee-free CLOB: Kalshi's order book without fees, other defaults unchanged."""
    return Config(**{"mechanism": "clob", "taker_fee_bp": 0, "maker_fee_bp": 0, **overrides})


def amm_config(**overrides: object) -> Config:
    """The AMM market: the LMSR with no fees, other defaults unchanged.

    Legacy feature: dropped from E1, which now compares only the CLOB and Kalshi.
    """
    return Config(**{"mechanism": "amm", "taker_fee_bp": 0, "maker_fee_bp": 0, **overrides})
