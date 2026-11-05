"""The single configuration object for a model run.

Every parameter is a field of `Config`, and every field's frozen
default lives in `configs/default.toml`, read once at import. Edit that file to change a
default; the experiments save each run's full configuration with its results. The
trait settings and the market maker were calibrated against real
Kalshi patterns (`experiments/calibrate.py`, `results/calibration.csv`).
Only informedness and cash vary across agents; each other trait is one population
value. `g` (set from lab estimates), informedness, `w_hearsay` and
`w_depth` (no data anchor), and the network, legacy AMM, and order-upkeep settings are
assumptions or starting guesses, and the manipulator's settings (E3) are the
experimenter's choice.
"""

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

NETWORKS = ("holme_kim", "watts_strogatz")  # "watts_strogatz" only for small demos
MECHANISMS = ("amm", "clob")  # "amm" is a legacy feature, unused by the experiments
MANIPULATORS = ("none", "twin", "pump", "spoof")

DEFAULTS_FILE = Path(__file__).resolve().parents[2] / "configs" / "default.toml"
"""The frozen defaults: one line per `Config` field, nothing else."""


def _read_toml(path: Path) -> dict[str, object]:
    try:
        with Path(path).open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"{path}: {e}") from e


_DEFAULTS = _read_toml(DEFAULTS_FILE)


def _default(name: str) -> object:
    if name not in _DEFAULTS:
        raise ValueError(f"{DEFAULTS_FILE} is missing the setting {name!r}")
    return _DEFAULTS[name]


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


@dataclass(frozen=True, kw_only=True)
class Config:
    """Parameters of one run. Experiments vary one field at a time around the defaults,
    which come from `configs/default.toml`."""

    n_agents: int = _default("n_agents")
    n_steps: int = _default("n_steps")
    turns_per_step: int = _default("turns_per_step")
    """Agents drawn each step, without replacement (equal chances unless `activity` is
    pinned)."""

    # Market rules. E1: "clob" with no fees vs "clob" with Kalshi fees. "amm" (the LMSR, no
    # fees) is a legacy feature, no longer used by any experiment.
    mechanism: str = _default("mechanism")
    lmsr_b: float = _default("lmsr_b")
    """LMSR liquidity, in contracts: one contract moves a 50c price by 25 / b cents.
    Legacy: used only by the AMM."""

    # Fee rates in basis points (700 = 0.07). Kalshi's default market charges takers
    # 0.07 and makers nothing. The AMM has no fees, so it requires both to be 0.
    taker_fee_bp: int = _default("taker_fee_bp")
    maker_fee_bp: int = _default("maker_fee_bp")

    # Order book reading and upkeep (starting guesses).
    spread_cap: int = _default("spread_cap")
    """Widest spread, in cents, at which agents read the midpoint instead of the last trade."""
    depth_window: int = _default("depth_window")
    """Cents from each best price counted in book imbalance (must cover the spoofer's k)."""
    stale_ticks: int = _default("stale_ticks")
    """A resting order this many cents or more behind the best quote on its side is stale."""
    stale_steps: int = _default("stale_steps")
    """Steps a resting order may sit stale before its owner cancels it to reprice."""

    # Market-maker component in the CLOB. mm_size and
    # mm_half_spread were calibrated; the rest are starting guesses.
    mm_cash: int = _default("mm_cash")
    """Market maker's starting cash, in cents."""
    mm_size: int = _default("mm_size")
    """Contracts quoted on each side; 0 turns the market maker off."""
    mm_half_spread: float = _default("mm_half_spread")
    """Base half-spread, in cents."""
    mm_max_half_spread: float = _default("mm_max_half_spread")
    mm_alpha: float = _default("mm_alpha")
    """Weight of each new trade price in the market maker's moving price estimate."""
    mm_skew: float = _default("mm_skew")
    """Cents the quotes move against each contract of inventory."""
    mm_widen: float = _default("mm_widen")
    """Extra half-spread, in cents, per cent of markout loss per quoted contract."""
    mm_decay: float = _default("mm_decay")
    """Share of the extra half-spread kept from one step to the next."""
    mm_max_inventory: int = _default("mm_max_inventory")
    """Hard cap on the market maker's inventory, in contracts long or short."""

    # World: M clues, each pointing to the truth with probability
    # q. The first K are the private pool; the other M - K are released as news.
    n_clues_total: int = _default("n_clues_total")
    """M; 0 gives a world with no information at all (P* stays at 50%)."""
    n_private: int = _default("n_private")
    clue_accuracy: float = _default("clue_accuracy")
    n_news_seeds: int = _default("n_news_seeds")
    """Agents who receive each news (or fake) clue directly when it is released."""

    # Information layer: a Holme-Kim network. E2 varies
    # `disinfo_ratio`.
    network: str = _default("network")
    """"holme_kim" in every experiment. "watts_strogatz" (a ring, rewired with
    `ws_rewire`) is only for small demos: a ring of 3 is a triangle (`configs/level_1.toml`)."""
    mean_degree: int = _default("mean_degree")
    """Must be even: each new Holme-Kim node makes `mean_degree / 2` links, and each
    Watts-Strogatz node links to `mean_degree / 2` neighbors on each side.

    0 means no network at all: news stays with its seed agents (single-agent tests).
    """
    hk_triad: float = _default("hk_triad")
    """Holme-Kim chance of closing a triangle after each preferential link."""
    ws_rewire: float = _default("ws_rewire")
    """Watts-Strogatz rewiring probability (demos only)."""
    beta: float = _default("beta")
    """Chance per step that a spreader tells each neighbor."""
    disinfo_ratio: float = _default("disinfo_ratio")
    """Fake clues per true news clue (0, 0.05, 0.1, 0.2, 0.5 give 0, 1, 2, 4, 10 fakes)."""

    # Manipulator (E3). "none" in E1 and E2; "twin" adds the
    # manipulator's agent trading honestly throughout (E3's baseline, the honest twin);
    # "pump" and "spoof" add its manipulation layer from `manip_start`. Order book only.
    # Every setting is the experimenter's choice (starting guesses).
    manipulator: str = _default("manipulator")
    manip_n_clues: int = _default("manip_n_clues")
    """Private clues the manipulator holds: the population's median informedness."""
    manip_cash_share: float = _default("manip_cash_share")
    """Manipulator's starting cash as a share of all agents' starting cash on the seed."""
    manip_start: int = _default("manip_start")
    """Step on which the manipulation layer starts; before it, and after it ends, the
    manipulator follows the honest rules."""
    manip_chunk: int = _default("manip_chunk")
    """Most contracts the pump buys, or the dump sells, per turn."""
    pump_push: int = _default("pump_push")
    """Cents above the start price at which the pump stops buying."""
    pump_max_steps: int = _default("pump_max_steps")
    """Longest the pump buys, and the dump sells, before moving on."""
    pump_hold: int = _default("pump_hold")
    """Steps the pump holds its inventory before dumping it."""
    spoof_size: int = _default("spoof_size")
    """Contracts in each fake bid."""
    spoof_ticks: int = _default("spoof_ticks")
    """Cents behind the best bid at which the fake bid rests (at most `depth_window`)."""
    spoof_qty: int = _default("spoof_qty")
    """Contracts the spoofer offers for real at the best ask, reposted whenever its
    offer has filled or the best ask has moved away from it."""
    spoof_steps: int = _default("spoof_steps")
    """Steps the spoofing lasts."""

    # Agent traits. Only informedness and cash vary across agents;
    # every other trait is one value for the whole population. Informedness is
    # heavy-tailed: with mean 20 and dispersion 0.2, about 40% of agents hold no private
    # clue and the top 10% hold most of the pool. The price is roughly a wealth-weighted
    # average of beliefs, so how much of the cash the informed hold (cash_rho) sets how
    # far it moves toward P*. Calibrated except informedness, w_hearsay,
    # w_depth, and g.
    n_clues_mean: float = _default("n_clues_mean")
    n_clues_dispersion: float = _default("n_clues_dispersion")
    cash_median: int = _default("cash_median")
    """Median starting cash, in cents (lognormal)."""
    cash_sigma: float = _default("cash_sigma")
    """Log-scale spread of starting cash."""
    cash_rho: float = _default("cash_rho")
    """Rank correlation (Gaussian copula) of starting cash with informedness: market
    selection leaves more of the money with the well-informed."""
    w_hearsay: float = _default("w_hearsay")
    """Weight on fakes heard from neighbors, 0 to 1 (no data anchor)."""
    market_trust: float = _default("market_trust")
    """How many clues each agent thinks the price is worth."""
    w_depth: float = _default("w_depth")
    """Log-odds nudge per unit of book imbalance, 0 to 1 (no data anchor)."""
    noise: float = _default("noise")
    """Belief noise per turn, in log-odds."""
    min_edge: float = _default("min_edge")
    """Smallest edge worth acting on, in cents."""
    patience: float = _default("patience")
    """Chance of posting a resting order instead of waiting."""
    kelly_frac: float = _default("kelly_frac")
    """Fraction of the full Kelly bet placed."""
    g: float = _default("g")
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
        if self.network not in NETWORKS:
            raise ValueError(f"network must be one of {NETWORKS}, got {self.network!r}")
        if self.mean_degree % 2 or not 0 <= self.mean_degree < self.n_agents:
            raise ValueError("mean_degree must be even, non-negative, and below n_agents")
        for name in ("hk_triad", "ws_rewire", "beta"):
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


def load_defaults(path: Path = DEFAULTS_FILE) -> dict[str, object]:
    """Read a defaults file and check it sets every `Config` field and nothing else.

    Raises ValueError for a missing or unknown setting or an unreadable file."""
    data = _read_toml(path)
    names = {f.name for f in fields(Config)}
    missing, unknown = sorted(names - set(data)), sorted(set(data) - names)
    if missing:
        raise ValueError(f"{path} is missing setting(s): {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{path} has unknown setting(s): {', '.join(unknown)}")
    return data


load_defaults()
Config()  # fail at import, not mid-run, if configs/default.toml holds an invalid value


def clob_config(**overrides: object) -> Config:
    """The E1 fee-free CLOB: Kalshi's order book without fees, other defaults unchanged."""
    return Config(**{"mechanism": "clob", "taker_fee_bp": 0, "maker_fee_bp": 0, **overrides})


def amm_config(**overrides: object) -> Config:
    """The AMM market: the LMSR with no fees, other defaults unchanged.

    Legacy feature: dropped from E1, which now compares only the CLOB and Kalshi.
    """
    return Config(**{"mechanism": "amm", "taker_fee_bp": 0, "maker_fee_bp": 0, **overrides})
