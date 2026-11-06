"""The logic behind the live GUI (`kalshi_model.gui`), kept free of any UI code.

- `FIELD_GROUPS` sorts every `Config` field into the sections the GUI shows.
- `PRESETS` are complete settings: Levels 0 to 2 (Level 2 is the frozen default of
  `configs/default.toml`), the E1 fee-free CLOB, the E2 disinformation ratios, and the
  E3 manipulators. Every preset starts from the defaults, so picking one resets the
  fields it does not set.
- `build_model` turns the GUI's settings into a fresh run, validated by the same code
  as `plot_run.py --config` (`custom.build_custom`).
- `latest_clue` reports who knows the most recently released news or fake clue.
- `series` reads the recorded columns for the live charts (`live_charts`).
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from kalshi_model.config import MANIPULATORS, MECHANISMS, NETWORKS, Config, clob_config
from kalshi_model.custom import build_custom, load_custom
from kalshi_model.model import MarketModel
from kalshi_model.network_plot import _layout as network_layout
from kalshi_model.network_plot import agent_graph
from kalshi_model.traits import TRAIT_NAMES

CONFIGS = Path(__file__).resolve().parents[2] / "configs"

FIELD_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Run", ("n_agents", "n_steps", "turns_per_step")),
    ("Market rules", ("mechanism", "lmsr_b", "taker_fee_bp", "maker_fee_bp")),
    ("Order book", ("spread_cap", "depth_window", "stale_ticks", "stale_steps")),
    (
        "Market maker",
        (
            "mm_cash",
            "mm_size",
            "mm_half_spread",
            "mm_max_half_spread",
            "mm_alpha",
            "mm_skew",
            "mm_widen",
            "mm_decay",
            "mm_max_inventory",
        ),
    ),
    ("World", ("n_clues_total", "n_private", "clue_accuracy", "n_news_seeds")),
    (
        "Information",
        ("network", "mean_degree", "hk_triad", "ws_rewire", "beta", "disinfo_ratio"),
    ),
    (
        "Manipulator",
        (
            "manipulator",
            "manip_n_clues",
            "manip_cash_share",
            "manip_start",
            "manip_chunk",
            "pump_push",
            "pump_max_steps",
            "pump_hold",
            "spoof_size",
            "spoof_ticks",
            "spoof_qty",
            "spoof_steps",
        ),
    ),
    (
        "Agent traits",
        (
            "n_clues_mean",
            "n_clues_dispersion",
            "cash_median",
            "cash_sigma",
            "cash_rho",
            "w_hearsay",
            "market_trust",
            "w_depth",
            "noise",
            "min_edge",
            "patience",
            "kelly_frac",
            "g",
        ),
    ),
)
"""Every `Config` field, once, in the section the GUI shows it under."""

CHOICES: dict[str, tuple[str, ...]] = {
    "mechanism": MECHANISMS,
    "network": NETWORKS,
    "manipulator": MANIPULATORS,
}
"""Fields with a fixed set of values, shown as a dropdown."""

FIELD_TYPES: dict[str, type] = {f.name: f.type for f in fields(Config)}
"""Each field's type: int, float, or str (the dataclass annotations are real types)."""

PIN_TRAITS: tuple[str, ...] = tuple(name for name in TRAIT_NAMES if name not in FIELD_TYPES)
"""The traits the GUI offers pins for: those with no `Config` field of their own
(n_clues, cash, activity). The others are one population value set by their field
under "Agent traits", so a pin would only repeat it."""

DEFAULT_SEED = 20000
"""The seed the GUI starts on: the first seed of `plot_run.py`."""


@dataclass(frozen=True)
class Preset:
    """A full set of GUI settings: every Config field and any pinned traits."""

    values: dict[str, object]
    pins: dict[str, object] = field(default_factory=dict)


def default_values() -> dict[str, object]:
    """Every Config field at its frozen default."""
    return asdict(Config())


def _from_file(name: str) -> Preset:
    custom = load_custom(CONFIGS / name)
    return Preset(asdict(custom.config), dict(custom.fixed_traits))


def _with(config: Config) -> Preset:
    return Preset(asdict(config))


E2_RATIOS = (0.0, 0.05, 0.1, 0.2, 0.5)
"""The E2 disinformation ratios (`experiments/e2_information.py`)."""

PRESETS: dict[str, Preset] = {
    "Level 0": _from_file("level_0.toml"),
    "Level 1": _from_file("level_1.toml"),
    "Level 2 (default)": _with(Config()),
    "E1: fee-free CLOB": _with(clob_config()),
    **{f"E2: disinformation ratio {r}": _with(Config(disinfo_ratio=r)) for r in E2_RATIOS},
    "E3: honest twin": _with(Config(manipulator="twin")),
    "E3: pump and dump": _with(Config(manipulator="pump")),
    "E3: spoofing": _with(Config(manipulator="spoof")),
}
DEFAULT_PRESET = "Level 2 (default)"


def format_pin(value: object) -> str:
    """A pinned trait as the GUI's text: `5`, `0.5`, or `[20, 0]` (TOML syntax)."""
    if isinstance(value, list | tuple):
        return "[" + ", ".join(format_pin(v) for v in value) + "]"
    return repr(value) if isinstance(value, float) else str(value)


def parse_pin(name: str, text: str) -> object | None:
    """Read a trait pin typed in the GUI: blank for none, else one number for every agent
    or a TOML list with one number per agent. Raises ValueError for anything else."""
    text = text.strip()
    if not text:
        return None
    try:
        value = tomllib.loads(f"value = {text}")["value"]
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"trait {name}: expected a number or a list like [20, 0]") from e
    items = value if isinstance(value, list) else [value]
    if not items or not all(isinstance(v, int | float) and not isinstance(v, bool) for v in items):
        raise ValueError(f"trait {name}: expected a number or a list like [20, 0]")
    return value


def build_model(values: dict[str, object], pins: dict[str, object], seed: int) -> MarketModel:
    """A fresh run at step 0. Raises ValueError for any invalid setting or pin."""
    unknown = sorted(set(pins) - set(TRAIT_NAMES))
    if unknown:
        raise ValueError(f"unknown trait(s): {', '.join(unknown)}")
    custom = build_custom({**values, "traits": dict(pins)})
    return MarketModel(custom.config, seed, fixed_traits=custom.fixed_traits)


@dataclass(frozen=True)
class ClueSpread:
    """Who knows the most recently released news or fake clue."""

    clue_id: int
    fake: bool
    points_yes: bool
    released: int
    """Step the clue was released."""
    state: np.ndarray
    """Per agent: 0 does not know it yet, 1 received it at release, 2 heard it from a
    neighbor."""


UNKNOWN, AT_RELEASE, FROM_NEIGHBOR = 0, 1, 2


def latest_clue(model: MarketModel) -> ClueSpread | None:
    """The newest released clue and each agent's knowledge of it; None before any release.

    Release seeds learn a clue on its release step. Nobody else can: an agent only starts
    spreading a clue after its next turn, so neighbors learn it from the next step on.
    """
    if not model.release_step:
        return None
    clue_id, released = max(model.release_step.items(), key=lambda item: (item[1], item[0]))
    state = np.full(len(model.traders), UNKNOWN, dtype=np.int8)
    for agent, step in model.learned_step.get(clue_id, {}).items():
        state[agent] = AT_RELEASE if step == released else FROM_NEIGHBOR
    fake = clue_id in set(model.fake_ids)
    return ClueSpread(clue_id, fake, model.points_yes(clue_id), released, state)


def series(model: MarketModel) -> dict[str, np.ndarray]:
    """Each recorded column as a float array (None is NaN), cut to the shortest column so
    a step being recorded while this reads cannot misalign them."""
    columns = dict(model.datacollector.model_vars)
    n = min((len(v) for v in columns.values()), default=0)
    return {
        name: np.array([np.nan if v is None else v for v in values[:n]], dtype=float)
        for name, values in columns.items()
    }


def network_layout_for(model: MarketModel) -> dict[int, np.ndarray]:
    """Node positions for a run's network: the same spring layout as `network_plot`."""
    return network_layout(agent_graph(model), model.seed)
