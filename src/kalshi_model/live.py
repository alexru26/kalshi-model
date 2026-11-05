"""The logic behind the live GUI (`kalshi_model.gui`), kept free of any UI code.

- `FIELD_GROUPS` sorts every `Config` field into the sections the GUI shows.
- `PRESETS` are complete settings: Levels 0 to 2 (Level 2 is the frozen default of
  `configs/default.toml`), the E1 fee-free CLOB, the E2 disinformation ratios, and the
  E3 manipulators. Every preset starts from the defaults, so picking one resets the
  fields it does not set.
- `build_model` turns the GUI's settings into a fresh run, validated by the same code
  as `plot_run.py --config` (`custom.build_custom`).
- `latest_clue` reports who knows the most recently released news or fake clue.
- The chart functions draw a run in progress on its full step axis, so the lines grow
  from left to right as the run plays.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from kalshi_model.config import MANIPULATORS, MECHANISMS, NETWORKS, Config, clob_config
from kalshi_model.custom import build_custom, load_custom
from kalshi_model.model import MarketModel
from kalshi_model.network_plot import LINK, agent_graph
from kalshi_model.network_plot import _layout as network_layout
from kalshi_model.plot import (
    BAND,
    BAND_EDGE,
    FAKE,
    INK,
    MUTED,
    NEWS,
    P_STAR,
    PRICE,
    SURFACE,
    clue_events,
    style_axes,
)
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


def _axes(fig: Figure, n_steps: int):
    ax = fig.subplots()
    style_axes(ax)
    ax.set_xlim(0, n_steps + 1)
    ax.set_xlabel("step", color=INK)
    return ax


WIDTH = 5.2
"""Chart width in inches: at 100 dpi, about the width of one column of the page, so
the browser shows the text at its drawn size."""


def _figure() -> Figure:
    # A bare Figure (no pyplot) draws off-screen and holds no global state.
    return Figure(figsize=(WIDTH, 2.6), facecolor=SURFACE, layout="constrained")


def _legend(ax) -> None:
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK, ncol=3)


def price_figure(model: MarketModel) -> Figure:
    """Price and P* in cents, with the release of each news or fake clue marked."""
    data = series(model)
    steps = np.arange(1, len(data["price"]) + 1)
    fig = _figure()
    ax = _axes(fig, model.config.n_steps)
    ax.axhline(50, color=MUTED, linewidth=0.8, linestyle=(0, (2, 3)), zorder=1)
    for e in clue_events(model):
        ax.axvline(e.step, color=FAKE if e.fake else NEWS, linewidth=0.6, alpha=0.3)
    ax.plot(steps, 100 * data["price"], color=PRICE, linewidth=1.0, label="price")
    ax.plot(steps, 100 * data["p_star"], color=P_STAR, linewidth=2, label="P* (benchmark)")
    ax.set_ylim(0, 106)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("cents", color=INK)
    ax.set_title("Price vs P*", loc="left", color=INK, fontsize=11)
    handles = ax.get_legend_handles_labels()[0]
    config = model.config
    if config.n_clues_total > config.n_private:
        handles.append(Line2D([], [], color=NEWS, linewidth=0.8, label="news release"))
    if config.n_fakes:
        handles.append(Line2D([], [], color=FAKE, linewidth=0.8, label="fake release"))
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8, labelcolor=INK, ncol=4)
    return fig


def volume_figure(model: MarketModel) -> Figure:
    """Contracts traded each step."""
    volume = series(model)["volume"]
    fig = _figure()
    ax = _axes(fig, model.config.n_steps)
    steps = np.arange(1, len(volume) + 1)
    # One line per step: thousands of bar patches would draw too slowly to play live.
    ax.vlines(steps, 0, volume, color=PRICE, linewidth=0.8)
    top = np.nanmax(volume) if len(volume) else 0
    ax.set_ylim(0, max(1.0, float(top)) * 1.05)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.set_ylabel("contracts", color=INK)
    ax.set_title("Volume per step", loc="left", color=INK, fontsize=11)
    return fig


def book_figure(model: MarketModel) -> Figure | None:
    """Best bid, best ask, and price in cents above the spread; None without a book."""
    if model.book is None:
        return None
    data = series(model)
    steps = np.arange(1, len(data["price"]) + 1)
    fig = Figure(figsize=(WIDTH, 3.0), facecolor=SURFACE, layout="constrained")
    top, bottom = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
    for ax in (top, bottom):
        style_axes(ax)
        ax.set_xlim(0, model.config.n_steps + 1)
    bid, ask = data["best_bid"], data["best_ask"]
    top.fill_between(
        steps,
        bid,
        ask,
        where=np.isfinite(bid) & np.isfinite(ask),
        color=BAND,
        alpha=0.7,
        linewidth=0,
        label="best bid to best ask",
    )
    for edge in (bid, ask):
        top.plot(steps, edge, color=BAND_EDGE, linewidth=0.5, alpha=0.8)
    top.plot(steps, 100 * data["price"], color=PRICE, linewidth=1.0, label="price")
    top.set_ylim(0, 100)
    top.set_ylabel("cents", color=INK)
    top.set_title("Bid, ask, and spread", loc="left", color=INK, fontsize=11)
    _legend(top)
    spread = data["spread"]
    bottom.plot(steps, spread, color=PRICE, linewidth=0.8, drawstyle="steps-post")
    empty = steps[data["empty_book"] > 0]
    if len(empty):
        bottom.scatter(
            empty,
            np.zeros(len(empty)),
            marker="|",
            s=40,
            color=INK,
            label="one side empty",
            clip_on=False,
        )
        _legend(bottom)
    highest = np.nanmax(spread) if np.isfinite(spread).any() else 1.0
    bottom.set_ylim(0, max(2.0, 1.1 * highest))
    bottom.set_ylabel("spread (c)", color=INK)
    bottom.set_xlabel("step", color=INK)
    return fig


def market_maker_figure(model: MarketModel) -> Figure | None:
    """Book depth, the market maker's inventory, and its half-spread; None without a book."""
    if model.book is None:
        return None
    data = series(model)
    steps = np.arange(1, len(data["depth"]) + 1)
    fig = Figure(figsize=(WIDTH, 3.4), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(3, 1, sharex=True)
    panels = (
        ("depth", f"depth\n(±{model.config.depth_window}c)"),
        ("mm_position", "MM position\n(contracts)"),
        ("mm_half_spread", "MM half-\nspread (c)"),
    )
    for ax, (name, label) in zip(axes, panels, strict=True):
        style_axes(ax)
        ax.set_xlim(0, model.config.n_steps + 1)
        ax.plot(steps, data[name], color=PRICE, linewidth=0.9, drawstyle="steps-post")
        if not np.isfinite(data[name]).any():  # nothing recorded yet: a plain 0 to 1 axis
            ax.set_ylim(0, 1)
        ax.set_ylabel(label, color=INK, fontsize=9)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    axes[1].axhline(0, color=MUTED, linewidth=0.8, linestyle=(0, (2, 3)))
    axes[0].set_title("Depth and market maker", loc="left", color=INK, fontsize=11)
    axes[-1].set_xlabel("step", color=INK)
    return fig


SPREAD_COLORS = {UNKNOWN: "#d6d5d0", AT_RELEASE: INK, FROM_NEIGHBOR: PRICE}
"""Node colors: gray does not know the clue yet, black got it at release, blue heard it
(purple, the fake color, when the clue is a fake)."""


def network_layout_for(model: MarketModel) -> dict[int, np.ndarray]:
    """Node positions for a run's network: the same spring layout as `network_plot`."""
    return network_layout(agent_graph(model), model.seed)


def network_figure(model: MarketModel, layout: dict[int, np.ndarray]) -> Figure:
    """The agents' network, colored by who knows the most recently released clue."""
    graph = agent_graph(model)
    n = graph.number_of_nodes()
    xy = np.array([layout[i] for i in range(n)])
    fig = Figure(figsize=(WIDTH, 4.6), facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    ax.set_facecolor(SURFACE)
    ax.set_axis_off()
    if graph.number_of_edges():
        segments = [(layout[a], layout[b]) for a, b in graph.edges]
        ax.add_collection(LineCollection(segments, colors=LINK, linewidths=0.4, alpha=0.6))
    spread = latest_clue(model)
    state = np.full(n, UNKNOWN) if spread is None else spread.state
    heard = SPREAD_COLORS[FROM_NEIGHBOR]
    if spread is not None and spread.fake:
        heard = FAKE
    colors = [heard if s == FROM_NEIGHBOR else SPREAD_COLORS[int(s)] for s in state.tolist()]
    size = max(8.0, min(60.0, 6000 / max(n, 1)))
    ax.scatter(xy[:, 0], xy[:, 1], s=size, c=colors, edgecolors=SURFACE, linewidths=0.4)
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="datalim")

    config = model.config
    if config.n_clues_total == config.n_private and not config.n_fakes:
        title = "Who knows the latest clue: this run releases no news or fakes"
    elif spread is None:
        title = "Who knows the latest clue: no clue released yet"
    else:
        kind = "fake" if spread.fake else "news"
        side = "YES" if spread.points_yes else "NO"
        knows = int((state != UNKNOWN).sum())
        title = (
            f"Latest clue: {kind} #{spread.clue_id} (points {side}), released step "
            f"{spread.released}. {knows} of {n} agents know it"
        )
    ax.set_title(title, loc="left", color=INK, fontsize=10, wrap=True)
    handles = [
        Line2D([], [], marker="o", linestyle="", color=SPREAD_COLORS[UNKNOWN], label="not yet"),
        Line2D([], [], marker="o", linestyle="", color=INK, label="got it at release"),
        Line2D([], [], marker="o", linestyle="", color=heard, label="heard from a neighbor"),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        frameon=False,
        fontsize=8,
        labelcolor=INK,
        ncol=3,
        bbox_to_anchor=(0, -0.06),
    )
    return fig
