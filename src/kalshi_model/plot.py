"""A chart of one finished run: price over time against P*, with clue releases marked.

Panels share the step axis, and each has one y-scale:

1. Price and P* in cents, the bid-ask band (order book only), a 50c guide, and a marker
   at each news or fake clue's release (up for a YES clue, down for a NO clue). The
   outcome, final price, and final P* are labeled at the right edge. The price is the
   scored price (`MarketModel.price`): the bid-ask midpoint when the spread is within
   `spread_cap`, else the last trade; the LMSR price under the legacy AMM.
2. Contracts traded per bin of steps.
3. Order book only: the bid-ask spread (best ask minus best bid) in cents, with steps
   where one side of the book is empty marked along the bottom.

Every panel carries a faint vertical guide at each clue's release.

Row r of the model's data is step r + 1, the same convention as `metrics.news_events`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from kalshi_model.model import MarketModel

VOLUME_BIN = 20
"""Steps per volume bar."""

# Reference palette (dataviz skill): categorical slots 1, 2, and 7, and text tokens.
PRICE = "#2a78d6"
P_STAR = "#eb6834"
BAND = "#86b6ef"  # sequential blue step 250, lighter than the price line it surrounds
BAND_EDGE = "#256abf"  # sequential blue step 500
FAKE = "#6250d6"
NEWS = "#52514e"
INK = "#0b0b0b"
MUTED = "#8a8985"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


@dataclass(frozen=True)
class ClueEvent:
    """One clue released during the run."""

    clue_id: int
    step: int
    points_yes: bool
    fake: bool


def clue_events(model: MarketModel) -> list[ClueEvent]:
    """Every news and fake clue the run released, in release order."""
    fakes = set(model.fake_ids)
    return [
        ClueEvent(clue_id, step, model.points_yes(clue_id), clue_id in fakes)
        for clue_id, step in sorted(model.release_step.items(), key=lambda item: item[::-1])
    ]


def run_label(model: MarketModel) -> str:
    """The market, seed, network, and disinformation ratio of a run, for chart titles."""
    config = model.config
    if model.book is None:  # legacy AMM
        market = "AMM (LMSR)"
    else:
        market = "Kalshi" if config.taker_fee_bp else "CLOB, no fees"
    network = "Holme-Kim network" if config.mean_degree else "no network"
    label = f"{market}, seed {model.seed}, {network}"
    if config.n_fakes:
        label += f", disinformation ratio {config.disinfo_ratio}"
    return label


def price_chart(model: MarketModel, title: str | None = None) -> Figure:
    """Draw a finished run. Raises ValueError if the run has not settled."""
    if not model.settled:
        raise ValueError("run the model to settlement before plotting it")
    df = model.datacollector.get_model_vars_dataframe()
    steps = np.arange(1, len(df) + 1)
    clob = model.book is not None

    heights = [3.2, 1.0, 1.0] if clob else [3.2, 1.0]
    # A bare Figure (no pyplot) draws off-screen and holds no global state.
    fig = Figure(figsize=(11, 1.4 * sum(heights)), facecolor=SURFACE)
    axes = fig.subplots(
        len(heights), 1, sharex=True, gridspec_kw={"height_ratios": heights, "hspace": 0.12}
    )
    for ax in axes:
        style_axes(ax)
        for e in clue_events(model):  # faint guides, shared by every panel
            ax.axvline(
                e.step, color=FAKE if e.fake else NEWS, linewidth=0.6, alpha=0.25, gid="clue"
            )

    _price_panel(axes[0], model, df, steps)
    _volume_panel(axes[1], df, steps)
    if clob:
        _spread_panel(axes[2], df, steps)
    axes[-1].set_xlabel("step", color=INK)
    axes[-1].set_xlim(0, steps[-1] + 1)
    for ax in axes:
        ax.yaxis.set_label_coords(-0.055, 0.5)

    # One legend for every panel, in a row under the title.
    handles, labels = [], []
    for ax in axes:
        h, lab = ax.get_legend_handles_labels()
        handles += h
        labels += lab
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.07, 0.955),
        ncol=4,
        frameon=False,
        fontsize=8.5,
        labelcolor=INK,
        handletextpad=0.4,
        columnspacing=1.6,
    )

    fig.suptitle(title or run_label(model), x=0.07, y=0.99, ha="left", color=INK, fontsize=12)
    fig.subplots_adjust(left=0.09, right=0.88, top=0.87, bottom=0.07)
    return fig


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)


def _price_panel(ax, model: MarketModel, df, steps: np.ndarray) -> None:
    ax.axhline(50, color=MUTED, linewidth=0.8, linestyle=(0, (2, 3)), zorder=1)
    events = clue_events(model)
    if model.book is None:  # legacy AMM
        label = "price (LMSR)"
    else:
        bid = df.best_bid.astype(float).to_numpy()
        ask = df.best_ask.astype(float).to_numpy()
        ax.fill_between(
            steps,
            bid,
            ask,
            where=np.isfinite(bid) & np.isfinite(ask),
            color=BAND,
            alpha=0.7,
            linewidth=0,
            label="best bid to best ask",
            zorder=2,
        )
        for edge in (bid, ask):
            ax.plot(steps, edge, color=BAND_EDGE, linewidth=0.5, alpha=0.8, zorder=3)
        cap = model.config.spread_cap
        label = f"price (bid-ask midpoint if the spread is at most {cap}c, else last trade)"
    # P* on top: the per-step price is noisy and would hide the benchmark.
    ax.plot(steps, 100 * df.price, color=PRICE, linewidth=1.0, label=label, zorder=4)
    ax.plot(steps, 100 * df.p_star, color=P_STAR, linewidth=2, label="P* (benchmark)", zorder=5)

    # Clue markers sit in a strip above the price, so they never hide the lines.
    for fake in (False, True):
        for yes in (True, False):
            group = [e for e in events if e.fake == fake and e.points_yes == yes]
            if not group:
                continue
            kind = "fake" if fake else "news"
            ax.scatter(
                [e.step for e in group],
                [104 if yes else 101] * len(group),
                marker="^" if yes else "v",
                s=36,
                color=FAKE if fake else NEWS,
                edgecolors=SURFACE,
                linewidths=0.8,
                label=f"{kind} clue, points {'YES' if yes else 'NO'}",
                zorder=6,
                clip_on=False,
            )
    ax.set_ylim(0, 106)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("cents", color=INK)

    final = 100 * model.price
    p_star = 100 * model.p_star
    outcome = "YES" if model.world.y else "NO"
    # No final price when one side of the book is empty and nothing ever traded.
    price_text = "n/a" if np.isnan(final) else f"{final:.1f}c"
    ax.annotate(
        f"resolved {outcome}\nfinal price {price_text}\nfinal P* {p_star:.1f}c",
        xy=(steps[-1], p_star if np.isnan(final) else final),
        xytext=(10, 0),
        textcoords="offset points",
        va="center",
        color=INK,
        fontsize=9,
        annotation_clip=False,
    )


def _volume_panel(ax, df, steps: np.ndarray) -> None:
    volume = df.volume.to_numpy()
    n_bins = int(np.ceil(len(volume) / VOLUME_BIN))
    totals = np.add.reduceat(volume, np.arange(0, len(volume), VOLUME_BIN))
    starts = 1 + VOLUME_BIN * np.arange(n_bins)
    ax.bar(starts, totals, width=VOLUME_BIN * 0.85, align="edge", color=PRICE, linewidth=0)
    # Contract counts: start at 0 and use whole numbers, even when nothing traded.
    ax.set_ylim(0, max(1, int(totals.max())) * 1.05)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.set_ylabel(f"volume\n(per {VOLUME_BIN} steps)", color=INK)


def _spread_panel(ax, df, steps: np.ndarray) -> None:
    spread = df.spread.astype(float).to_numpy()
    ax.plot(steps, spread, color=PRICE, linewidth=0.8, drawstyle="steps-post", gid="spread")
    empty = steps[df.empty_book.to_numpy(dtype=bool)]
    if len(empty):
        ax.scatter(
            empty,
            np.zeros(len(empty)),
            marker="|",
            s=40,
            color=INK,
            label="one side of the book empty",
            clip_on=False,
            zorder=3,
        )
    top = np.nanmax(spread) if np.isfinite(spread).any() else 1.0
    ax.set_ylim(0, max(2.0, 1.1 * top))
    ax.set_ylabel("spread (c)", color=INK)
