"""Figures for E1 to E3, drawn from the saved results.

Each figure reads only `results/<experiment>.csv` (through `experiment.load`), so it can
be redrawn without rerunning anything. Per-run metrics are drawn as violins, one per
configuration or group, showing their spread over seeds, with a dot at the mean and a
bar for its 95% interval (`experiment.mean_ci`). Curves over time are a mean with a 95%
band. Where configurations share seeds, the panel also states the paired difference.
`experiments/figures.py` writes them all to `results/figures/`.

- `e1_news_response.png`: the price's move toward each news clue, by lag.
- `e1_rules.png`: what the fee changes: volume, news half-life, pricing error, and
  the best-informed tenth's profit.
- `e2_distortion.png`: distortion against the number of fakes, over the whole run and
  its last quarter.
- `e2_degree.png`: when the best and worst connected quarters learn news, and what
  they earn.
- `e3_distortion.png`: price minus the honest twin's after the manipulation starts.
- `e3_premium.png`: distortion, the manipulator's premium, and agents' gain from it.
- `e3_liquidity.png`: distortion against book depth before the manipulation starts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import StrMethodFormatter

from kalshi_model.config import Config
from kalshi_model.experiment import RESULTS_DIR, load, mean_ci
from kalshi_model.metrics import DISTORTION_LAGS, RESPONSE_LAGS, distortion_column, response_column
from kalshi_model.plot import GRID, INK, MUTED, SURFACE, style_axes

FIGURES_DIR = RESULTS_DIR / "figures"

# Reference palette (dataviz skill), categorical slots 1 and 2.
BLUE = "#2a78d6"
ORANGE = "#eb6834"

E1_MARKETS = {"clob": ("CLOB, no fees", BLUE), "kalshi": ("Kalshi", ORANGE)}
E3_STRATEGIES = {"pump": ("Pump and dump", BLUE), "spoof": ("Spoofing", ORANGE)}

Results = dict[str, dict[int, dict]]
Metric = str | Callable[[dict], float]


def _figure(width: float, height: float, title: str) -> Figure:
    """A figure with its title at the top left."""
    fig = Figure(figsize=(width, height), facecolor=SURFACE, layout="constrained")
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK)
    return fig


def _rows(results: Results, label: str) -> list[dict]:
    return [row for _, row in sorted(results[label].items())]


def _values(rows: list[dict], metric: Metric) -> np.ndarray:
    """The metric on every run, finite values only."""
    get = metric if callable(metric) else (lambda row: row[metric])
    values = np.array([get(row) for row in rows], dtype=float)
    return values[np.isfinite(values)]


def _paired(a: list[dict], b: list[dict], metric: Metric) -> tuple[float, float]:
    """Mean and 95% half-width of b - a, run by run (the rows share seeds, in order)."""
    get = metric if callable(metric) else (lambda row: row[metric])
    return mean_ci([get(rb) - get(ra) for ra, rb in zip(a, b, strict=True)])


def _fmt(value: float) -> str:
    if abs(value) < 10:
        return f"{value:,.2f}"
    return f"{value:,.1f}" if abs(value) < 100 else f"{value:,.0f}"


def _violins(
    ax,
    positions: Sequence[float],
    samples: Sequence[np.ndarray],
    colors: Sequence[str],
    *,
    width: float = 0.7,
    label_means: bool = True,
) -> list[float]:
    """One violin per sample with a dot at its mean and a bar for the 95% interval.

    The violins are drawn before the means, so the mean dots are the axis's first line.
    Returns the means.
    """
    parts = ax.violinplot(list(samples), positions=list(positions), widths=width, showextrema=False)
    for body, color in zip(parts["bodies"], colors, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.3)
        body.set_linewidth(1)
    estimates = [mean_ci(s) for s in samples]
    means = [m for m, _ in estimates]
    ax.plot(positions, means, linestyle="none", marker="o", markersize=8, color=INK, zorder=3)
    ax.errorbar(
        positions, means, yerr=[h for _, h in estimates], fmt="none", ecolor=INK, lw=2, zorder=3
    )
    if max(abs(v) for sample in samples for v in sample) >= 1000:
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    if label_means:
        for x, mean in zip(positions, means, strict=True):
            ax.annotate(
                _fmt(mean),
                (x, mean),
                xytext=(9, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color=INK,
            )
    return means


def _note(ax, text: str) -> None:
    """A short note in the panel's top right corner (a paired difference)."""
    ax.text(0.98, 0.97, text, transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK)
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.15 * (high - low))  # room for the note above the data


def _panel_title(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)


def _zero_line(ax) -> None:
    ax.axhline(0, color=MUTED, linewidth=1, zorder=1)


def _curve_band(ax, x, curves: np.ndarray, color: str, label: str) -> None:
    """Mean curve over seeds (rows of `curves`) with a 95% band."""
    mean = np.nanmean(curves, axis=0)
    half = 1.96 * np.nanstd(curves, axis=0, ddof=1) / np.sqrt(np.isfinite(curves).sum(axis=0))
    ax.fill_between(x, mean - half, mean + half, color=color, alpha=0.18, linewidth=0)
    ax.plot(x, mean, color=color, linewidth=2, label=label)


# E1


def e1_news_response(results: Results) -> Figure:
    fig = _figure(7.5, 4.2, "Speed of News Pricing")
    ax = fig.add_subplot()
    style_axes(ax)
    lags = np.array(RESPONSE_LAGS)
    for key, (name, color) in E1_MARKETS.items():
        rows = _rows(results, key)
        curves = np.array([[r[response_column(lag)] for lag in lags] for r in rows])
        _curve_band(ax, lags, curves, color, name)
    jump = np.mean([r["p_star_jump"] for key in E1_MARKETS for r in _rows(results, key)])
    ax.axhline(jump, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate(
        "P* move",
        (lags[-1], jump),
        xytext=(-4, 4),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=MUTED,
    )
    ax.set_xlabel("Steps after a news clue", color=INK)
    ax.set_ylabel("Price move toward the clue (c)", color=INK)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    return fig


E1_PANELS: tuple[tuple[str, Metric, int], ...] = (
    ("Volume (thousand contracts)", lambda r: r["volume"] / 1000, 1),
    ("News half-life (steps)", "news_half_life", 1),
    ("Pricing error |price - P*| (c)", "error", 2),
    ("Best-informed tenth's P&L ($ per agent)", "pnl_top_at_p_star", 1),
)
"""E1 panels: liquidity, speed, accuracy, and the reward for information, with the
decimals of each panel's paired difference."""


def e1_rules(results: Results) -> Figure:
    fig = _figure(10, 7, "Impact of Kalshi Fees")
    names = [E1_MARKETS[k][0] for k in E1_MARKETS]
    colors = [E1_MARKETS[k][1] for k in E1_MARKETS]
    clob, kalshi = _rows(results, "clob"), _rows(results, "kalshi")
    for ax, (title, metric, digits) in zip(fig.subplots(2, 2).flat, E1_PANELS, strict=True):
        style_axes(ax)
        _violins(ax, [0, 1], [_values(clob, metric), _values(kalshi, metric)], colors)
        ax.set_xticks([0, 1], names)
        ax.set_xlim(-0.6, 1.6)
        _panel_title(ax, title)
        mean, half = _paired(clob, kalshi, metric)
        _note(ax, f"Kalshi - CLOB: {mean:+,.{digits}f} ± {half:,.{digits}f}")
    return fig


# E2


def _ratio_labels(results: Results) -> list[tuple[float, str]]:
    labels = [(float(label.split()[-1]), label) for label in results]
    return sorted(labels)


def e2_distortion(results: Results) -> Figure:
    """Per seed, the signed gap toward the fakes minus the same seed's gap without them."""
    fig = _figure(10, 4.4, "Distortion from Disinformation")
    ratios = _ratio_labels(results)
    clean = results[ratios[0][1]]
    fakes = [Config(disinfo_ratio=ratio).n_fakes for ratio, _ in ratios[1:]]
    n_real = Config().n_clues_total - Config().n_private
    axes = fig.subplots(1, 2, sharey=True)
    for ax, (metric, title) in zip(
        axes,
        (("signed_gap", "Whole run"), ("signed_gap_last_quarter", "Last quarter of the run")),
        strict=True,
    ):
        style_axes(ax)
        samples = []
        for _, label in ratios[1:]:
            rows = results[label]
            samples.append(np.array([rows[s][metric] - clean[s][metric] for s in sorted(rows)]))
        means = _violins(ax, fakes, samples, [BLUE] * len(fakes), width=0.8, label_means=False)
        # Least-squares line through the origin: the distortion each fake adds.
        slope = float(np.dot(fakes, means) / np.dot(fakes, fakes))
        x = np.array([0, max(fakes)])
        ax.plot(x, slope * x, color=MUTED, linewidth=1, linestyle="--", zorder=2)
        middle = (fakes[-2] + fakes[-1]) / 2  # between the violins, clear of both
        ax.annotate(
            f"{slope:.2f}c per fake",
            (middle, slope * middle),
            xytext=(-6, 6),
            textcoords="offset points",
            ha="right",
            fontsize=9,
            color=INK,
        )
        _zero_line(ax)
        ax.set_xticks(fakes, [str(n) for n in fakes])
        ax.set_xlabel(f"Fake clues (against {n_real} real news clues)", color=INK)
        _panel_title(ax, title)
    axes[0].set_ylabel("Distortion toward the fakes (c)", color=INK)
    return fig


def e2_degree(results: Results) -> Figure:
    """Top vs bottom degree quarter in the runs without fakes. Fakes spread on their own
    random numbers, so real news spreads the same way at every ratio."""
    fig = _figure(10, 4.4, "Network Position and News")
    rows = _rows(results, _ratio_labels(results)[0][1])
    names = ["Best-connected quarter", "Least-connected quarter"]
    for ax, (top, bottom, title, digits) in zip(
        fig.subplots(1, 2),
        (
            (
                "learn_delay_top_degree",
                "learn_delay_bottom_degree",
                "Steps to learn a news clue",
                2,
            ),
            ("pnl_top_degree_at_p_star", "pnl_bottom_degree_at_p_star", "P&L ($ per agent)", 1),
        ),
        strict=True,
    ):
        style_axes(ax)
        _violins(ax, [0, 1], [_values(rows, top), _values(rows, bottom)], [BLUE, ORANGE])
        ax.set_xticks([0, 1], names)
        ax.set_xlim(-0.6, 1.6)
        _panel_title(ax, title)
        mean, half = mean_ci([r[top] - r[bottom] for r in rows])
        _note(ax, f"Best - least: {mean:+,.{digits}f} ± {half:,.{digits}f}")
    return fig


# E3


def e3_distortion(results: Results) -> Figure:
    fig = _figure(7.5, 4.2, "Price Distortion from Manipulation")
    ax = fig.add_subplot()
    style_axes(ax)
    lags = list(DISTORTION_LAGS)
    for key, (name, color) in E3_STRATEGIES.items():
        rows = _rows(results, key)
        curves = np.array([[r[distortion_column(lag)] for lag in lags] for r in rows])
        _curve_band(ax, lags, curves, color, name)
    _zero_line(ax)
    # Linear up to 10 steps, then log: the pump acts within the first hundred steps.
    ax.set_xscale("symlog", linthresh=10, linscale=0.6)
    ax.set_xticks([0, 10, 100, 1000], ["0", "10", "100", "1,000"])
    ax.set_xlim(0, lags[-1])
    spoof_end = Config().spoof_steps
    ax.axvline(spoof_end, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate(
        f"Spoofing stops ({spoof_end} steps)",
        (spoof_end, ax.get_ylim()[1]),
        xytext=(-4, -12),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=MUTED,
    )
    ax.set_xlabel("Steps after the manipulation starts", color=INK)
    ax.set_ylabel("Price minus the honest twin's (c)", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    return fig


def e3_premium(results: Results) -> Figure:
    fig = _figure(11, 4.4, "Does Manipulation Pay?")
    names = [E3_STRATEGIES[k][0] for k in E3_STRATEGIES]
    colors = [E3_STRATEGIES[k][1] for k in E3_STRATEGIES]
    for ax, (metric, title) in zip(
        fig.subplots(1, 3),
        (
            ("distortion", "Distortion while manipulating (c)"),
            ("premium", "Manipulator's premium ($)"),
            (lambda r: -r["taken_from_agents"], "Agents' gain ($)"),
        ),
        strict=True,
    ):
        style_axes(ax)
        _violins(ax, [0, 1], [_values(_rows(results, k), metric) for k in E3_STRATEGIES], colors)
        _zero_line(ax)
        ax.set_xticks([0, 1], names)
        ax.set_xlim(-0.6, 1.6)
        _panel_title(ax, title)
    return fig


def e3_liquidity(results: Results) -> Figure:
    fig = _figure(10, 4.4, "Book Depth and Distortion")
    axes = fig.subplots(1, 2, sharey=False)
    for ax, (key, (name, color)) in zip(axes, E3_STRATEGIES.items(), strict=True):
        style_axes(ax)
        rows = _rows(results, key)
        depth = np.array([r["depth_before"] for r in rows]) / 1000
        dist = np.array([r["distortion"] for r in rows])
        ax.scatter(depth, dist, s=18, color=color, alpha=0.6, edgecolor=SURFACE, linewidth=0.8)
        _zero_line(ax)
        rho = _spearman(depth, dist)
        _panel_title(ax, f"{name} (Spearman {rho:+.2f})")
        ax.set_xlabel("Depth before manipulation (thousand contracts)", color=INK)
        ax.set_ylabel("Distortion while manipulating (c)", color=INK)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    return fig


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x.argsort().argsort(), y.argsort().argsort())[0, 1])


FIGURES: dict[str, tuple[str, Callable[[Results], Figure]]] = {
    "e1_news_response": ("e1_rules", e1_news_response),
    "e1_rules": ("e1_rules", e1_rules),
    "e2_distortion": ("e2_information", e2_distortion),
    "e2_degree": ("e2_information", e2_degree),
    "e3_distortion": ("e3_manipulation", e3_distortion),
    "e3_premium": ("e3_manipulation", e3_premium),
    "e3_liquidity": ("e3_manipulation", e3_liquidity),
}
"""Figure name -> (results it reads, function that draws it)."""


def write_figures(
    results_dir: Path = RESULTS_DIR, out_dir: Path | None = None, dpi: int = 150
) -> list[Path]:
    """Draw every figure from the saved results; return the PNG paths written."""
    out_dir = out_dir or results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, Results] = {}
    written = []
    for name, (experiment, draw) in FIGURES.items():
        if experiment not in cache:
            cache[experiment] = load(experiment, results_dir)
        path = out_dir / f"{name}.png"
        draw(cache[experiment]).savefig(path, dpi=dpi, facecolor=SURFACE)
        written.append(path)
    return written
