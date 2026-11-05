"""Figures for E1 to E3, drawn from the saved results (PLAN.md Section 4).

Each figure reads only `results/<experiment>.csv` (through `experiment.load`), so it can
be redrawn without rerunning anything. Every mark is a mean over seeds with its 95%
interval (`experiment.mean_ci`): a band for curves, a bar through each dot otherwise.
`experiments/figures.py` writes them all to `results/figures/`.

- `e1_news_response.png` (E1.1): the price's move toward each news clue, by lag.
- `e1_rules.png` (E1.2-E1.5): volume, accuracy, maker vs taker return, fees paid.
- `e2_distortion.png` (E2.5): distortion against the number of fakes, total and per fake.
- `e2_degree.png` (E2.3): when the best and worst connected quarters learn news, and
  what they earn.
- `e3_distortion.png` (E3.1, E3.2): price minus the honest twin's after the layer starts.
- `e3_premium.png` (E3.1, E3.2): the manipulator's premium and agents' gain from it.
- `e3_liquidity.png` (E3.3): distortion against book depth before the layer starts.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from kalshi_model.config import Config
from kalshi_model.experiment import RESULTS_DIR, load, mean_ci
from kalshi_model.metrics import DISTORTION_LAGS, RESPONSE_LAGS, distortion_column, response_column
from kalshi_model.plot import GRID, INK, MUTED, SURFACE, style_axes

FIGURES_DIR = RESULTS_DIR / "figures"

# Reference palette (dataviz skill), categorical slots 1 to 3. Aqua sits below 3:1 on
# the light surface, so every aqua mark is labeled directly.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

E1_MARKETS = {"clob": ("CLOB, no fees", BLUE), "kalshi": ("Kalshi", ORANGE)}
E3_STRATEGIES = {"pump": ("Pump and dump", BLUE), "spoof": ("Spoofing", ORANGE)}

Results = dict[str, dict[int, dict]]


def _figure(width: float, height: float, title: str, subtitle: str) -> Figure:
    """A figure with a title at the top and a wrapped note under the plots."""
    fig = Figure(figsize=(width, height), facecolor=SURFACE, layout="constrained")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    note = textwrap.fill(subtitle, width=int(width * 15))
    lines = note.count("\n") + 1
    fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=8.5, color=MUTED)
    bottom = (0.16 * lines + 0.08) / height  # inches of note, as a share of the height
    fig.get_layout_engine().set(rect=(0, bottom, 1, 1 - 0.42 / height))
    return fig


def _rows(results: Results, label: str) -> list[dict]:
    return [row for _, row in sorted(results[label].items())]


def _ci(rows: list[dict], metric: str | Callable[[dict], float]) -> tuple[float, float]:
    get = metric if callable(metric) else (lambda row: row[metric])
    return mean_ci([get(row) for row in rows])


def _curve_band(ax, x, curves: np.ndarray, color: str, label: str) -> None:
    """Mean curve over seeds (rows of `curves`) with a 95% band."""
    mean = curves.mean(axis=0)
    half = 1.96 * curves.std(axis=0, ddof=1) / np.sqrt(len(curves))
    ax.fill_between(x, mean - half, mean + half, color=color, alpha=0.18, linewidth=0)
    ax.plot(x, mean, color=color, linewidth=2, label=label)


def _dots(ax, labels: list[str], estimates: list[tuple[float, float]], colors: list[str]):
    """One dot per group with its 95% interval, each labeled with its value."""
    for i, ((mean, half), color) in enumerate(zip(estimates, colors, strict=True)):
        ax.errorbar(i, mean, yerr=half, fmt="o", color=color, markersize=8, capsize=0, lw=2)
        ax.annotate(
            f"{mean:,.2f}" if abs(mean) < 10 else f"{mean:,.0f}",
            (i, mean),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=INK,
        )
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlim(-0.6, len(labels) - 0.4)


def _zero_line(ax) -> None:
    ax.axhline(0, color=MUTED, linewidth=1)


# E1


def e1_news_response(results: Results) -> Figure:
    fig = _figure(
        7.5,
        4.2,
        "E1.1: Kalshi's fee slows how fast news reaches the price",
        "Mean move of the price toward each news clue, in cents, by steps since its "
        "release (95% band over 100 seeds). Dashed: the mean move of P*.",
    )
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
        fontsize=8.5,
        color=MUTED,
    )
    ax.set_xlabel("steps after the news clue", color=INK)
    ax.set_ylabel("price move toward the clue (c)", color=INK)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    return fig


def e1_rules(results: Results) -> Figure:
    fig = _figure(
        10,
        6.2,
        "E1.2-E1.5: what Kalshi's fee changes",
        "Means over 100 seeds with 95% intervals. Returns and P&L are valued at the final "
        "P*. Fees per contract are Kalshi only; the CLOB charges none.",
    )
    axes = fig.subplots(2, 2)
    names = [E1_MARKETS[k][0] for k in E1_MARKETS]
    colors = [E1_MARKETS[k][1] for k in E1_MARKETS]
    panels = [
        (axes[0, 0], "E1.2 Volume (thousand contracts)", lambda r: r["volume"] / 1000),
        (axes[0, 1], "E1.2 Mean |price - P*| (c)", lambda r: r["error"]),
    ]
    for ax, title, metric in panels:
        style_axes(ax)
        _dots(ax, names, [_ci(_rows(results, k), metric) for k in E1_MARKETS], colors)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_ylim(bottom=0)

    ax = axes[1, 0]
    style_axes(ax)
    kalshi = _rows(results, "kalshi")
    _dots(
        ax,
        ["Maker fills", "Taker fills"],
        [_ci(kalshi, "maker_return_at_p_star"), _ci(kalshi, "taker_return_at_p_star")],
        [BLUE, ORANGE],
    )
    _zero_line(ax)
    ax.set_title("E1.4 Kalshi: agents' return per contract (c)", loc="left", fontsize=10, color=INK)

    ax = axes[1, 1]
    style_axes(ax)
    _dots(
        ax,
        ["No clues", "Best-informed tenth"],
        [_ci(kalshi, "fee_per_contract_none"), _ci(kalshi, "fee_per_contract_top")],
        [BLUE, ORANGE],
    )
    ax.set_ylim(bottom=0)
    ax.set_title("E1.5 Kalshi: fee per contract traded (c)", loc="left", fontsize=10, color=INK)
    return fig


# E2


def _ratio_labels(results: Results) -> list[tuple[float, str]]:
    labels = [(float(label.split()[-1]), label) for label in results]
    return sorted(labels)


def e2_distortion(results: Results) -> Figure:
    fig = _figure(
        10,
        4.2,
        "E2.5: distortion grows in proportion to the number of fakes",
        "Price gap from P* in the fakes' direction minus the same seed's gap without fakes, "
        "in cents (means over 100 seeds, 95% intervals).",
    )
    total, per_fake = fig.subplots(1, 2)
    ratios = _ratio_labels(results)
    clean = results[ratios[0][1]]
    fakes, whole, last, per = [], [], [], []
    for ratio, label in ratios[1:]:
        n = Config(disinfo_ratio=ratio).n_fakes
        rows = results[label]
        diff = {s: rows[s]["signed_gap"] - clean[s]["signed_gap"] for s in rows}
        late = {
            s: rows[s]["signed_gap_last_quarter"] - clean[s]["signed_gap_last_quarter"]
            for s in rows
        }
        fakes.append(n)
        whole.append(mean_ci(list(diff.values())))
        last.append(mean_ci(list(late.values())))
        per.append(mean_ci([v / n for v in diff.values()]))

    style_axes(total)
    for estimates, color, name in ((whole, BLUE, "whole run"), (last, ORANGE, "last quarter")):
        means = [m for m, _ in estimates]
        total.errorbar(
            fakes,
            means,
            yerr=[h for _, h in estimates],
            color=color,
            marker="o",
            markersize=7,
            lw=2,
            capsize=0,
            label=name,
        )
    _zero_line(total)
    total.set_xticks(fakes, [f"{n}\n({r:g})" for n, (r, _) in zip(fakes, ratios[1:], strict=True)])
    total.set_xlabel("fake clues (disinformation ratio)", color=INK)
    total.set_title("Distortion (c)", loc="left", fontsize=10, color=INK)
    total.legend(frameon=False, fontsize=9, loc="upper left")

    style_axes(per_fake)
    _dots(per_fake, [f"{n} fake" + ("s" if n > 1 else "") for n in fakes], per, [BLUE] * len(fakes))
    _zero_line(per_fake)
    per_fake.set_title("Distortion per fake (c), whole run", loc="left", fontsize=10, color=INK)
    return fig


def e2_degree(results: Results, ratio_label: str = "ratio 0.1") -> Figure:
    fig = _figure(
        10,
        4.2,
        "E2.3: well-connected agents hear news first",
        "Top and bottom quarters of agents by number of links (means over 100 seeds, 95% "
        "intervals). P&L is valued at P* in the runs without fakes.",
    )
    delay, pnl = fig.subplots(1, 2)
    rows = _rows(results, ratio_label)
    style_axes(delay)
    _dots(
        delay,
        ["Top quarter", "Bottom quarter"],
        [_ci(rows, "learn_delay_top_degree"), _ci(rows, "learn_delay_bottom_degree")],
        [BLUE, ORANGE],
    )
    delay.set_ylim(bottom=0)
    delay.set_title(
        f"Steps to learn each news clue ({ratio_label})", loc="left", fontsize=10, color=INK
    )

    clean = _rows(results, _ratio_labels(results)[0][1])
    style_axes(pnl)
    _dots(
        pnl,
        ["Top quarter", "Bottom quarter"],
        [_ci(clean, "pnl_top_degree_at_p_star"), _ci(clean, "pnl_bottom_degree_at_p_star")],
        [BLUE, ORANGE],
    )
    _zero_line(pnl)
    pnl.set_title("P&L per agent ($), no fakes", loc="left", fontsize=10, color=INK)
    return fig


# E3


def e3_distortion(results: Results) -> Figure:
    fig = _figure(
        7.5,
        4.2,
        "E3.1-E3.2: a pump moves the price more than a spoof, but not for long",
        "Price minus the honest twin's price on the same seed, in cents, by steps after the "
        "layer starts at step 1000 (95% band over 100 seeds).",
    )
    ax = fig.add_subplot()
    style_axes(ax)
    lags = list(DISTORTION_LAGS)
    for key, (name, color) in E3_STRATEGIES.items():
        rows = _rows(results, key)
        curves = np.array([[r[distortion_column(lag)] for lag in lags] for r in rows])
        _curve_band(ax, lags, curves, color, name)
    _zero_line(ax)
    spoof_end = Config().spoof_steps
    ax.axvline(spoof_end, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate(
        f"spoofing stops ({spoof_end} steps)",
        (spoof_end, ax.get_ylim()[1]),
        xytext=(4, -12),
        textcoords="offset points",
        fontsize=8.5,
        color=MUTED,
    )
    ax.set_xlabel("steps after the manipulation layer starts", color=INK)
    ax.set_ylabel("distortion (c)", color=INK)
    ax.legend(frameon=False, fontsize=9)
    return fig


def e3_premium(results: Results) -> Figure:
    fig = _figure(
        10,
        4.2,
        "E3.1-E3.2: manipulation does not pay",
        "Means over 100 seeds with 95% intervals, valued at the final P*. Premium: the "
        "manipulator's P&L minus its honest twin's on the same seed.",
    )
    premium, taken = fig.subplots(1, 2)
    names = [E3_STRATEGIES[k][0] for k in E3_STRATEGIES]
    colors = [E3_STRATEGIES[k][1] for k in E3_STRATEGIES]
    for ax, metric, title in (
        (premium, lambda r: r["premium"], "Manipulation premium ($ per run)"),
        (taken, lambda r: -r["taken_from_agents"], "Agents' P&L, minus the twin run's ($)"),
    ):
        style_axes(ax)
        _dots(ax, names, [_ci(_rows(results, k), metric) for k in E3_STRATEGIES], colors)
        _zero_line(ax)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    return fig


def e3_liquidity(results: Results) -> Figure:
    fig = _figure(
        10,
        4.2,
        "E3.3: deeper books do not distort less",
        "One dot per seed: contracts resting within 2c of the best prices just before step "
        "1000, against the mean distortion while the layer acts.",
    )
    axes = fig.subplots(1, 2, sharey=False)
    for ax, (key, (name, color)) in zip(axes, E3_STRATEGIES.items(), strict=True):
        style_axes(ax)
        rows = _rows(results, key)
        depth = np.array([r["depth_before"] for r in rows]) / 1000
        dist = np.array([r["distortion"] for r in rows])
        ax.scatter(depth, dist, s=28, color=color, alpha=0.75, edgecolor=SURFACE, linewidth=1)
        _zero_line(ax)
        rho = _spearman(depth, dist)
        ax.set_title(f"{name}: Spearman {rho:+.2f}", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("depth before the layer (thousand contracts)", color=INK)
        ax.set_ylabel("distortion while acting (c)", color=INK)
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
