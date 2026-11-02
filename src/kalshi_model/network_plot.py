"""A chart of one finished run's social network, with each agent's P&L.

Left: the network the run generated, laid out by a spring layout seeded by the run's
seed (a grid when there are no links). Each node is one agent (trader), each line a link
news and fakes can spread along. Node color is the agent's realized P&L in dollars: red
loss, gray zero, blue gain, saturating at the 95th percentile of |P&L| in large runs so a
few big winners or losers do not wash out the rest. Node area grows with the private
clues the agent was dealt. Small networks label each node with its agent index.

Right: each agent's P&L against its degree (number of links) and against its private
clues, with the mean per value (or per bin when values are many). A panel whose x never
varies is left out; if both are, a bar chart shows each agent's P&L instead.

The subtitle sums the money (traders, market maker, fees, manipulator) and, with
disinformation, how many agents heard a fake clue.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from kalshi_model.lmsr import subsidy_cents
from kalshi_model.model import MarketModel
from kalshi_model.plot import INK, MUTED, NEWS, SURFACE, run_label, style_axes

# Diverging pair (dataviz skill reference palette): red and blue poles, gray midpoint.
LOSS = "#e34948"
NEUTRAL = "#f0efec"
GAIN = "#2a78d6"
PNL_CMAP = LinearSegmentedColormap.from_list("pnl", [LOSS, NEUTRAL, GAIN])
LINK = "#b9b8b3"

SMALL = 30
"""Networks with at most this many agents label their nodes and color to the full range."""
MAX_MEAN_POINTS = 12
"""Above this many distinct x values, means are taken over quantile bins."""


def agent_graph(model: MarketModel) -> nx.Graph:
    """The run's network with nodes relabeled as agent indices 0..N-1.

    `model.neighbors[a]` lists agent a's neighbors, so this is the generated graph with
    each node renamed to the agent placed on it."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(model.traders)))
    graph.add_edges_from((a, b) for a, nbrs in enumerate(model.neighbors) for b in nbrs)
    return graph


def private_clues(model: MarketModel) -> np.ndarray:
    """Private clues each agent was dealt (its requested count, capped at K)."""
    return np.minimum(model.traits.n_clues, model.world.n_private).astype(int)


def heard_fake(model: MarketModel) -> np.ndarray:
    """Whether each agent learned at least one fake clue during the run."""
    reached = np.zeros(len(model.traders), dtype=bool)
    for clue_id in model.fake_ids:
        reached[list(model.learned_step.get(clue_id, {}))] = True
    return reached


def money_summary(model: MarketModel) -> dict[str, float]:
    """Where the money went, in dollars. The parts sum to zero, except that under the
    legacy AMM the market maker's P&L is net of its subsidy."""
    out = {"traders": sum(t.pnl for t in model.traders) / 100}
    if model.lmsr is not None:
        out["market maker"] = (model.lmsr.cash - subsidy_cents(model.config.lmsr_b)) / 100
    else:
        if model.config.mm_size:
            out["market maker"] = (model.market_maker.account.cash - model.config.mm_cash) / 100
        out["fees"] = model.book.fees_collected / 100
    if model.manipulator is not None:
        out["manipulator"] = model.manipulator.pnl / 100
    return out


def network_chart(model: MarketModel, title: str | None = None) -> Figure:
    """Draw a finished run's network and P&L. Raises ValueError if it has not settled."""
    if not model.settled:
        raise ValueError("run the model to settlement before plotting it")
    graph = agent_graph(model)
    pnl = np.array([t.pnl for t in model.traders]) / 100
    degree = np.array([len(n) for n in model.neighbors])
    clues = private_clues(model)

    panels = [
        (x, label, gid)
        for x, label, gid in (
            (degree, "degree (links)", "degree"),
            (clues, "private clues dealt", "clues"),
        )
        if len(np.unique(x)) > 1
    ]
    fig = Figure(figsize=(13, 7.2), facecolor=SURFACE)
    grid = fig.add_gridspec(max(1, len(panels)), 2, width_ratios=[1.9, 1], wspace=0.18, hspace=0.45)
    _network_panel(fig, fig.add_subplot(grid[:, 0]), model, graph, pnl, clues)
    if panels:
        first = None
        for row, (x, label, gid) in enumerate(panels):
            ax = fig.add_subplot(grid[row, 1], sharey=first)
            _pnl_scatter(ax, x, pnl, label, gid)
            first = first or ax
        first.legend(
            loc="lower left",
            bbox_to_anchor=(0, 1.0),
            ncol=2,
            frameon=False,
            fontsize=8.5,
            labelcolor=INK,
        )
    else:
        _pnl_bars(fig.add_subplot(grid[0, 1]), pnl)

    fig.suptitle(title or run_label(model), x=0.04, y=0.985, ha="left", color=INK, fontsize=12)
    fig.text(0.04, 0.935, _subtitle(model, graph), ha="left", color=MUTED, fontsize=9)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.82, bottom=0.08)
    return fig


def _subtitle(model: MarketModel, graph: nx.Graph) -> str:
    n, links = graph.number_of_nodes(), graph.number_of_edges()
    if links:
        parts = [f"{n} agents, {links} links, mean degree {2 * links / n:.1f}"]
    else:
        parts = [f"{n} agents, no network"]
    parts.append("P&L: " + ", ".join(f"{k} {_dollars(v)}" for k, v in money_summary(model).items()))
    clues = private_clues(model)
    if clues.min() == clues.max():  # every node is the same size, so no size key
        parts.append(f"every agent dealt {_clues(int(clues[0]))}")
    if model.fake_ids:
        parts.append(f"{heard_fake(model).sum()} of {n} agents heard a fake clue")
    return "   |   ".join(parts)


def _dollars(value: float) -> str:
    sign = "+" if value > 0 else "-" if value < 0 else ""
    return f"{sign}\\${abs(value):,.2f}"  # escaped: a bare $ starts matplotlib math text


def _layout(graph: nx.Graph, seed: int) -> dict[int, np.ndarray]:
    n = graph.number_of_nodes()
    if graph.number_of_edges() == 0:  # no links: a plain grid in agent order
        cols = math.ceil(math.sqrt(n))
        return {i: np.array([i % cols, -(i // cols)], dtype=float) for i in range(n)}
    return nx.spring_layout(graph, seed=seed % 2**32, iterations=200)


def color_limit(pnl: np.ndarray) -> float:
    """Half the color scale's range in dollars: the largest |P&L| in small runs, its
    95th percentile in large ones."""
    size = np.abs(pnl)
    limit = size.max() if len(pnl) <= SMALL else np.percentile(size, 95)
    return float(limit) if limit > 0 else max(float(size.max()), 0.01)


def _network_panel(fig, ax, model, graph, pnl, clues) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_axis_off()
    pos = _layout(graph, model.seed)
    xy = np.array([pos[i] for i in range(graph.number_of_nodes())])
    if graph.number_of_edges():
        segments = [(pos[a], pos[b]) for a, b in graph.edges]
        ax.add_collection(
            LineCollection(segments, colors=LINK, linewidths=0.6, alpha=0.6, zorder=1, gid="links")
        )

    n = len(xy)
    small = n <= SMALL
    base = 260 if small else max(18, 2400 / n)
    top = max(1, int(clues.max()))
    varies = clues.min() < clues.max()
    sizes = base * (1 + 3 * clues / top) if varies else np.full(n, 2.0 * base)
    limit = color_limit(pnl)
    nodes = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=sizes,
        c=pnl,
        cmap=PNL_CMAP,
        norm=Normalize(-limit, limit),
        edgecolors=MUTED,
        linewidths=0.5,
        zorder=2,
        gid="agents",
    )
    if small:
        for i, (x, y) in enumerate(xy):
            ax.annotate(str(i), (x, y), ha="center", va="center", fontsize=7, color=INK, zorder=3)
    ax.margins(0.08)
    ax.set_aspect("equal", adjustable="datalim")

    clipped = np.abs(pnl).max() > limit
    bar = fig.colorbar(
        nodes,
        ax=ax,
        orientation="horizontal",
        fraction=0.035,
        pad=0.02,
        aspect=40,
        shrink=0.6,
        extend="both" if clipped else "neither",
    )
    label = "agent P&L ($)" + (", 95th percentile and beyond at the ends" if clipped else "")
    bar.set_label(label.replace("$", "\\$"), color=INK, fontsize=9)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)

    if varies:  # node-size key, in a row above the panel so it never covers a node
        values = sorted({0, top // 2, top})
        areas = [base * (1 + 3 * v / top) for v in values]
        handles = [_size_key(_clues(v), a) for v, a in zip(values, areas, strict=True)]
        ax.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0, 1.0),
            ncol=len(handles),
            frameon=False,
            fontsize=8.5,
            labelcolor=INK,
            handletextpad=0.4 + math.sqrt(areas[-1]) / 34,
            columnspacing=2.0,
            borderaxespad=0.2,
        )


def _clues(count: int) -> str:
    return f"{count} private clue{'' if count == 1 else 's'}"


def _size_key(label: str, area: float) -> Line2D:
    return Line2D(
        [],
        [],
        linestyle="",
        marker="o",
        markersize=math.sqrt(area),
        markerfacecolor=NEUTRAL,
        markeredgecolor=MUTED,
        markeredgewidth=0.5,
        label=label,
    )


def mean_points(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean y at each distinct x, or over up to 8 quantile bins of x (at the bin's mean
    x) when there are more than `MAX_MEAN_POINTS` distinct values."""
    values = np.unique(x)
    if len(values) <= MAX_MEAN_POINTS:
        return values.astype(float), np.array([y[x == v].mean() for v in values])
    edges = np.unique(np.quantile(x, np.linspace(0, 1, 9)))
    bins = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    used = np.unique(bins)
    return (
        np.array([x[bins == b].mean() for b in used]),
        np.array([y[bins == b].mean() for b in used]),
    )


def _pnl_scatter(ax, x: np.ndarray, pnl: np.ndarray, xlabel: str, gid: str) -> None:
    style_axes(ax)
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
    # Fixed jitter so agents with the same integer x do not stack on one point.
    jitter = np.random.default_rng(0).uniform(-0.2, 0.2, len(x))
    ax.scatter(
        x + jitter,
        pnl,
        s=16,
        color=NEWS,
        alpha=0.5,
        linewidths=0,
        zorder=2,
        gid=gid,
        label="one agent",
    )
    mx, my = mean_points(x, pnl)
    if len(mx) < len(x):  # with one agent per x value, the mean is just the dots
        binned = len(np.unique(x)) > MAX_MEAN_POINTS
        ax.plot(
            mx,
            my,
            color=INK,
            linewidth=1.5,
            marker="o",
            markersize=5,
            markeredgecolor=SURFACE,
            zorder=3,
            gid=f"{gid}_mean",
            label="mean per bin" if binned else "mean",
        )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
    ax.set_ylabel("P&L (\\$)", color=INK, fontsize=9)


def _pnl_bars(ax, pnl: np.ndarray) -> None:
    """Each agent's P&L: by agent index in small runs, else ranked."""
    style_axes(ax)
    n = len(pnl)
    order = np.arange(n) if n <= SMALL else np.argsort(pnl)
    values = pnl[order]
    limit = color_limit(pnl)
    colors = PNL_CMAP(Normalize(-limit, limit)(values))
    ax.bar(np.arange(n), values, color=colors, width=0.8, linewidth=0, gid="pnl_bars")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    if n <= SMALL:
        ax.set_xticks(np.arange(n))
        ax.set_xlabel("agent", color=INK, fontsize=9)
    else:
        ax.set_xticks([])
        ax.set_xlabel("agents, ranked by P&L", color=INK, fontsize=9)
    ax.set_ylabel("P&L (\\$)", color=INK, fontsize=9)
