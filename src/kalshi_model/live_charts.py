"""Plotly charts for the live GUI, built once per run and then updated in place.

Each chart has `build(run)`, which lays out a figure with empty traces on the run's full
step axis, and `update(run)`, which returns new data for every trace (in build order)
plus any layout changes. The GUI keeps one Plotly widget per chart and applies each
update to it, so the browser redraws only the data instead of a whole new image.

`applies(run)` says whether a chart has anything to show: order book charts need the
order book (not the legacy AMM), the manipulator charts need a manipulator (E3), and the
honest-twin chart needs a twin run, which the GUI runs alongside pump and dump or
spoofing on the same seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from kalshi_model.live import (
    AT_RELEASE,
    FROM_NEIGHBOR,
    UNKNOWN,
    latest_clue,
    series,
)
from kalshi_model.manipulator import LAYER_PHASES
from kalshi_model.model import MarketModel
from kalshi_model.plot import BAND, BAND_EDGE, FAKE, INK, MUTED, NEWS, P_STAR, PRICE, SURFACE

GRID = "#e4e3df"
TWIN = "#8a8985"
"""The honest twin's price: muted, so the manipulated price stands out."""
BUY_MARK, SELL_MARK = "#1f9e6e", "#e34948"
PHASE_COLORS = {"pump": "#1f9e6e", "hold": "#c9a227", "dump": "#e34948", "spoof": "#6250d6"}
SPREAD_COLORS = {UNKNOWN: "#d6d5d0", AT_RELEASE: INK, FROM_NEIGHBOR: PRICE}
"""Node colors: gray does not know the clue yet, black got it at release, blue heard it
(purple, the fake color, when the clue is a fake)."""
HEIGHT = 300
"""Height of a one-row chart in pixels; multi-row charts add `ROW` per extra row."""
ROW = 110


@dataclass(frozen=True)
class Run:
    """What the charts draw: the run, its honest twin (or None), and node positions."""

    model: MarketModel
    twin: MarketModel | None = None
    layout: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class Update:
    """New data for each trace, in build order, and layout changes (may be empty)."""

    traces: list[dict]
    layout: dict = field(default_factory=dict)


def _steps(n: int) -> np.ndarray:
    return np.arange(1, n + 1)


def _base(title: str, rows: int = 1, heights: list[float] | None = None) -> go.Figure:
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=heights
    )
    fig.update_layout(
        title={
            "text": title,
            "x": 0.01,
            "y": 1,
            "yref": "container",
            "yanchor": "top",
            "pad": {"t": 10},
            "font": {"size": 15, "color": INK},
        },
        height=HEIGHT + ROW * (rows - 1),
        margin={"l": 64, "r": 16, "t": 76, "b": 40},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"color": INK, "size": 12},
        legend={
            "orientation": "h",
            "x": 0,
            "y": 1.01,
            "yanchor": "bottom",
            "traceorder": "normal",
            "font": {"size": 11},
        },
        hovermode="x unified",
        uirevision="keep",  # keep the viewer's zoom across data updates
    )
    fig.update_xaxes(showgrid=False, linecolor=MUTED, ticks="outside", tickcolor=MUTED)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=MUTED)
    return fig


def _x_range(fig: go.Figure, model: MarketModel, rows: int = 1) -> None:
    for row in range(1, rows + 1):
        fig.update_xaxes(range=[0, model.config.n_steps + 1], row=row, col=1)
    fig.update_xaxes(title_text="step", row=rows, col=1)


def _line(name: str, color: str, width: float = 1.2, **kwargs) -> go.Scatter:
    return go.Scatter(
        x=[], y=[], name=name, mode="lines", line={"color": color, "width": width}, **kwargs
    )


def _xy(x: np.ndarray, y: np.ndarray) -> dict:
    return {"x": x, "y": y}


class Chart:
    """A live chart: subclasses set `title` and implement `build` and `update`."""

    title = ""

    def applies(self, run: Run) -> bool:
        return True

    def build(self, run: Run) -> go.Figure:
        raise NotImplementedError

    def update(self, run: Run) -> Update:
        raise NotImplementedError


def _release_keys(model: MarketModel) -> list[tuple[str, str]]:
    """Legend keys (label, color) for the kinds of clue this run releases."""
    config = model.config
    keys = []
    if config.n_clues_total > config.n_private:
        keys.append(("news release", NEWS))
    if config.n_fakes:
        keys.append(("fake release", FAKE))
    return keys


def _guide(x: float | None = None, y: float | None = None, **line) -> dict:
    """A full-height vertical line at step x, or a full-width horizontal one at y."""
    if x is not None:
        return {
            "type": "line",
            "xref": "x",
            "yref": "paper",
            "x0": x,
            "x1": x,
            "y0": 0,
            "y1": 1,
            "line": line,
            "opacity": 0.25,
        }
    return {
        "type": "line",
        "xref": "paper",
        "yref": "y",
        "x0": 0,
        "x1": 1,
        "y0": y,
        "y1": y,
        "line": line,
    }


class PriceChart(Chart):
    """Price and P* in cents, with a vertical guide at each news or fake release."""

    title = "Price vs P*"

    def build(self, run: Run) -> go.Figure:
        fig = _base(self.title)
        fig.add_trace(_line("price", PRICE))
        fig.add_trace(_line("P* (benchmark)", P_STAR, width=2.5))
        # Legend keys for the release guides, which are shapes and have no legend entry.
        for label, color in _release_keys(run.model):
            fig.add_trace(_line(label, color, width=1, visible="legendonly"))
        fig.update_yaxes(title_text="cents", range=[0, 101], tickvals=[0, 25, 50, 75, 100])
        _x_range(fig, run.model)
        return fig

    def update(self, run: Run) -> Update:
        model = run.model
        data = series(model)
        steps = _steps(len(data["price"]))
        fakes = set(model.fake_ids)
        shapes = [_guide(y=50, color=MUTED, width=1, dash="dot")]
        shapes += [
            _guide(x=step, color=FAKE if clue in fakes else NEWS, width=1)
            for clue, step in model.release_step.items()
        ]
        traces = [_xy(steps, 100 * data["price"]), _xy(steps, 100 * data["p_star"])]
        traces += [{} for _ in _release_keys(model)]
        return Update(traces, {"shapes": shapes})


class VolumeChart(Chart):
    """Contracts traded each step."""

    title = "Volume per step"

    def build(self, run: Run) -> go.Figure:
        fig = _base(self.title)
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                name="contracts",
                mode="lines",
                line={"color": PRICE, "width": 1, "shape": "hv"},
                fill="tozeroy",
                showlegend=False,
            )
        )
        fig.update_yaxes(title_text="contracts", rangemode="tozero")
        _x_range(fig, run.model)
        return fig

    def update(self, run: Run) -> Update:
        volume = series(run.model)["volume"]
        return Update([_xy(_steps(len(volume)), volume)])


def band(x: np.ndarray, low: np.ndarray, high: np.ndarray) -> tuple[list, list]:
    """A filled band from low to high as closed outlines, one per run of steps where
    both are known, separated by None so gaps stay empty."""
    known = np.isfinite(low) & np.isfinite(high)
    xs: list = []
    ys: list = []
    start = None
    for i, ok in enumerate([*known.tolist(), False]):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            run = slice(start, i)
            xs += [*x[run].tolist(), *x[run][::-1].tolist(), None]
            ys += [*low[run].tolist(), *high[run][::-1].tolist(), None]
            start = None
    return xs, ys


class BookChart(Chart):
    """Best bid to best ask around the price, and the spread below."""

    title = "Bid, ask, and spread"

    def applies(self, run: Run) -> bool:
        return run.model.book is not None

    def build(self, run: Run) -> go.Figure:
        fig = _base(self.title, rows=2, heights=[0.68, 0.32])
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                name="best bid to best ask",
                mode="lines",
                fill="toself",
                fillcolor=BAND,
                line={"color": BAND_EDGE, "width": 0.6},
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(_line("price", PRICE), row=1, col=1)
        fig.add_trace(_line("spread", PRICE, width=1, showlegend=False), row=2, col=1)
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                name="one side empty",
                mode="markers",
                marker={"symbol": "line-ns-open", "size": 10, "color": INK},
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="cents", range=[0, 100], row=1, col=1)
        fig.update_yaxes(title_text="spread (c)", rangemode="tozero", row=2, col=1)
        _x_range(fig, run.model, rows=2)
        return fig

    def update(self, run: Run) -> Update:
        data = series(run.model)
        steps = _steps(len(data["price"]))
        empty = steps[data["empty_book"] > 0]
        band_x, band_y = band(steps, data["best_bid"], data["best_ask"])
        return Update(
            [
                {"x": band_x, "y": band_y},
                _xy(steps, 100 * data["price"]),
                _xy(steps, data["spread"]),
                _xy(empty, np.zeros(len(empty))),
            ]
        )


class MarketMakerChart(Chart):
    """Book depth, the market maker's inventory, and its half-spread."""

    title = "Depth and market maker"
    PANELS = (
        ("depth", "depth"),
        ("mm_position", "MM position"),
        ("mm_half_spread", "MM half-spread (c)"),
    )

    def applies(self, run: Run) -> bool:
        return run.model.book is not None

    def build(self, run: Run) -> go.Figure:
        fig = _base(self.title, rows=3)
        window = run.model.config.depth_window
        for row, (_, label) in enumerate(self.PANELS, start=1):
            fig.add_trace(_line(label, PRICE, width=1, showlegend=False), row=row, col=1)
            title = f"depth (±{window}c)" if row == 1 else label
            fig.update_yaxes(title_text=title, title_font={"size": 11}, row=row, col=1)
        _x_range(fig, run.model, rows=3)
        return fig

    def update(self, run: Run) -> Update:
        data = series(run.model)
        steps = _steps(len(data["depth"]))
        return Update([_xy(steps, data[name]) for name, _ in self.PANELS])


class NetworkChart(Chart):
    """The agents' network, colored by who knows the most recently released clue."""

    title = "Who knows the latest clue"

    def build(self, run: Run) -> go.Figure:
        model, layout = run.model, run.layout
        xy = np.array([layout[i] for i in range(len(model.traders))])
        edge_x, edge_y = [], []
        for a, neighbors in enumerate(model.neighbors):
            for b in neighbors:
                if a < b:
                    edge_x += [layout[a][0], layout[b][0], None]
                    edge_y += [layout[a][1], layout[b][1], None]
        fig = _base(self.title)
        fig.update_layout(
            title_subtitle={"text": self._subtitle(run), "font": {"size": 12, "color": MUTED}}
        )
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line={"color": "#b9b8b3", "width": 0.5},
                opacity=0.6,
                hoverinfo="skip",
                showlegend=False,
            )
        )
        size = float(max(6.0, min(14.0, 2400 / max(len(xy), 1))))
        fig.add_trace(
            go.Scatter(
                x=xy[:, 0],
                y=xy[:, 1],
                mode="markers",
                marker={"size": size, "color": SPREAD_COLORS[UNKNOWN], "line": {"width": 0}},
                text=[f"agent {i}" for i in range(len(xy))],
                hoverinfo="text",
                showlegend=False,
            )
        )
        for state, label in (
            (UNKNOWN, "not yet"),
            (AT_RELEASE, "got it at release"),
            (FROM_NEIGHBOR, "heard from a neighbor"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    name=label,
                    marker={"size": 9, "color": SPREAD_COLORS[state]},
                )
            )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False, scaleanchor="x")
        fig.update_layout(height=500, hovermode="closest")
        return fig

    def _subtitle(self, run: Run) -> str:
        model = run.model
        config = model.config
        if config.n_clues_total == config.n_private and not config.n_fakes:
            return "This run releases no news or fakes"
        spread = latest_clue(model)
        if spread is None:
            return "No clue released yet"
        kind = "fake" if spread.fake else "news"
        side = "YES" if spread.points_yes else "NO"
        knows = int((spread.state != UNKNOWN).sum())
        return (
            f"Latest clue: {kind} #{spread.clue_id} (points {side}), released step "
            f"{spread.released}: {knows} of {len(spread.state)} agents know it"
        )

    def update(self, run: Run) -> Update:
        spread = latest_clue(run.model)
        n = len(run.model.traders)
        state = np.full(n, UNKNOWN) if spread is None else spread.state
        heard = FAKE if spread is not None and spread.fake else SPREAD_COLORS[FROM_NEIGHBOR]
        colors = [heard if s == FROM_NEIGHBOR else SPREAD_COLORS[int(s)] for s in state.tolist()]
        return Update(
            [{}, {"marker.color": colors}, {}, {}, {"marker.color": heard}],
            {"title.subtitle.text": self._subtitle(run)},
        )


def phase_spans(model: MarketModel) -> list[tuple[str, int, int]]:
    """The manipulation layer's phases as (phase, first step, last step), in order."""
    spans: list[tuple[str, int, int]] = []
    for step, phase in sorted(model.manipulator.phases.items()):
        if phase not in LAYER_PHASES:
            continue
        if spans and spans[-1][0] == phase and spans[-1][2] == step - 1:
            spans[-1] = (phase, spans[-1][1], step)
        else:
            spans.append((phase, step, step))
    return spans


def _phase_shapes(model: MarketModel) -> list[dict]:
    return [
        {
            "type": "rect",
            "xref": "x",
            "yref": "paper",
            "x0": first - 0.5,
            "x1": last + 0.5,
            "y0": 0,
            "y1": 1,
            "fillcolor": PHASE_COLORS[phase],
            "opacity": 0.12,
            "line": {"width": 0},
            "layer": "below",
        }
        for phase, first, last in phase_spans(model)
    ]


STRATEGY_PHASES = {"twin": (), "pump": ("pump", "hold", "dump"), "spoof": ("spoof",)}
"""The manipulation phases each strategy can shade."""


def _phase_keys(fig: go.Figure, model: MarketModel) -> int:
    """Add a legend key for each phase the strategy can shade; return how many."""
    phases = STRATEGY_PHASES[model.config.manipulator]
    for phase in phases:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                name=f"{phase} phase",
                mode="markers",
                marker={
                    "symbol": "square",
                    "size": 12,
                    "color": PHASE_COLORS[phase],
                    "opacity": 0.35,
                },
            )
        )
    return len(phases)


def _no_change(n: int) -> list[dict]:
    return [{} for _ in range(n)]


class ManipulatorChart(Chart):
    """The manipulator's position and P&L at P*, with its manipulation phases shaded."""

    title = "Manipulator: position and P&L"

    def applies(self, run: Run) -> bool:
        return run.model.manipulator is not None

    def build(self, run: Run) -> go.Figure:
        fig = _base(f"{self.title} ({run.model.config.manipulator})", rows=2)
        fig.add_trace(_line("position", PRICE, showlegend=False), row=1, col=1)
        fig.add_trace(_line("P&L", P_STAR, showlegend=False), row=2, col=1)
        _phase_keys(fig, run.model)
        fig.update_yaxes(title_text="YES contracts", row=1, col=1)
        fig.update_yaxes(title_text="P&L at P* ($)", row=2, col=1)
        _x_range(fig, run.model, rows=2)
        return fig

    def update(self, run: Run) -> Update:
        data = series(run.model)
        steps = _steps(len(data["manip_position"]))
        keys = len(STRATEGY_PHASES[run.model.config.manipulator])
        return Update(
            [_xy(steps, data["manip_position"]), _xy(steps, data["manip_pnl"]), *_no_change(keys)],
            {"shapes": _phase_shapes(run.model)},
        )


class TwinChart(Chart):
    """The manipulated price against the honest twin's on the same seed, and the gap:
    E3's distortion, step by step."""

    title = "Price vs the honest twin"

    def applies(self, run: Run) -> bool:
        return run.twin is not None

    def build(self, run: Run) -> go.Figure:
        fig = _base(f"{self.title} ({run.model.config.manipulator})", rows=2)
        fig.add_trace(_line("honest twin's price", TWIN, width=1.5), row=1, col=1)
        fig.add_trace(_line("price", PRICE), row=1, col=1)
        fig.add_trace(_line("price - twin price", P_STAR, showlegend=False), row=2, col=1)
        _phase_keys(fig, run.model)
        fig.update_yaxes(title_text="cents", range=[0, 100], row=1, col=1)
        fig.update_yaxes(title_text="distortion (c)", row=2, col=1)
        _x_range(fig, run.model, rows=2)
        return fig

    def update(self, run: Run) -> Update:
        price = 100 * series(run.model)["price"]
        twin = 100 * series(run.twin)["price"]
        n = min(len(price), len(twin))
        steps = _steps(n)
        return Update(
            [
                _xy(steps, twin[:n]),
                _xy(steps, price[:n]),
                _xy(steps, price[:n] - twin[:n]),
                *_no_change(len(STRATEGY_PHASES[run.model.config.manipulator])),
            ],
            {"shapes": _phase_shapes(run.model)},
        )


class OrdersChart(Chart):
    """The manipulator's own orders against the book: its best open bid (the fake bid
    while spoofing) and offer, with its buys and sells marked, and their sizes."""

    title = "Manipulator: its orders vs the book"

    def applies(self, run: Run) -> bool:
        return run.model.manipulator is not None

    def build(self, run: Run) -> go.Figure:
        fig = _base(f"{self.title} ({run.model.config.manipulator})", rows=2, heights=[0.65, 0.35])
        fig.add_trace(_line("best bid", BAND_EDGE, width=0.8), row=1, col=1)
        fig.add_trace(_line("best ask", BAND_EDGE, width=0.8, line_dash="dot"), row=1, col=1)
        fig.add_trace(_line("its bid", BUY_MARK, width=2, line_shape="hv"), row=1, col=1)
        fig.add_trace(_line("its offer", SELL_MARK, width=2, line_shape="hv"), row=1, col=1)
        for name, color, symbol in (
            ("it bought", BUY_MARK, "triangle-up"),
            ("it sold", SELL_MARK, "triangle-down"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    name=name,
                    mode="markers",
                    marker={"color": color, "symbol": symbol, "size": 8},
                ),
                row=1,
                col=1,
            )
        fig.add_trace(
            _line("its bid size", BUY_MARK, width=1, line_shape="hv", showlegend=False),
            row=2,
            col=1,
        )
        fig.add_trace(
            _line("its offer size", SELL_MARK, width=1, line_shape="hv", showlegend=False),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="cents", range=[0, 100], row=1, col=1)
        _phase_keys(fig, run.model)
        fig.update_yaxes(title_text="open size", rangemode="tozero", row=2, col=1)
        _x_range(fig, run.model, rows=2)
        return fig

    def update(self, run: Run) -> Update:
        data = series(run.model)
        steps = _steps(len(data["price"]))
        price = 100 * data["price"]
        bought, sold = data["manip_bought"] > 0, data["manip_sold"] > 0
        return Update(
            [
                _xy(steps, data["best_bid"]),
                _xy(steps, data["best_ask"]),
                _xy(steps, data["manip_bid"]),
                _xy(steps, data["manip_ask"]),
                {"x": steps[bought], "y": price[bought], "text": data["manip_bought"][bought]},
                {"x": steps[sold], "y": price[sold], "text": data["manip_sold"][sold]},
                _xy(steps, data["manip_bid_size"]),
                _xy(steps, data["manip_ask_size"]),
                *_no_change(len(STRATEGY_PHASES[run.model.config.manipulator])),
            ],
            {"shapes": _phase_shapes(run.model)},
        )


MAIN_CHARTS: tuple[tuple[Chart, ...], tuple[Chart, ...]] = (
    (PriceChart(), VolumeChart(), BookChart()),
    (NetworkChart(), MarketMakerChart()),
)
"""The two columns of charts every run shows (order book charts only with a book)."""
MANIPULATOR_CHARTS: tuple[Chart, ...] = (ManipulatorChart(), TwinChart(), OrdersChart())
"""The E3 charts, shown below the others when the run has a manipulator."""


CONFIG = {"responsive": True, "displaylogo": False}
"""Plotly config: fill the column and follow its width; no Plotly logo."""


LEGEND_WIDTH = 440
"""Pixels a legend row can use in a half-width column (the plot area's width)."""


def legend_rows(names: list[str]) -> int:
    """Rows a horizontal legend can wrap to, sized as if every entry were as wide as
    the widest (its symbol, ~40 px, plus ~7 px per character): Plotly sometimes wraps
    sooner than the entries' own widths suggest, and too much room beats a clipped
    title."""
    if not names:
        return 0
    widest = 40 + 7 * max(len(name) for name in names)
    per_row = max(1, int(LEGEND_WIDTH // widest))
    return -(-len(names) // per_row)


def fit_top_margin(fig: go.Figure) -> None:
    """Leave room above the plot for the title's lines and the legend's rows, so a
    legend that wraps never runs into the title."""
    names = [t.name for t in fig.data if t.showlegend is not False and t.name]
    lines = 2 if fig.layout.title.subtitle.text else 1
    fig.update_layout(margin_t=24 + 20 * lines + 22 * legend_rows(names))


def _figure(chart: Chart, run: Run) -> go.Figure:
    fig = chart.build(run)
    fit_top_margin(fig)
    return fig


def widget(chart: Chart, run: Run) -> go.FigureWidget:
    """A live Plotly widget for one chart, laid out for a run. The GUI makes one per
    chart and keeps it for the whole session (`reload` points it at later runs): each
    new widget sends the whole plotly.js library to the browser, about 5 MB."""
    figure = go.FigureWidget(_figure(chart, run))
    figure._config = CONFIG
    return figure


def reload(figure: go.FigureWidget, chart: Chart, run: Run) -> None:
    """Point an existing widget at a new run: its traces and layout replaced in place.

    The old shapes are cleared first. Assigning a layout only relayouts the keys it
    has, and a layout without shapes has no `shapes` key, so the browser would keep
    the old run's phase bands and release guides."""
    fig = _figure(chart, run)
    figure.layout.shapes = ()
    figure.data = ()
    figure.add_traces(fig.data)
    figure.layout = fig.layout


def apply(widget: go.FigureWidget, update: Update) -> None:
    """Push an update into a live figure: the trace data and other layout changes in one
    message, then the shapes, if they changed, in a second.

    Shapes are replaced whole. Merged into the batch, a list that both grows and changes
    an existing shape becomes `shapes` plus `shapes.1.x1` in one message, which the
    browser rejects."""
    layout = dict(update.layout)
    shapes = layout.pop("shapes", None)
    with widget.batch_update():
        for trace, data in zip(widget.data, update.traces, strict=True):
            if data:
                trace.update(data)
        if layout:
            widget.update_layout(layout)
    if shapes is not None and [s.to_plotly_json() for s in widget.layout.shapes] != shapes:
        widget.layout.shapes = shapes
