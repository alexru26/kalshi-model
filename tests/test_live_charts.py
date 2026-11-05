"""The live Plotly charts and the manipulator's recorded columns behind them."""

import itertools

import numpy as np
import pytest

from kalshi_model import live
from kalshi_model.config import Config
from kalshi_model.gui import make_run
from kalshi_model.live_charts import (
    MAIN_CHARTS,
    MANIPULATOR_CHARTS,
    BookChart,
    ManipulatorChart,
    NetworkChart,
    OrdersChart,
    PriceChart,
    TwinChart,
    apply,
    band,
    legend_rows,
    phase_spans,
    reload,
    widget,
)
from kalshi_model.metrics import manipulator_pnl_at_p_star
from kalshi_model.model import MarketModel

ALL_CHARTS = [c for column in MAIN_CHARTS for c in column] + list(MANIPULATOR_CHARTS)


def _run(name: str, steps: int):
    preset = live.PRESETS[name]
    run = make_run(preset.values, preset.pins, live.DEFAULT_SEED)
    for _ in range(steps):
        run.model.step()
        if run.twin is not None:
            run.twin.step()
    return run


@pytest.mark.parametrize("strategy", ["twin", "pump", "spoof"])
def test_manipulator_columns_match_its_account(strategy):
    model = MarketModel(Config(manipulator=strategy), live.DEFAULT_SEED)
    model.run()
    data = model.datacollector.model_vars
    assert data["manip_pnl"][-1] == pytest.approx(manipulator_pnl_at_p_star(model))
    assert sum(data["manip_bought"]) - sum(data["manip_sold"]) == model.manipulator.closing_position
    assert data["manip_position"][-1] == model.manipulator.closing_position


def test_spoofing_shows_the_fake_bid_behind_the_best_bid():
    config = Config(manipulator="spoof")
    model = MarketModel(config, live.DEFAULT_SEED)
    model.run()
    data = live.series(model)
    spoofing = [t for t in model.manipulator.layer_steps if t < config.n_steps]
    rows = np.array(spoofing) - 1  # row r is step r + 1
    assert (data["manip_bid_size"][rows] >= config.spoof_size).mean() > 0.9
    behind = data["manip_bid"][rows] <= data["best_bid"][rows]
    assert behind[np.isfinite(data["manip_bid"][rows])].all()


def test_runs_without_a_manipulator_record_no_manipulator_columns():
    model = MarketModel(Config(), 1)
    model.step()
    assert not any(name.startswith("manip_") for name in model.datacollector.model_vars)


def test_only_pump_and_spoof_run_an_honest_twin():
    assert _run("E3: pump and dump", 0).twin.config == Config(manipulator="twin")
    assert _run("E3: spoofing", 0).twin.config == Config(manipulator="twin")
    assert _run("E3: honest twin", 0).twin is None
    assert _run("Level 2 (default)", 0).twin is None


@pytest.mark.parametrize(
    ("name", "shown"),
    [
        ("Level 2 (default)", 5),
        ("E3: honest twin", 7),
        ("E3: pump and dump", 8),
        ("E3: spoofing", 8),
    ],
)
def test_which_charts_apply(name, shown):
    run = _run(name, 0)
    assert sum(c.applies(run) for c in ALL_CHARTS) == shown


def test_amm_has_no_order_book_charts():
    run = make_run({**live.default_values(), "mechanism": "amm", "taker_fee_bp": 0}, {}, 1)
    assert not BookChart().applies(run)
    assert not OrdersChart().applies(run)


@pytest.mark.parametrize("steps", [0, 1, 1100])
@pytest.mark.parametrize("name", ["Level 2 (default)", "E3: pump and dump", "E3: spoofing"])
def test_every_chart_takes_its_updates(name, steps):
    run = _run(name, steps)
    for chart in ALL_CHARTS:
        if chart.applies(run):
            figure = widget(chart, run)
            apply(figure, chart.update(run))
            apply(figure, chart.update(run))  # unchanged shapes are a no-op


def test_price_chart_grows_along_the_full_step_axis():
    run = _run("Level 2 (default)", 120)
    figure = widget(PriceChart(), run)
    apply(figure, PriceChart().update(run))
    assert tuple(figure.layout.xaxis.range) == (0, run.model.config.n_steps + 1)
    price, p_star = figure.data[0], figure.data[1]
    assert list(price.x) == list(range(1, 121))
    assert np.allclose(np.array(p_star.y, dtype=float), 100 * live.series(run.model)["p_star"])
    releases = [s for s in figure.layout.shapes if s.xref == "x"]
    assert sorted(s.x0 for s in releases) == sorted(run.model.release_step.values())


def test_band_skips_steps_with_an_empty_side():
    x = np.array([1, 2, 3, 4, 5])
    low = np.array([40, 41, np.nan, 42, 43])
    high = np.array([44, 45, 46, np.nan, 47])
    xs, ys = band(x, low, high)
    assert xs == [1, 2, 2, 1, None, 5, 5, None]
    assert ys == [40.0, 41.0, 45.0, 44.0, None, 43.0, 47.0, None]


def test_network_colors_follow_the_latest_clue():
    run = _run("Level 2 (default)", 0)
    chart = NetworkChart()
    figure = widget(chart, run)
    while not run.model.release_step:
        run.model.step()
    for _ in range(10):
        run.model.step()
    apply(figure, chart.update(run))
    spread = live.latest_clue(run.model)
    colors = list(figure.data[1].marker.color)
    assert colors.count("#0b0b0b") == run.model.config.n_news_seeds  # got it at release
    assert len(colors) - colors.count("#d6d5d0") == int((spread.state != live.UNKNOWN).sum())
    assert f"news #{spread.clue_id}" in figure.layout.title.subtitle.text


def test_phase_spans_join_consecutive_steps():
    run = _run("E3: pump and dump", 1300)
    spans = phase_spans(run.model)
    assert [phase for phase, _, _ in spans] == ["pump", "hold", "dump"]
    for (_, _, end), (_, start, _) in itertools.pairwise(spans):
        assert start == end + 1
    assert spans[0][1] == run.model.config.manip_start


def test_manipulator_charts_shade_its_phases():
    run = _run("E3: pump and dump", 1300)
    for chart in (ManipulatorChart(), TwinChart(), OrdersChart()):
        figure = widget(chart, run)
        apply(figure, chart.update(run))
        assert len(figure.layout.shapes) == len(phase_spans(run.model))


def test_twin_gap_is_zero_before_the_manipulation_starts():
    run = _run("E3: spoofing", 1100)
    figure = widget(TwinChart(), run)
    apply(figure, TwinChart().update(run))
    gap = np.array(figure.data[2].y, dtype=float)
    start = run.model.config.manip_start
    assert np.all(gap[: start - 1] == 0)
    assert np.any(gap[start:] != 0)


@pytest.mark.parametrize(
    ("names", "rows"),
    [
        ([], 0),
        (["price", "P* (benchmark)"], 1),
        (["honest twin's price", "price", "a", "b", "c"], 3),
    ],
)
def test_legend_rows_size_every_entry_like_the_widest(names, rows):
    assert legend_rows(names) == rows


def test_wrapping_legends_get_room_above_the_plot():
    run = _run("E3: pump and dump", 0)
    one_row = widget(PriceChart(), run).layout.margin.t
    three_rows = widget(TwinChart(), run).layout.margin.t  # 5 entries, 2 per row
    assert three_rows - one_row == 2 * 22


def test_reload_points_a_widget_at_a_new_run():
    plain = _run("Level 2 (default)", 30)
    figure = widget(PriceChart(), plain)
    apply(figure, PriceChart().update(plain))
    fakes = _run("E2: disinformation ratio 0.5", 0)
    reload(figure, PriceChart(), fakes)
    assert [t.name for t in figure.data] == [
        "price",
        "P* (benchmark)",
        "news release",
        "fake release",
    ]
    assert len(figure.data[0].x) == 0
    apply(figure, PriceChart().update(fakes))  # the update matches the new traces
    reload(figure, PriceChart(), plain)
    assert len(figure.data) == 3
