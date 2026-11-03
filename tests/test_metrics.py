import math

import numpy as np
import pytest

from kalshi_model.config import Config, amm_config
from kalshi_model.metrics import (
    NEWS_WINDOW,
    RESPONSE_LAGS,
    degree_quarters,
    expected_pnl,
    fill_values,
    half_life,
    news_events,
    run_metrics,
)
from kalshi_model.model import MarketModel


def run(config: Config, seed: int = 0) -> MarketModel:
    model = MarketModel(config, seed, debug=True)
    model.run()
    return model


@pytest.fixture(scope="module")
def kalshi_run():
    """A short Kalshi run: 20 news clues over 800 steps, so several windows fit."""
    return run(Config(n_steps=800), 4)


# Half-life of a response curve


def test_half_life_interpolates_between_lags():
    linear = np.array(RESPONSE_LAGS, dtype=float)  # reaches half its end value halfway
    assert half_life(linear) == pytest.approx(NEWS_WINDOW / 2)
    step = np.where(np.array(RESPONSE_LAGS) >= 30, 1.0, 0.0)  # 0 at lag 25, 1 at lag 30
    assert half_life(step) == pytest.approx(27.5)


def test_half_life_is_zero_when_the_price_moves_at_once():
    assert half_life(np.ones(len(RESPONSE_LAGS))) == 0.0


def test_half_life_is_nan_when_the_price_does_not_move_toward_the_news():
    assert math.isnan(half_life(np.zeros(len(RESPONSE_LAGS))))
    assert math.isnan(half_life(-np.array(RESPONSE_LAGS, dtype=float)))


# News response


def test_news_events_measure_the_signed_move_from_the_step_before_release(kalshi_run):
    model = kalshi_run
    df = model.datacollector.get_model_vars_dataframe()
    price = 100 * df.price.to_numpy()
    events = news_events(model)
    full_windows = [
        (c, t)
        for c, t in sorted(model.release_step.items())
        if t - 1 + NEWS_WINDOW < model.config.n_steps
    ]
    assert len(events) == len(full_windows) > 0
    for event, (clue_id, t) in zip(events, full_windows, strict=True):
        sign = 1 if model.points_yes(clue_id) else -1
        assert event["before"] == price[t - 2]  # the step before release (row t - 2)
        lag = RESPONSE_LAGS.index(10)
        assert event["response"][lag] == pytest.approx(sign * (price[t - 1 + 10] - price[t - 2]))
        assert event["p_star_jump"] > 0  # a clue always moves P* its own way


def test_fakes_are_not_news_events():
    clean = news_events(run(Config(n_steps=800), 4))
    dirty = news_events(run(Config(n_steps=800, disinfo_ratio=0.2), 4))
    assert len(clean) == len(dirty) > 0
    assert [e["p_star_jump"] for e in clean] == [e["p_star_jump"] for e in dirty]


# P&L by fill role


def test_maker_and_taker_values_add_up_to_each_agents_pnl(kalshi_run):
    model = kalshi_run
    v = fill_values(model)
    for trader in model.traders:
        i = trader.index
        assert v["maker_value"][i] + v["taker_value"][i] == trader.pnl
    # The market maker only posts, and Kalshi makers pay no fee: agents pay every fee.
    assert v["fees"].sum() == model.book.fees_collected > 0


def test_pnl_at_p_star_from_closing_positions_matches_fills_valued_at_p_star(kalshi_run):
    # Two routes to the same number: each agent's closing position valued at P*, and
    # each of its fills valued at P*.
    model = kalshi_run
    by_fill = fill_values(model, payout=100 * model.p_star)
    assert np.allclose(expected_pnl(model), (by_fill["maker_value"] + by_fill["taker_value"]) / 100)
    assert not np.allclose(expected_pnl(model), [t.pnl / 100 for t in model.traders])


# The row of one run


def test_clob_row_is_in_cents_and_dollars(kalshi_run):
    model = kalshi_run
    row = run_metrics(model)
    df = model.datacollector.get_model_vars_dataframe()
    assert row["error"] == pytest.approx(100 * (df.price - df.p_star).abs().mean())
    assert row["fees"] == model.book.fees_collected / 100
    assert row["brier"] == (model.price - model.world.y) ** 2
    assert 0 <= row["agent_liquidity_share"] <= 1
    assert row["volume"] == sum(f.qty for f in model.book.fills)
    assert all(isinstance(v, float) for v in row.values())


def test_amm_row_has_no_order_book_metrics():
    row = run_metrics(run(amm_config(n_steps=300), 4))
    for metric in ("spread", "depth", "maker_return", "agent_liquidity_share"):
        assert math.isnan(row[metric])
    assert math.isfinite(row["error"]) and math.isfinite(row["maker_pnl"])


def test_unfinished_run_is_rejected():
    model = MarketModel(Config(n_steps=300), 0)
    model.step()
    with pytest.raises(ValueError):
        run_metrics(model)


def test_degree_quarters_are_disjoint_quarters_when_degrees_tie():
    # With one link per new agent the network is a tree, and two thirds of agents are
    # leaves of degree 1, so ties span the whole bottom quarter.
    model = MarketModel(Config(mean_degree=2), 0)
    degree = np.array([len(n) for n in model.neighbors])
    assert np.mean(degree == 1) > 0.5
    top, bottom = degree_quarters(model)
    quarter = model.config.n_agents // 4
    assert len(top) == len(set(top)) == quarter
    assert len(bottom) == len(set(bottom)) == quarter
    assert not set(top) & set(bottom)


def test_degree_quarters_rank_agents_by_degree():
    model = MarketModel(Config(), 0)
    degree = np.array([len(n) for n in model.neighbors])
    top, bottom = degree_quarters(model)
    assert degree[top].min() >= degree[bottom].max()
