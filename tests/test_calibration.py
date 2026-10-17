import math
import pickle
from dataclasses import replace

import numpy as np
import pytest

from kalshi_model.calibration import (
    SIZE_QUANTILES,
    SPACE,
    Dimension,
    central_order,
    config_overrides,
    evaluate,
    latin_hypercube,
    measure,
    spread_out,
    targets,
)
from kalshi_model.config import Config
from kalshi_model.model import MarketModel

TARGET = {
    "spread": 2.0,
    "spread_band": (1.0, 6.0),
    "sizes": {"q25": 2.0, "q50": 10.0, "q75": 35.0, "q90": 105.0},
}


def patterns(**overrides):
    base = {
        "spread": 2.0,
        "empty_share": 0.0,
        "sizes": dict(TARGET["sizes"]),
        "price_brier": 0.20,
        "agent_brier": 0.22,
        "price_gap": 0.10,
        "agent_gap": 0.12,
        "informed_pnl": 5.0,
        "uninformed_pnl": -1.0,
    }
    return base | overrides


def test_a_perfect_fit_has_zero_distance_and_is_accepted():
    result = evaluate(patterns(), TARGET)
    assert result["accepted"]
    assert result["distance"] == pytest.approx(0.0)


def test_trade_sizes_within_a_factor_of_two_on_average_are_accepted():
    doubled = {q: 2 * v for q, v in TARGET["sizes"].items()}
    result = evaluate(patterns(sizes=doubled), TARGET)
    assert result["size_error"] == pytest.approx(math.log(2))
    assert result["met"]["sizes"]
    tripled = {q: 3 * v for q, v in TARGET["sizes"].items()}
    assert not evaluate(patterns(sizes=tripled), TARGET)["met"]["sizes"]


@pytest.mark.parametrize(
    ("spread", "met"), [(0.5, False), (1.0, True), (4.0, True), (6.0, True), (7.0, False)]
)
def test_spread_band(spread, met):
    assert evaluate(patterns(spread=spread), TARGET)["met"]["spread"] is met


def test_price_accuracy_against_agents_is_reported_but_not_required():
    result = evaluate(patterns(price_brier=0.23), TARGET)
    assert not result["met"]["beats_agents"]
    assert result["accepted"]
    assert evaluate(patterns(), TARGET)["met"]["beats_agents"]


@pytest.mark.parametrize(
    ("informed", "uninformed", "met"),
    [
        (5.0, -1.0, True),
        (-0.5, -1.0, True),  # informed lose less than the uninformed: they still earn more
        (-2.0, -1.0, False),  # informed do worse
        (5.0, 0.5, False),  # uninformed should not profit
    ],
)
def test_informed_must_earn_more_and_uninformed_must_lose(informed, uninformed, met):
    result = evaluate(patterns(informed_pnl=informed, uninformed_pnl=uninformed), TARGET)
    assert result["met"]["informed_earn"] is met


@pytest.mark.parametrize(("share", "met"), [(0.0, True), (0.02, True), (0.03, False)])
def test_book_must_rarely_be_empty(share, met):
    assert evaluate(patterns(empty_share=share), TARGET)["met"]["book_filled"] is met


def test_targets_come_from_the_committed_kalshi_summary():
    target = targets()
    assert set(target["sizes"]) == set(SIZE_QUANTILES)
    assert target["spread_band"][0] <= target["spread"] <= target["spread_band"][1]
    assert list(target["sizes"].values()) == sorted(target["sizes"].values())


def test_measure_reads_patterns_off_finished_runs():
    models = []
    for seed in range(2):
        model = MarketModel(Config(n_steps=300, n_clues_total=65), seed)
        model.run()
        models.append(model)
    measured = measure(models)
    assert measured["spread"] >= 1
    assert 0 <= measured["empty_share"] <= 1
    assert all(measured["sizes"][q] >= 1 for q in SIZE_QUANTILES)
    assert 0 <= measured["price_brier"] <= 1 and 0 <= measured["agent_brier"] <= 1
    assert math.isfinite(measured["informed_pnl"]) and math.isfinite(measured["uninformed_pnl"])


def test_models_pickle_for_multiprocessing():
    model = MarketModel(Config(n_steps=40, n_clues_total=62), 0)
    model.run()
    copy = pickle.loads(pickle.dumps(model))
    assert copy.price == model.price
    assert copy.datacollector.get_model_vars_dataframe().equals(
        model.datacollector.get_model_vars_dataframe()
    )


# Searching the parameter space


def test_latin_hypercube_puts_one_point_in_each_slice_of_every_axis():
    points = latin_hypercube(50, 4, np.random.default_rng(1))
    assert points.shape == (50, 4)
    assert ((points >= 0) & (points < 1)).all()
    for axis in range(4):
        assert sorted((points[:, axis] * 50).astype(int)) == list(range(50))


def test_dimensions_map_the_unit_interval_onto_their_range():
    linear = Dimension("x", 2.0, 4.0)
    log = Dimension("y", 1.0, 100.0, log=True)
    whole = Dimension("z", 3, 100, log=True, integer=True)
    assert (linear.value(0.0), linear.value(0.5), linear.value(1.0)) == (2.0, 3.0, 4.0)
    assert log.value(0.5) == pytest.approx(10.0)
    assert isinstance(whole.value(0.3), int) and 3 <= whole.value(0.3) <= 100


def test_overrides_are_valid_config_fields_and_kelly_mean_maps_to_beta():
    for u in latin_hypercube(20, len(SPACE), np.random.default_rng(2)):
        overrides = config_overrides(u)
        config = replace(Config(), **overrides)
        mean = config.kelly_alpha / (config.kelly_alpha + config.kelly_beta)
        assert 0.05 <= mean <= 0.5
        assert config.kelly_alpha + config.kelly_beta == pytest.approx(10.0)


def test_central_order_starts_at_the_point_nearest_the_median():
    points = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.4, 0.6], [0.9, 0.1]])
    order = central_order(points)
    assert order[0] == 1  # the median is (0.5, 0.5)
    assert sorted(order.tolist()) == list(range(5))


def test_spread_out_picks_far_apart_points_starting_from_the_given_one():
    corners = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.52, 0.5]])
    chosen = spread_out(corners, start=4, k=5)
    assert chosen[0] == 4
    assert set(chosen[1:]) == {0, 1, 2, 3}  # the four corners before the near-duplicate
    assert len(spread_out(corners, start=0, k=99)) == len(corners)
