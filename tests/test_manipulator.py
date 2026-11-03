import math
from dataclasses import replace

import numpy as np
import pytest

from kalshi_model.config import Config
from kalshi_model.manipulator import (
    DUMP,
    HOLD,
    HONEST,
    OWNER,
    PUMP,
    SPOOF,
    manipulator_traits,
)
from kalshi_model.metrics import COUNTERPARTIES, manipulation_metrics, run_metrics
from kalshi_model.model import MarketModel
from kalshi_model.orderbook import BUY, SELL
from kalshi_model.rng import make_streams
from kalshi_model.traits import draw_traits

SHORT = Config(
    n_steps=400,
    n_clues_total=70,
    manip_start=200,
    pump_max_steps=40,
    pump_hold=10,
    spoof_steps=100,
)
SEED = 3


def run(config: Config, seed: int = SEED) -> MarketModel:
    model = MarketModel(config, seed, debug=True)  # invariants checked every step
    model.run()
    return model


@pytest.fixture(scope="module")
def runs() -> dict[str, MarketModel]:
    """The default without a manipulator, the twin, and both manipulations, one seed."""
    return {s: run(replace(SHORT, manipulator=s)) for s in ("none", "twin", "pump", "spoof")}


def fills_on(model: MarketModel, steps: set[int]):
    """The manipulator's fills during `steps`, as (fill, manipulator bought?)."""
    start = 0
    for step, mark in enumerate(model.fill_marks, start=1):
        for fill in model.book.fills[start:mark]:
            if step in steps and OWNER in (fill.taker, fill.maker):
                yield fill, (fill.taker == OWNER) == (fill.taker_side == BUY)
        start = mark


@pytest.mark.parametrize(
    "overrides",
    [
        {"manipulator": "whale"},
        {"manipulator": "pump", "mechanism": "amm", "taker_fee_bp": 0},
        {"manipulator": "spoof", "spoof_ticks": 3},  # outside the 2c depth window
        {"manipulator": "twin", "manip_start": 2001},
        {"manipulator": "twin", "manip_n_clues": 61},
        {"manip_cash_share": 1.0},
        {"pump_push": 0},
    ],
)
def test_config_rejects_bad_manipulator_settings(overrides):
    with pytest.raises(ValueError):
        Config(**overrides)


def test_manipulator_traits_are_the_population_values_at_its_informedness():
    config = Config()
    traits = manipulator_traits(config, cash=123)
    population = draw_traits(config, make_streams(0)["traits"])
    assert traits["n_clues"] == config.manip_n_clues
    assert traits["w_depth"] == 0.0  # it knows which of the book's depth is its own
    assert traits["cash"] == 123
    assert traits["activity"] == math.inf  # a turn every step
    for name in ("w_hearsay", "market_trust", "noise", "min_edge", "patience", "kelly_frac", "g"):
        assert np.all(getattr(population, name) == traits[name]), name


def test_manipulator_gets_its_cash_share_and_clues_and_counts_in_p_star(runs):
    model = runs["twin"]
    man = model.manipulator
    assert man.initial_cash == round(0.05 * int(model.traits.cash.sum()))
    assert man.n_own == SHORT.manip_n_clues and man.known <= set(range(SHORT.n_private))
    assert man.known <= model._known_real
    assert model.manipulator not in model.traders


def test_adding_the_manipulator_leaves_world_population_and_news_unchanged(runs):
    none, twin = runs["none"], runs["twin"]
    assert none.manipulator is None
    assert none.world.y == twin.world.y
    assert np.array_equal(none.traits.cash, twin.traits.cash)
    assert none.release_step == twin.release_step
    # Agents start with the same clues; news reaches the same agents (it is off the network).
    assert [t.known for t in none.traders] == [t.known for t in twin.traders]


def test_every_configuration_is_identical_until_the_manipulation_starts(runs):
    twin = runs["twin"]
    cut = SHORT.manip_start - 1  # rows before the start step
    prices = twin.datacollector.get_model_vars_dataframe().price.to_numpy()[:cut]
    for strategy in ("pump", "spoof"):
        model = runs[strategy]
        assert model.manipulator.initial_cash == twin.manipulator.initial_cash
        assert model.manipulator.known == twin.manipulator.known
        other = model.datacollector.get_model_vars_dataframe().price.to_numpy()[:cut]
        assert np.array_equal(prices, other, equal_nan=True)
        assert model.p_star == twin.p_star


def test_twin_never_leaves_the_honest_rules(runs):
    man = runs["twin"].manipulator
    assert set(man.phases.values()) == {HONEST} and man.layer_steps == []
    assert len(man.phases) == SHORT.n_steps  # one turn every step


def test_pump_buys_then_holds_then_dumps_no_lower_than_its_start(runs):
    man = runs["pump"].manipulator
    order = [man.phases[t] for t in sorted(man.phases)]
    changes = [p for i, p in enumerate(order) if i == 0 or p != order[i - 1]]
    assert changes[:4] == [HONEST, PUMP, HOLD, DUMP]
    assert changes[-1] == HONEST
    assert order.index(PUMP) == SHORT.manip_start - 1
    steps = {phase: {t for t, p in man.phases.items() if p == phase} for phase in (PUMP, DUMP)}
    assert len(steps[PUMP]) <= SHORT.pump_max_steps + 1
    pumped = list(fills_on(runs["pump"], steps[PUMP]))
    dumped = list(fills_on(runs["pump"], steps[DUMP]))
    assert pumped and all(bought for _, bought in pumped)
    assert all(f.price <= math.ceil(man.start_price + SHORT.pump_push) for f, _ in pumped)
    assert all(not bought and f.price >= math.ceil(man.start_price) for f, bought in dumped)
    assert sum(f.qty for f, _ in dumped) <= sum(f.qty for f, _ in pumped)
    assert all(f.qty <= SHORT.manip_chunk for f, _ in pumped)


def test_spoof_keeps_a_fake_bid_behind_the_best_bid_and_a_real_offer_at_the_best_ask():
    config = replace(SHORT, manipulator="spoof", n_steps=260)
    model = MarketModel(config, SEED, debug=True)
    while model.steps < config.manip_start + 5:
        model.step()
    man, book = model.manipulator, model.book
    man._spoof(book)
    bids = [o for o in book.open_orders(OWNER) if o.side == BUY]
    asks = [o for o in book.open_orders(OWNER) if o.side == SELL]
    assert len(bids) == len(asks) == 1
    assert asks[0].price == book.best_ask
    others_best_bid = max(
        price
        for price in range(1, 100)
        if any(o.side == BUY and o.price == price and o.owner != OWNER for o in _orders(book))
    )
    assert bids[0].price == others_best_bid - config.spoof_ticks
    assert bids[0].remaining <= config.spoof_size
    # Repeating the turn keeps the real offer (and its place in the queue).
    man._spoof(book)
    assert [o.order_id for o in book.open_orders(OWNER) if o.side == SELL] == [asks[0].order_id]
    model.check_invariants()


def test_spoof_cancels_its_orders_when_it_stops(runs):
    man = runs["spoof"].manipulator
    steps = man.layer_steps
    assert steps == list(range(SHORT.manip_start, SHORT.manip_start + SHORT.spoof_steps))
    assert man.phases[steps[-1] + 1] == HONEST
    assert set(man.phases.values()) == {HONEST, SPOOF}


def test_manipulation_metrics_add_up(runs):
    twin = runs["twin"]
    twin_row = manipulation_metrics(twin)
    assert math.isnan(twin_row["premium"]) and math.isnan(twin_row["distortion_0000"])
    for strategy in ("pump", "spoof"):
        model = runs[strategy]
        row = manipulation_metrics(model, twin)
        assert set(row) == set(twin_row)
        # Money is conserved: what the manipulator gains, others and the fee pot lose.
        assert row["premium"] - row["taken_from_agents"] - row["taken_from_mm"] == pytest.approx(
            -row["fees_change"]
        )
        assert sum(row[f"sold_to_{c}"] for c in COUNTERPARTIES) == row["bought"]
        assert sum(row[f"bought_from_{c}"] for c in COUNTERPARTIES) == row["sold"]
        assert sum(row[f"share_{c}"] for c in COUNTERPARTIES) == pytest.approx(1.0)
        assert row["layer_steps"] == len(model.manipulator.layer_steps)
        # run_metrics still describes the agents only.
        assert set(run_metrics(model)) == set(run_metrics(twin))


def _orders(book):
    return [o for owner in book.accounts for o in book.open_orders(owner)]
