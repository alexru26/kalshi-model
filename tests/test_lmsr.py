import itertools
import math

import numpy as np
import pytest

from kalshi_model.lmsr import LMSR, lmsr_cost, lmsr_price, subsidy_cents


def two_outcome_cost(q_yes: float, q_no: float, b: float) -> float:
    """Hanson's textbook form, C = b ln(e^(q_yes / b) + e^(q_no / b)), for small inputs."""
    return b * math.log(math.exp(q_yes / b) + math.exp(q_no / b))


@pytest.mark.parametrize("b", [5.0, 50.0, 100.0])
@pytest.mark.parametrize(("start", "k"), [(0, 1), (0, 10), (7, 3), (-12, 5), (20, -8)])
def test_cost_matches_textbook_lmsr(b, start, k):
    ours = lmsr_cost(start + k, b) - lmsr_cost(start, b)
    textbook = two_outcome_cost(start + k, 0, b) - two_outcome_cost(start, 0, b)
    assert ours == pytest.approx(textbook, rel=1e-12)


def test_hand_computed_first_contract():
    # b = 100: 100 x 100 x ln((1 + e^0.01) / 2) = 50.125c, rounded up for the buyer.
    assert LMSR(100).quote(1) == 51
    assert lmsr_price(0, 100) == 0.5


@pytest.mark.parametrize("net", [-1e6, -5000, -1, 0, 1, 5000, 1e6])
def test_price_is_a_probability_and_cost_never_overflows(net):
    price = lmsr_price(net, 50.0)
    assert 0 <= price <= 1
    assert math.isfinite(lmsr_cost(net, 50.0))
    if abs(net) < 1000:
        assert 0 < price < 1


def test_price_is_symmetric_and_increasing():
    nets = np.arange(-300, 301)
    prices = np.array([lmsr_price(n, 50.0) for n in nets])
    assert np.all(np.diff(prices) > 0)
    assert np.allclose(prices + prices[::-1], 1.0)


def test_marginal_price_is_the_derivative_of_cost():
    b, n, h = 50.0, 17.0, 1e-5
    derivative = (lmsr_cost(n + h, b) - lmsr_cost(n - h, b)) / (2 * h)
    assert derivative == pytest.approx(lmsr_price(n, b), rel=1e-8)


def test_cost_is_path_independent():
    b = 50.0
    one_trade = lmsr_cost(30, b) - lmsr_cost(0, b)
    pieces = [0, 4, 11, 25, 30]
    split = sum(lmsr_cost(hi, b) - lmsr_cost(lo, b) for lo, hi in itertools.pairwise(pieces))
    assert split == pytest.approx(one_trade, rel=1e-12)
    round_trip = (lmsr_cost(30, b) - lmsr_cost(0, b)) + (lmsr_cost(0, b) - lmsr_cost(30, b))
    assert round_trip == pytest.approx(0.0, abs=1e-12)


def test_rounding_favors_the_maker_by_under_a_cent_per_trade():
    split, whole = LMSR(50.0), LMSR(50.0)
    paid_split = sum(split.trade(k) for k in (3, 5, 2, 10))
    paid_whole = whole.trade(20)
    assert split.net == whole.net == 20
    assert 0 <= paid_split - paid_whole < 4


def test_buys_round_up_and_sales_round_down():
    maker = LMSR(50.0)
    exact_buy = 100 * (lmsr_cost(3, 50.0) - lmsr_cost(0, 50.0))
    assert maker.quote(3) == math.ceil(exact_buy)
    maker.trade(10)
    exact_sale = 100 * (lmsr_cost(10, 50.0) - lmsr_cost(6, 50.0))
    assert maker.quote(-4) == -math.floor(exact_sale)


def test_trade_updates_net_and_cash():
    maker = LMSR(50.0)
    start = maker.cash
    paid = maker.trade(5)
    received = maker.trade(-2)
    assert maker.net == 3
    assert maker.cash == start + paid + received
    assert paid > 0 > received


def test_subsidy_is_the_loss_bound_in_cents():
    assert subsidy_cents(100.0) == math.ceil(100 * 100 * math.log(2)) == 6932
    assert LMSR(100.0).cash == 6932


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("y", [True, False])
def test_maker_loss_is_at_most_b_ln_2(seed, y):
    rng = np.random.default_rng(seed)
    maker = LMSR(20.0)
    for k in rng.integers(-40, 41, size=60).tolist():
        maker.trade(k)
        assert maker.cash >= 0
    maker.settle(y)
    loss = subsidy_cents(20.0) - maker.cash
    assert maker.cash >= 0
    assert loss <= 100 * 20.0 * math.log(2)
    assert maker.net == 0


def test_worst_case_loss_approaches_the_bound():
    # Selling far more YES than b, then YES winning, costs nearly the full b ln 2.
    maker = LMSR(20.0)
    maker.trade(2000)
    maker.settle(True)
    loss = subsidy_cents(20.0) - maker.cash
    assert 0.99 * 100 * 20 * math.log(2) < loss <= 100 * 20 * math.log(2)


@pytest.mark.parametrize("start", [-40, 0, 25])
@pytest.mark.parametrize("limit", [0.3, 0.5, 0.62, 0.9])
def test_max_buy_below_stops_at_the_limit(start, limit):
    maker = LMSR(50.0)
    maker.trade(start)
    k = maker.max_buy_below(limit)
    if k:
        assert lmsr_price(maker.net + k, 50.0) <= limit
    assert lmsr_price(maker.net + k + 1, 50.0) > limit or maker.price > limit


@pytest.mark.parametrize("start", [-25, 0, 40])
@pytest.mark.parametrize("limit", [0.1, 0.38, 0.5, 0.7])
def test_max_sell_above_stops_at_the_limit(start, limit):
    maker = LMSR(50.0)
    maker.trade(start)
    k = maker.max_sell_above(limit)
    if k:
        assert lmsr_price(maker.net - k, 50.0) >= limit
    assert lmsr_price(maker.net - k - 1, 50.0) < limit or maker.price < limit


def test_limits_outside_the_price_allow_no_trade():
    maker = LMSR(50.0)
    assert maker.max_buy_below(0.4) == 0
    assert maker.max_sell_above(0.6) == 0
    with pytest.raises(ValueError):
        maker.max_buy_below(1.0)


@pytest.mark.parametrize("b", [0, -1.0])
def test_rejects_non_positive_liquidity(b):
    with pytest.raises(ValueError):
        LMSR(b)
