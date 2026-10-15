from fractions import Fraction
from math import ceil

import pytest

from kalshi_model.config import Config
from kalshi_model.fees import fee_cents

TAKER_BP = 700
MAKER_BP = 175


def exact_fee_cents(contracts: int, price_cents: int, rate: Fraction) -> Fraction:
    """Kalshi's formula in exact dollars-to-cents arithmetic: rate x C x P x (1 - P) x 100."""
    p = Fraction(price_cents, 100)
    return rate * contracts * p * (1 - p) * 100


@pytest.mark.parametrize(
    ("price_cents", "one_contract", "hundred_contracts"),
    [
        (5, 1, 34),
        (10, 1, 63),
        (25, 2, 132),
        (50, 2, 175),
        (90, 1, 63),
    ],
)
def test_taker_fee_table(price_cents, one_contract, hundred_contracts):
    assert fee_cents(1, price_cents, TAKER_BP) == one_contract
    assert fee_cents(100, price_cents, TAKER_BP) == hundred_contracts


@pytest.mark.parametrize("rate_bp", [TAKER_BP, MAKER_BP])
@pytest.mark.parametrize("contracts", [1, 2, 3, 7, 10, 99, 100, 1000])
def test_fee_is_exact_formula_rounded_up(contracts, rate_bp):
    rate = Fraction(rate_bp, 10_000)
    for price in range(1, 100):
        assert fee_cents(contracts, price, rate_bp) == ceil(exact_fee_cents(contracts, price, rate))


@pytest.mark.parametrize("rate_bp", [TAKER_BP, MAKER_BP])
@pytest.mark.parametrize("contracts", [1, 3, 100])
def test_fee_symmetric_in_yes_and_no_price(contracts, rate_bp):
    for price in range(1, 100):
        assert fee_cents(contracts, price, rate_bp) == fee_cents(contracts, 100 - price, rate_bp)


def test_fee_largest_at_fifty_cents():
    fees = {price: fee_cents(100, price, TAKER_BP) for price in range(1, 100)}
    # Rounding up makes a plateau around 50c (47c to 53c all pay 175c), so 50c ties for max.
    assert fees[50] == max(fees.values())
    assert all(fees[p] <= fees[p + 1] for p in range(1, 50))


def test_fee_rounds_per_fill_not_per_contract():
    # 100 separate one-contract fills at 50c pay 200c; one 100-contract fill pays 175c.
    assert 100 * fee_cents(1, 50, TAKER_BP) == 200
    assert fee_cents(100, 50, TAKER_BP) == 175


def test_fee_hits_small_cheap_orders_hardest():
    # One contract at 5c pays 1c, 20% of its cost (PLAN.md Section 2.1).
    assert fee_cents(1, 5, TAKER_BP) * 100 == 20 * 5


def test_fee_is_never_below_one_cent_at_positive_rate():
    assert min(fee_cents(1, price, TAKER_BP) for price in range(1, 100)) == 1
    assert fee_cents(1, 1, 1) == 1


@pytest.mark.parametrize("price_cents", [1, 50, 99])
def test_zero_rate_costs_nothing(price_cents):
    assert fee_cents(1000, price_cents, 0) == 0


def test_fee_returns_int_and_accepts_integer_like_values():
    class Cents:
        def __index__(self) -> int:
            return 50

    result = fee_cents(1, Cents(), TAKER_BP)
    assert result == 2
    assert type(result) is int


def test_default_config_rates_give_kalshi_fees():
    config = Config()
    assert fee_cents(1, 50, config.taker_fee_bp) == 2
    assert fee_cents(100, 50, config.maker_fee_bp) == 0


@pytest.mark.parametrize(
    ("contracts", "price_cents", "rate_bp"),
    [
        (0, 50, TAKER_BP),
        (-1, 50, TAKER_BP),
        (1, 0, TAKER_BP),
        (1, 100, TAKER_BP),
        (1, -5, TAKER_BP),
        (1, 50, -1),
    ],
)
def test_fee_rejects_out_of_range_inputs(contracts, price_cents, rate_bp):
    with pytest.raises(ValueError):
        fee_cents(contracts, price_cents, rate_bp)


@pytest.mark.parametrize(
    ("contracts", "price_cents", "rate_bp"),
    [
        (1.0, 50, TAKER_BP),
        (1, 50.0, TAKER_BP),
        (1, 50, 0.07),
        (1, "50", TAKER_BP),
    ],
)
def test_fee_rejects_non_integers(contracts, price_cents, rate_bp):
    with pytest.raises(TypeError):
        fee_cents(contracts, price_cents, rate_bp)
