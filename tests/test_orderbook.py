import numpy as np
import pytest

from kalshi_model.fees import fee_cents
from kalshi_model.orderbook import BUY, SELL, InvariantError, OrderBook, OrderRejected

CASH = 10_000


def book_with(*owners, cash=CASH, taker_fee_bp=0, maker_fee_bp=0):
    book = OrderBook(taker_fee_bp=taker_fee_bp, maker_fee_bp=maker_fee_bp)
    for owner in owners:
        book.open_account(owner, cash)
    return book


# Matching


def test_crossing_order_trades_at_the_resting_price():
    book = book_with("a", "b", taker_fee_bp=700)
    book.submit("a", BUY, 40, 10)
    result = book.submit("b", SELL, 35, 4)
    assert [(f.price, f.qty, f.maker, f.taker) for f in result.fills] == [(40, 4, "a", "b")]
    assert result.order_id is None  # fully filled, nothing rests
    # Taker pays ceil(0.07 x 4 x 0.4 x 0.6 x 100) = ceil(6.72) = 7c; the maker pays nothing.
    assert result.fills[0].taker_fee == 7
    assert result.fills[0].maker_fee == 0
    a, b = book.account("a"), book.account("b")
    assert (a.cash, a.position) == (CASH - 160, 4)
    assert (b.cash, b.position) == (CASH + 160 - 7, -4)
    assert book.fees_collected == 7
    assert book.best_bid == 40 and book.depth(BUY, 0) == 6
    assert book.last_price == 40


def test_non_crossing_order_rests():
    book = book_with("a", "b")
    book.submit("a", BUY, 40, 5)
    result = book.submit("b", SELL, 41, 5)
    assert result.fills == []
    assert result.order_id is not None
    assert (book.best_bid, book.best_ask) == (40, 41)


def test_price_then_time_priority():
    book = book_with("a", "c", "d", "s")
    book.submit("a", BUY, 40, 3)
    book.submit("c", BUY, 40, 3)
    book.submit("d", BUY, 41, 2)
    result = book.submit("s", SELL, 39, 6)
    assert [(f.maker, f.price, f.qty) for f in result.fills] == [
        ("d", 41, 2),
        ("a", 40, 3),
        ("c", 40, 1),
    ]
    assert book.depth(BUY, 0) == 2


def test_sweep_stops_at_the_limit_price_and_rests_the_rest():
    book = book_with("m1", "m2", "m3", "t")
    book.submit("m1", SELL, 50, 1)
    book.submit("m2", SELL, 52, 1)
    book.submit("m3", SELL, 55, 1)
    result = book.submit("t", BUY, 52, 5)
    assert [f.price for f in result.fills] == [50, 52]
    assert book.best_bid == 52 and book.depth(BUY, 0) == 3
    assert book.best_ask == 55


def test_partial_fill_of_a_resting_order_keeps_its_place():
    book = book_with("a", "c", "s")
    book.submit("a", BUY, 40, 5)
    book.submit("c", BUY, 40, 5)
    book.submit("s", SELL, 40, 2)
    result = book.submit("s", SELL, 40, 4)
    assert [(f.maker, f.qty) for f in result.fills] == [("a", 3), ("c", 1)]


def test_ioc_order_never_rests():
    book = book_with("a", "b")
    book.submit("a", SELL, 50, 2)
    result = book.submit("b", BUY, 50, 5, ioc=True)
    assert sum(f.qty for f in result.fills) == 2
    assert result.order_id is None
    assert book.best_bid is None
    assert book.account("b").bid_reserve == 0


def test_post_only_order_that_would_cross_is_rejected_and_changes_nothing():
    book = book_with("a", "b")
    book.submit("a", SELL, 50, 2)
    with pytest.raises(OrderRejected):
        book.submit("b", BUY, 50, 1, post_only=True)
    assert book.account("b").cash == CASH
    assert book.depth(SELL, 0) == 2
    assert book.submit("b", BUY, 49, 1, post_only=True).order_id is not None


def test_self_trade_cancels_the_resting_order_instead_of_trading():
    book = book_with("a")
    bid = book.submit("a", BUY, 40, 3).order_id
    result = book.submit("a", SELL, 40, 2)
    assert result.fills == []
    assert result.self_cancelled == [bid]
    assert book.best_bid is None and book.best_ask == 40
    assert book.account("a").position == 0


# Fees


def test_taker_fee_uses_one_accumulator_per_order():
    # Four 1-contract fills at 50c cost 1.75c each exactly; Kalshi's per-order
    # accumulator charges ceil(7.00) = 7c in total, not 4 x 2c = 8c.
    book = book_with("m1", "m2", "m3", "m4", "t", taker_fee_bp=700)
    for maker in ("m1", "m2", "m3", "m4"):
        book.submit(maker, SELL, 50, 1)
    result = book.submit("t", BUY, 50, 4)
    assert [f.taker_fee for f in result.fills] == [2, 2, 2, 1]
    assert sum(f.taker_fee for f in result.fills) == fee_cents(4, 50, 700) == 7
    assert book.account("t").cash == CASH - 200 - 7


def test_sweep_across_prices_rounds_the_order_total_up_once():
    book = book_with("m1", "m2", "m3", "t", taker_fee_bp=700)
    for maker, price in (("m1", 50), ("m2", 51), ("m3", 52)):
        book.submit(maker, SELL, price, 1)
    result = book.submit("t", BUY, 52, 3)
    # Exact fees 1.75 + 1.7493 + 1.7472 = 5.2465c, rounded up once to 6c.
    assert sum(f.taker_fee for f in result.fills) == 6


def test_maker_fee_is_reserved_up_front_and_charged_through_the_accumulator():
    book = book_with("m", "t1", "t2", "t3", maker_fee_bp=175)
    book.submit("m", BUY, 50, 10)
    reserved = book.account("m").bid_reserve
    assert reserved == 10 * 50 + fee_cents(10, 50, 175)
    paid = 0
    for taker, qty in (("t1", 3), ("t2", 3), ("t3", 4)):
        paid += sum(f.maker_fee for f in book.submit(taker, SELL, 50, qty).fills)
    assert paid == fee_cents(10, 50, 175) == 5
    assert book.account("m").bid_reserve == 0
    assert book.account("m").cash == CASH - 500 - 5


def test_fee_free_book_charges_nothing():
    book = book_with("a", "b")
    book.submit("a", BUY, 50, 100)
    book.submit("b", SELL, 50, 100)
    assert book.fees_collected == 0


# Collateral and cancellation


def test_bids_reserve_their_full_cost():
    book = book_with("a", cash=100)
    with pytest.raises(OrderRejected):
        book.submit("a", BUY, 30, 4)  # 120c
    book.submit("a", BUY, 30, 3)  # 90c
    book.submit("a", BUY, 10, 1)  # 10c: everything is reserved
    assert book.account("a").free_cash == 0
    with pytest.raises(OrderRejected):
        book.submit("a", BUY, 1, 1)


def test_short_sales_reserve_the_worst_case_payout():
    book = book_with("a", cash=300)
    with pytest.raises(OrderRejected):
        book.submit("a", SELL, 40, 6)  # 6 x 60c at risk
    book.submit("a", SELL, 40, 5)  # 5 x 60c = 300c
    assert book.account("a").ask_reserve == 300


def test_held_contracts_cover_sales():
    book = book_with("b")
    book.open_account("a", 300)
    book.submit("b", SELL, 60, 5)
    book.submit("a", BUY, 60, 5)
    assert (book.account("a").cash, book.account("a").position) == (0, 5)
    # With no cash left, the 5 YES held still cover a sale of 5: if YES wins they are
    # delivered. That uses 5 x 10c of the 500c the contracts are worth if YES wins.
    book.submit("a", SELL, 90, 5)
    assert book.account("a").ask_reserve == 50
    # The remaining 450c covers 4 short sales at 1c (4 x 99c) but not a fifth.
    book.submit("a", SELL, 1, 4)
    with pytest.raises(OrderRejected):
        book.submit("a", SELL, 1, 1)


def test_taker_fee_counts_toward_collateral():
    book = book_with("m", taker_fee_bp=700)
    book.open_account("t", 100)
    book.submit("m", SELL, 50, 2)
    with pytest.raises(OrderRejected):
        book.submit("t", BUY, 50, 2)  # 100c + 4c fee
    assert book.submit("t", BUY, 50, 1).fills[0].taker_fee == 2


def test_cancel_releases_the_reserve_and_is_free():
    book = book_with("a")
    order = book.submit("a", BUY, 40, 10).order_id
    assert book.account("a").free_cash == CASH - 400
    assert book.cancel(order)
    assert book.account("a").free_cash == CASH
    assert book.account("a").cash == CASH
    assert not book.cancel(order)
    assert book.best_bid is None


def test_cancel_all_and_open_orders():
    book = book_with("a", "b")
    book.submit("a", BUY, 40, 1)
    book.submit("a", SELL, 60, 2)
    book.submit("b", BUY, 41, 1)
    assert sorted((o.side, o.price, o.remaining) for o in book.open_orders("a")) == [
        (BUY, 40, 1),
        (SELL, 60, 2),
    ]
    assert book.cancel_all("a") == 2
    assert book.open_orders("a") == []
    assert book.best_bid == 41


def test_can_submit_matches_submit_and_changes_nothing():
    book = book_with("a", cash=100, taker_fee_bp=700)
    assert book.can_submit("a", BUY, 30, 3)
    assert not book.can_submit("a", BUY, 30, 4)
    assert book.account("a").bid_reserve == 0


@pytest.mark.parametrize(
    ("side", "price", "qty"),
    [(BUY, 0, 1), (BUY, 100, 1), (SELL, 50, 0), ("hold", 50, 1), (BUY, 50.0, 1)],
)
def test_invalid_orders_raise(side, price, qty):
    book = book_with("a")
    with pytest.raises((ValueError, TypeError)):
        book.submit("a", side, price, qty)


def test_unknown_and_duplicate_accounts():
    book = book_with("a")
    with pytest.raises(KeyError):
        book.submit("zed", BUY, 50, 1)
    with pytest.raises(ValueError):
        book.open_account("a", 5)
    with pytest.raises(ValueError):
        book.open_account("neg", -1)


# Market data


def test_quotes_depth_and_imbalance():
    book = book_with("a", "b")
    assert book.best_bid is book.best_ask is book.midpoint is book.spread is None
    book.submit("a", BUY, 48, 3)
    book.submit("a", BUY, 47, 2)
    book.submit("a", BUY, 40, 9)
    book.submit("b", SELL, 52, 1)
    book.submit("b", SELL, 55, 4)
    assert (book.best_bid, book.best_ask, book.spread) == (48, 52, 4)
    assert book.midpoint == 50.0
    assert book.depth(BUY, 2) == 5  # 48 and 47, not 40
    assert book.depth(SELL, 2) == 1  # 52 only
    assert book.imbalance(2) == pytest.approx((5 - 1) / 6)
    assert book.imbalance(0) == pytest.approx((3 - 1) / 4)


def test_reference_price_rules():
    book = book_with("a", "b")
    assert book.reference_price(10) is None  # nothing quoted or traded
    book.submit("a", BUY, 30, 1)
    book.submit("b", SELL, 60, 1)
    assert book.reference_price(10) is None  # spread 30c is too wide; no trades yet
    book.submit("a", BUY, 60, 1)  # trades at 60
    book.submit("a", BUY, 45, 1)
    book.submit("b", SELL, 70, 1)
    assert book.reference_price(10) == 0.60  # spread 25c: use the last trade
    book.submit("b", SELL, 50, 1)
    assert book.reference_price(10) == pytest.approx(0.475)  # mid of 45 and 50


def test_imbalance_is_zero_on_an_empty_book():
    assert book_with("a").imbalance(2) == 0.0


# Settlement


@pytest.mark.parametrize("y", [True, False])
def test_settlement_pays_winners_and_cancels_orders(y):
    book = book_with("a", "b", taker_fee_bp=700)
    book.submit("a", BUY, 70, 10)
    book.submit("b", SELL, 70, 4)
    book.submit("b", SELL, 90, 3)  # still resting
    book.settle(y)
    a, b = book.account("a"), book.account("b")
    assert a.position == b.position == 0
    assert book.open_orders("a") == book.open_orders("b") == []
    if y:
        assert a.cash == CASH - 280 + 400
        assert b.cash == CASH + 280 - book.fees_collected - 400
    else:
        assert a.cash == CASH - 280
        assert b.cash == CASH + 280 - book.fees_collected
    assert a.cash + b.cash + book.fees_collected == 2 * CASH
    with pytest.raises(RuntimeError):
        book.submit("a", BUY, 50, 1)


def test_yes_and_no_are_symmetric():
    """Buying NO at x is selling YES at 100 - x: mirrored markets settle the same."""

    def play(mirror: bool, y: bool) -> list[int]:
        book = book_with("a", "b", "c", taker_fee_bp=700, maker_fee_bp=175)
        flip = {BUY: SELL, SELL: BUY}
        orders = [("a", BUY, 30, 5), ("b", SELL, 25, 3), ("c", BUY, 35, 4), ("b", SELL, 30, 6)]
        for owner, side, price, qty in orders:
            if mirror:
                side, price = flip[side], 100 - price
            book.submit(owner, side, price, qty)
        book.settle(y if not mirror else not y)
        return [book.account(o).cash for o in ("a", "b", "c")] + [book.fees_collected]

    for y in (True, False):
        assert play(False, y) == play(True, y)


# Invariants


def test_invariant_check_catches_tampering():
    book = book_with("a", "b")
    book.submit("a", BUY, 40, 2)
    book.check_invariants()
    book.account("a").cash += 1
    with pytest.raises(InvariantError):
        book.check_invariants()


def test_invariant_check_catches_a_wrong_reserve():
    book = book_with("a")
    book.submit("a", BUY, 40, 2)
    book.account("a").bid_reserve -= 1
    with pytest.raises(InvariantError):
        book.check_invariants()


@pytest.mark.parametrize("seed", range(50))
def test_random_trading_never_breaks_the_invariants(seed):
    """Random orders, cancels, IOC and post-only flags, fees, and tight budgets."""
    rng = np.random.default_rng(seed)
    book = OrderBook(taker_fee_bp=700, maker_fee_bp=int(rng.choice([0, 175])))
    owners = list(range(8))
    for owner in owners:
        book.open_account(owner, int(rng.integers(50, 3000)))
    rejected = 0
    for _ in range(400):
        owner = int(rng.choice(owners))
        if rng.random() < 0.15:
            orders = book.open_orders(owner)
            if orders:
                book.cancel(orders[int(rng.integers(len(orders)))].order_id)
        else:
            side = BUY if rng.random() < 0.5 else SELL
            price, qty = int(rng.integers(1, 100)), int(rng.integers(1, 20))
            flags = {"ioc": bool(rng.random() < 0.2), "post_only": bool(rng.random() < 0.2)}
            allowed = book.can_submit(owner, side, price, qty, **flags)
            try:
                book.submit(owner, side, price, qty, **flags)
                assert allowed
            except OrderRejected:
                assert not allowed
                rejected += 1
        book.check_invariants()
        for owner_id in owners:
            assert book.account(owner_id).cash >= 0
    assert rejected > 0  # budgets were tight enough to matter
    book.settle(bool(rng.random() < 0.5))
    book.check_invariants()
