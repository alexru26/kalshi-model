import pytest

from kalshi_model.config import Config
from kalshi_model.market_maker import OWNER, MarketMaker
from kalshi_model.orderbook import BUY, SELL, OrderBook


def setup(**overrides):
    config = Config(**overrides)
    book = OrderBook(config.taker_fee_bp, config.maker_fee_bp)
    book.open_account("agent", 10**7)
    return book, MarketMaker(book, config)


def quotes(book):
    return sorted((o.side, o.price, o.remaining) for o in book.open_orders(OWNER))


def test_first_quotes_straddle_fifty_cents():
    book, mm = setup()
    mm.requote()
    assert quotes(book) == [(BUY, 48, 10), (SELL, 52, 10)]
    assert (mm.bid, mm.ask) == (48, 52)


def test_requote_replaces_the_old_quotes():
    book, mm = setup()
    mm.requote()
    mm.requote()
    assert len(book.open_orders(OWNER)) == 2


def test_estimate_is_a_moving_average_of_trade_prices():
    book, mm = setup(mm_alpha=0.5)
    book.open_account("seller", 10**7)
    book.submit("agent", BUY, 60, 1)
    book.submit("seller", SELL, 60, 1)  # trade at 60
    mm.requote()
    assert mm.estimate == pytest.approx(55.0)


def test_long_inventory_skews_both_quotes_down():
    book, mm = setup(mm_skew=0.1)
    mm.requote()
    book.submit("agent", SELL, 48, 10)  # the market maker buys 10 at 48
    assert mm.account.position == 10
    mm.requote()
    # Estimate 50 + 0.1 x (48 - 50) = 49.8; the fill marks out at a profit, so the
    # half-spread stays 2. Center 49.8 - 0.1 x 10 = 48.8: quotes 46 and 51.
    assert mm.extra_half_spread == 0.0
    assert (mm.bid, mm.ask) == (46, 51)


def test_spread_widens_after_an_adverse_fill_and_then_decays():
    book, mm = setup(mm_alpha=1.0, mm_widen=0.5, mm_decay=0.5)
    mm.requote()
    book.submit("agent", BUY, 52, 10)  # agent lifts the 52 offer
    book.open_account("seller", 10**7)
    book.submit("agent", BUY, 70, 1)
    book.submit("seller", SELL, 70, 1)  # then the price jumps to 70
    mm.requote()
    # The market maker sold 10 at 52 and the estimate is now 70: markout loss 180c,
    # so the half-spread grows by 0.5 x 180 / 10 = 9c.
    assert mm.extra_half_spread == pytest.approx(9.0)
    assert mm.half_spread == pytest.approx(10.0)  # capped at mm_max_half_spread
    mm.requote()
    assert mm.extra_half_spread == pytest.approx(4.5)


def test_profitable_fills_do_not_widen_the_spread():
    book, mm = setup(mm_alpha=1.0)
    mm.requote()
    book.submit("agent", BUY, 52, 10)  # trade at 52 moves the estimate to 52: markout 0
    mm.requote()
    assert mm.extra_half_spread == 0.0


def test_quotes_never_cross_agent_orders():
    book, mm = setup()
    book.submit("agent", SELL, 47, 1)
    mm.requote()
    assert mm.bid == 46  # one cent under the agent's 47 offer
    book.check_invariants()
    assert all(f.taker != OWNER for f in book.fills)


def test_quotes_never_cross_agent_bids():
    book, mm = setup()
    book.submit("agent", BUY, 55, 1)
    mm.requote()
    assert mm.ask == 56
    assert mm.bid == 48
    book.check_invariants()


def test_inventory_limit_stops_the_side_that_adds_to_it():
    book, mm = setup(mm_max_inventory=10)
    mm.requote()
    book.submit("agent", SELL, 48, 10)  # market maker now long 10
    mm.requote()
    assert mm.bid is None and mm.ask is not None


def test_size_shrinks_to_what_the_market_maker_can_afford():
    book, mm = setup(mm_cash=200)
    mm.requote()
    # 200c covers 4 contracts bid at 48 (192c) and 4 offered at 52 (4 x 48c = 192c).
    assert quotes(book) == [(BUY, 48, 4), (SELL, 52, 4)]


def test_broke_market_maker_quotes_nothing():
    book, mm = setup(mm_cash=0)
    mm.requote()
    assert quotes(book) == []
    assert mm.bid is None and mm.ask is None


def test_inventory_never_passes_the_cap():
    book, mm = setup(mm_max_inventory=5)
    mm.requote()
    assert sorted((o.side, o.remaining) for o in book.open_orders(OWNER)) == [
        (BUY, 5),
        (SELL, 5),
    ]
    book.submit("agent", SELL, 1, 50)  # sell into every bid the market maker shows
    assert mm.account.position == 5
    mm.requote()
    assert mm.bid is None
    book.submit("agent", BUY, 99, 50)  # buy every offer
    assert mm.account.position >= -5
