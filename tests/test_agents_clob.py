import math

import pytest
from conftest import make_traits

from kalshi_model.agents import kelly_contracts
from kalshi_model.config import Config, clob_config
from kalshi_model.model import MarketModel
from kalshi_model.orderbook import BUY, SELL
from kalshi_model.world import clue_log_odds, sigmoid

W = clue_log_odds(0.53)
TAKE = 1.0  # u >= patience: never posts
POST = 0.0  # u < patience: posts if it took nothing


def one_trader(config=None, **traits):
    """(trader, book): one agent in an order book, plus an account 'other' to trade against."""
    config = config or Config(
        n_agents=1, mean_degree=0, turns_per_step=1, n_news_seeds=0, n_clues_total=60
    )
    defaults = {"market_trust": 1e-6, "kelly_frac": 0.5, "patience": 0.5}
    model = MarketModel(config, 0, traits=make_traits(1, **(defaults | traits)), debug=True)
    model.book.open_account("other", 10**8)
    return model.traders[0], model.book


def believe(trader, net):
    """Give the trader `net` YES clues (negative: NO clues); returns its value in cents."""
    for i in range(abs(net)):
        trader.learn_own(1000 + i, net > 0)
    return 100 * sigmoid(net * W)


def test_trader_cash_lives_in_the_order_book():
    trader, book = one_trader()
    assert trader.account is book.account(trader.index)
    assert trader.cash == book.account(0).cash == 10_000


def test_takes_an_offer_worth_more_than_its_price_plus_fee():
    trader, book = one_trader()
    value = believe(trader, 10)  # about 77c
    book.submit("other", SELL, 60, 100)
    taken = trader.take_turn_clob(book, 0.0, TAKE, 1)
    fee = 0.07 * 60 * 40 / 100
    expected = kelly_contracts(value / 100, (60 + fee) / 100, 0.5, 10_000)
    assert taken == expected
    assert trader.position == expected
    assert book.open_orders(0) == []  # immediate-or-cancel: nothing rests


def test_taker_fee_can_remove_the_edge():
    # Value about 53.0c against an offer at 51: a 2c edge, below 1.5c min_edge + 1.75c fee.
    trader, book = one_trader(min_edge=1.5)
    believe(trader, 1)
    book.submit("other", SELL, 51, 5)
    assert trader.take_turn_clob(book, 0.0, TAKE, 1) == 0
    # Without fees the same edge is worth taking.
    trader, book = one_trader(
        clob_config(n_agents=1, mean_degree=0, turns_per_step=1, n_news_seeds=0, n_clues_total=60),
        min_edge=1.5,
    )
    believe(trader, 1)
    book.submit("other", SELL, 51, 5)
    assert trader.take_turn_clob(book, 0.0, TAKE, 1) > 0


def test_sweeps_only_as_deep_as_the_edge_allows():
    trader, book = one_trader(kelly_frac=1.0, cash=10**6, min_edge=2.0)
    value = believe(trader, 6)  # about 67.6c
    for price in (60, 63, 66, 69):
        book.submit("other", SELL, price, 1)
    trader.take_turn_clob(book, 0.0, TAKE, 1)
    paid = sorted(f.price for f in book.fills)
    assert paid == [60, 63]  # 66 + fee + 2c edge exceeds the value
    assert all(value - p - 0.07 * p * (100 - p) / 100 > 2.0 for p in paid)


def test_sells_a_bid_that_is_too_high():
    trader, book = one_trader()
    believe(trader, -10)  # about 23c
    book.submit("other", BUY, 40, 100)
    assert trader.take_turn_clob(book, 0.0, TAKE, 1) < 0
    assert trader.position < 0
    assert trader.cash + 100 * trader.position >= 0


def test_posts_a_bid_min_edge_below_value_when_nothing_is_worth_taking():
    trader, book = one_trader(min_edge=3.0)
    value = believe(trader, 4)  # about 61.8c
    book.submit("other", SELL, 70, 5)
    book.submit("other", BUY, 40, 5)
    assert trader.take_turn_clob(book, 0.0, POST, 1) == 0
    [order] = book.open_orders(0)
    assert (order.side, order.price) == (BUY, math.floor(value - 3.0))
    target = kelly_contracts(value / 100, order.price / 100, 0.5, 10_000)
    assert order.remaining == target


def test_posts_an_offer_when_its_value_is_below_the_market():
    trader, book = one_trader(min_edge=2.0)
    value = believe(trader, -4)  # about 38.2c
    book.submit("other", SELL, 60, 5)
    book.submit("other", BUY, 39, 5)  # selling at 39 is not worth it after the fee
    assert trader.take_turn_clob(book, 0.0, POST, 1) == 0
    [order] = book.open_orders(0)
    assert (order.side, order.price) == (SELL, math.ceil(value + 2.0))
    target = kelly_contracts(1 - value / 100, (100 - order.price) / 100, 0.5, 10_000)
    assert order.remaining == target


def test_posted_offer_never_crosses_the_best_bid():
    # Value about 38.2c and a bid at 40 that is not worth hitting after the 1.68c fee
    # and 1c min_edge: the offer would be ceil(39.2) = 40, so it moves up to 41.
    trader, book = one_trader(min_edge=1.0)
    believe(trader, -4)
    book.submit("other", BUY, 40, 5)
    assert trader.take_turn_clob(book, 0.0, POST, 1) == 0
    [order] = book.open_orders(0)
    assert (order.side, order.price) == (SELL, 41)
    book.check_invariants()


def test_impatient_agent_waits_instead_of_posting():
    trader, book = one_trader(patience=0.3)
    believe(trader, 4)
    book.submit("other", SELL, 70, 5)
    trader.take_turn_clob(book, 0.0, 0.5, 1)  # u = 0.5 >= patience 0.3
    assert book.open_orders(0) == []


def test_resting_orders_only_cover_the_gap_to_the_target():
    trader, book = one_trader(min_edge=3.0)
    believe(trader, 4)
    book.submit("other", SELL, 70, 5)
    trader.take_turn_clob(book, 0.0, POST, 1)
    first = sum(o.remaining for o in book.open_orders(0))
    trader.take_turn_clob(book, 0.0, POST, 2)
    assert sum(o.remaining for o in book.open_orders(0)) == first


def test_cancels_an_order_whose_edge_is_gone():
    trader, book = one_trader(min_edge=3.0)
    believe(trader, 4)
    book.submit("other", SELL, 70, 5)
    trader.take_turn_clob(book, 0.0, POST, 1)
    assert book.open_orders(0)
    believe(trader, -8)  # the agent now sees 4 net NO clues: its bid is no longer worth it
    trader.take_turn_clob(book, 0.0, TAKE, 2)
    assert all(o.side != BUY for o in book.open_orders(0))
    assert book.account(0).bid_reserve == 0


def test_cancels_a_stale_order_only_after_it_sat_behind_long_enough():
    config = Config(
        n_agents=1,
        mean_degree=0,
        turns_per_step=1,
        n_news_seeds=0,
        n_clues_total=60,
        stale_steps=20,
    )
    trader, book = one_trader(config, min_edge=3.0)
    believe(trader, 4)  # value 61.8: bids at 58
    book.submit("other", SELL, 70, 5)
    trader.take_turn_clob(book, 0.0, POST, 1)
    [order] = book.open_orders(0)
    book.submit("other", BUY, order.price + 3, 1)  # now 3c behind the best bid
    trader.take_turn_clob(book, 0.0, TAKE, 5)  # first seen stale at step 5
    trader.take_turn_clob(book, 0.0, TAKE, 24)  # 19 steps: keep
    assert [o.order_id for o in book.open_orders(0)] == [order.order_id]
    trader.take_turn_clob(book, 0.0, TAKE, 25)  # 20 steps: cancel to reprice
    assert book.open_orders(0) == []


def test_never_posts_beyond_what_it_can_afford():
    trader, book = one_trader(cash=100, kelly_frac=1.0, min_edge=1.0)
    believe(trader, 20)
    book.submit("other", SELL, 99, 1)
    trader.take_turn_clob(book, 0.0, POST, 1)
    account = book.account(0)
    assert account.free_cash >= 0
    assert account.bid_reserve <= 100
    book.check_invariants()


@pytest.mark.parametrize("u", [TAKE, POST])
def test_agent_with_no_clues_follows_the_market_and_does_not_trade(u):
    trader, book = one_trader(market_trust=3.0)
    book.submit("other", SELL, 55, 5)
    book.submit("other", BUY, 45, 5)
    assert trader.take_turn_clob(book, 0.0, u, 1) == 0
    # Its value equals the 50c midpoint, so no price gives it an edge: nothing rests.
    assert book.open_orders(0) == []
