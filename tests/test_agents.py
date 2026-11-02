import itertools
import math

import pytest
from conftest import make_traits

from kalshi_model.agents import (
    belief_log_odds,
    kelly_contracts,
    logit,
    perceived_yes,
    solvent,
)
from kalshi_model.config import amm_config
from kalshi_model.model import MarketModel
from kalshi_model.world import clue_log_odds, sigmoid

W = clue_log_odds(0.53)


def belief(**overrides):
    args = {
        "l_own": 0.0,
        "n_own": 0,
        "l_heard": 0.0,
        "n_heard": 0,
        "w_hearsay": 0.5,
        "market_trust": 3.0,
        "reference_price": None,
        "w_depth": 0.0,
        "imbalance": 0.0,
        "noise": 0.0,
        "z": 0.0,
        **overrides,
    }
    return belief_log_odds(**args)


# Belief rule


def test_without_a_price_the_belief_is_the_clue_evidence():
    assert belief(l_own=2 * W, n_own=4) == pytest.approx(2 * W)
    assert belief(l_own=W, n_own=1, l_heard=-3 * W, n_heard=3) == pytest.approx(W - 1.5 * W)


def test_an_agent_with_no_clues_follows_the_market():
    assert belief(reference_price=0.7) == pytest.approx(logit(0.7))


def test_the_price_adds_to_the_clues_discounted_by_what_the_agent_knows():
    # market_trust 3 and 3 own clues: the price gets weight 3 / (3 + 3) = 0.5.
    assert belief(l_own=3 * W, n_own=3, reference_price=0.4) == pytest.approx(
        3 * W + 0.5 * logit(0.4)
    )
    # Hearsay counts as w_hearsay clues each: n_eff = 1 + 0.5 x 4 = 3.
    value = belief(l_own=W, n_own=1, l_heard=4 * W, n_heard=4, reference_price=0.6)
    assert value == pytest.approx(W + 0.5 * 4 * W + 0.5 * logit(0.6))


def test_agreeing_clues_and_price_add_up_past_both():
    # Unlike averaging, the belief can be more extreme than either piece of evidence,
    # so the price can sum information no single agent holds.
    value = belief(l_own=5 * W, n_own=7, reference_price=0.7)
    assert value > max(5 * W, logit(0.7))


def test_the_more_an_agent_knows_the_less_it_leans_on_the_price():
    leans = [belief(l_own=0.0, n_own=n, reference_price=0.8) for n in (0, 1, 3, 10, 30)]
    assert leans[0] == pytest.approx(logit(0.8))
    assert all(a > b > 0 for a, b in itertools.pairwise(leans))


def test_depth_and_noise_add_to_the_belief():
    assert belief(w_depth=0.2, imbalance=-0.5) == pytest.approx(-0.1)
    assert belief(noise=0.3, z=2.0) == pytest.approx(0.6)


def test_perceived_probability_hand_cases():
    assert perceived_yes(0.0, 0.7) == 0.5
    assert perceived_yes(logit(0.8), 1.0) == pytest.approx(0.8)
    # g < 1 overweights the longshot: a 10% belief is perceived as more likely.
    assert perceived_yes(logit(0.1), 0.75) == pytest.approx(sigmoid(0.75 * logit(0.1)))
    assert perceived_yes(logit(0.1), 0.75) > 0.1


def test_perceived_yes_and_no_always_sum_to_one():
    for b in (0.03, 0.2, 0.5, 0.71):
        p_yes = perceived_yes(logit(b), 0.6)
        p_no = perceived_yes(logit(1 - b), 0.6)
        assert p_yes + p_no == pytest.approx(1.0)


# Kelly sizing


def test_kelly_hand_case():
    # p 0.6 at price 0.5: f = 0.2, stake 0.25 x 0.2 x 10,000c = 500c, 10 contracts at 50c.
    assert kelly_contracts(0.6, 0.5, 0.25, 10_000) == 10
    # NO side: p_no 0.7 at a NO price of 0.4: f = 0.5, stake 1000c, 25 contracts at 40c.
    assert kelly_contracts(0.7, 0.4, 0.2, 10_000) == 25


def test_kelly_takes_at_least_one_contract_and_none_without_edge():
    assert kelly_contracts(0.51, 0.5, 0.01, 100) == 1
    assert kelly_contracts(0.5, 0.5, 0.2, 10_000) == 0
    assert kelly_contracts(0.4, 0.5, 0.2, 10_000) == 0


def test_solvency_covers_both_outcomes():
    assert solvent(0, 0)
    assert solvent(500, -5)
    assert not solvent(499, -5)
    assert not solvent(-1, 3)


# Trading with the LMSR (legacy AMM: no longer in any experiment, kept working)


def one_trader(**traits):
    """A single-agent AMM market; returns (model, trader)."""
    config = amm_config(
        n_agents=1, mean_degree=0, turns_per_step=1, n_news_seeds=0, n_steps=100, n_clues_total=60
    )
    model = MarketModel(config, seed=0, traits=make_traits(1, **traits), debug=True)
    return model, model.traders[0]


def give_clues(trader, n_yes, n_no):
    for i in range(n_yes):
        trader.learn_own(1000 + i, True)
    for i in range(n_no):
        trader.learn_own(2000 + i, False)


def test_agent_with_no_clues_never_trades():
    model, trader = one_trader()
    assert trader.take_turn_amm(model.lmsr, 0.0) == 0
    assert trader.cash == 10_000


def test_informed_agent_buys_toward_its_belief_and_stops_at_the_limit():
    model, trader = one_trader(market_trust=0.001, kelly_frac=1.0, min_edge=1.0, cash=10**7)
    give_clues(trader, 10, 0)
    p_yes = sigmoid(10 * W)
    bought = trader.take_turn_amm(model.lmsr, 0.0)
    assert bought > 0
    assert model.lmsr.price <= p_yes - 0.01
    assert model.lmsr.price > p_yes - 0.01 - 0.01  # within a contract's move
    # Inside the no-trade band it holds.
    assert trader.take_turn_amm(model.lmsr, 0.0) == 0


def test_kelly_target_caps_the_trade():
    model, trader = one_trader(market_trust=0.001, kelly_frac=0.1)
    give_clues(trader, 10, 0)
    p_yes = sigmoid(10 * W)
    expected = kelly_contracts(p_yes, 0.5, 0.1, 10_000)
    assert trader.take_turn_amm(model.lmsr, 0.0) == expected
    assert trader.position == expected


def test_agent_never_trims_a_position_because_its_cash_fell():
    model, trader = one_trader(market_trust=0.001, kelly_frac=0.5)
    give_clues(trader, 10, 0)
    trader.take_turn_amm(model.lmsr, 0.0)
    held = trader.position
    # The price is still below the belief, but the cash-based target now sits below the
    # position. The agent must hold, not sell.
    for _ in range(5):
        assert trader.take_turn_amm(model.lmsr, 0.0) >= 0
    assert trader.position >= held


def test_agent_flips_to_no_when_its_clues_point_to_no():
    model, trader = one_trader(market_trust=0.001, kelly_frac=0.2)
    give_clues(trader, 10, 0)
    trader.take_turn_amm(model.lmsr, 0.0)
    assert trader.position > 0
    give_clues(trader, 0, 25)  # net -15
    sold = trader.take_turn_amm(model.lmsr, 0.0)
    assert sold < 0
    assert trader.position < 0
    assert model.lmsr.price < 0.5


def test_trades_are_capped_by_cash_and_keep_the_agent_solvent():
    model, trader = one_trader(market_trust=0.001, kelly_frac=1.0, cash=120)
    give_clues(trader, 30, 0)
    trader.take_turn_amm(model.lmsr, 0.0)
    assert trader.position >= 1
    assert 0 <= trader.cash < model.lmsr.quote(1)
    assert solvent(trader.cash, trader.position)


def test_short_sales_reserve_the_worst_case_payout():
    model, trader = one_trader(market_trust=0.001, kelly_frac=1.0, cash=300)
    give_clues(trader, 0, 30)
    trader.take_turn_amm(model.lmsr, 0.0)
    assert trader.position < 0
    assert trader.cash + 100 * trader.position >= 0


def test_min_edge_blocks_small_edges():
    model, trader = one_trader(market_trust=0.001, min_edge=5.0)
    give_clues(trader, 1, 0)  # perceived 53%: a 3c edge, below 5c
    assert trader.take_turn_amm(model.lmsr, 0.0) == 0


def test_probability_weighting_changes_what_the_agent_pays():
    plain_model, plain = one_trader(market_trust=0.001, kelly_frac=1.0, cash=10**7)
    weighted_model, weighted = one_trader(market_trust=0.001, kelly_frac=1.0, cash=10**7, g=0.6)
    for trader in (plain, weighted):
        give_clues(trader, 15, 0)
    plain.take_turn_amm(plain_model.lmsr, 0.0)
    weighted.take_turn_amm(weighted_model.lmsr, 0.0)
    # g < 1 pulls an 85% belief toward 50%, so the favorite is underbought.
    assert weighted_model.lmsr.price < plain_model.lmsr.price
    assert math.isclose(weighted_model.lmsr.price, sigmoid(0.6 * 15 * W), abs_tol=0.01)


def test_each_clue_counts_once():
    _, trader = one_trader()
    for _ in range(3):
        trader.learn_own(5, True)
    assert (trader.n_own, trader.own_net) == (1, 1)
