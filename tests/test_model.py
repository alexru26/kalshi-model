import math

import numpy as np
import pytest
from conftest import make_traits

from kalshi_model.config import Config, amm_config
from kalshi_model.model import InvariantError, MarketModel


def run(config: Config, seed: int, **kwargs) -> MarketModel:
    model = MarketModel(config, seed, debug=True, **kwargs)
    model.run()
    return model


@pytest.fixture(scope="module")
def default_runs():
    """Default AMM population on 16 seeds."""
    return [run(amm_config(), seed) for seed in range(16)]


def test_clob_is_not_built_yet():
    with pytest.raises(NotImplementedError):
        MarketModel(Config(), 0)


def test_amm_rejects_fees():
    with pytest.raises(ValueError):
        Config(mechanism="amm")


def test_run_takes_n_steps_and_settles():
    model = run(amm_config(n_steps=50, n_clues_total=70), 0)
    assert model.steps == 50
    assert model.price == model.datacollector.get_model_vars_dataframe().price.iloc[-1]
    assert model.settled and not model.running
    assert len(model.datacollector.get_model_vars_dataframe()) == 50
    with pytest.raises(RuntimeError):
        model.step()


def test_same_seed_gives_identical_runs():
    first = run(amm_config(n_steps=300, n_clues_total=70), 3)
    second = run(amm_config(n_steps=300, n_clues_total=70), 3)
    a = first.datacollector.get_model_vars_dataframe()
    b = second.datacollector.get_model_vars_dataframe()
    assert a.equals(b)
    assert [t.cash for t in first.traders] == [t.cash for t in second.traders]


def test_population_world_and_news_are_shared_across_configurations():
    # Different market depth changes every trade, but not what the seed decides.
    shallow = run(amm_config(n_steps=400, lmsr_b=10.0), 5)
    deep = run(amm_config(n_steps=400, lmsr_b=200.0), 5)
    assert shallow.world.y == deep.world.y
    assert np.array_equal(shallow.world.points_yes, deep.world.points_yes)
    assert np.array_equal(shallow.traits.cash, deep.traits.cash)
    assert [t.known for t in shallow.traders] == [t.known for t in deep.traders]
    assert shallow.p_star == deep.p_star


def test_news_reaches_exactly_the_seed_agents():
    model = run(amm_config(n_steps=210, n_clues_total=62, n_news_seeds=7), 1)
    # Two news clues (ids 60, 61), released at steps 70 and 140.
    for clue_id in (60, 61):
        holders = [t for t in model.traders if clue_id in t.known]
        assert len(holders) == 7
    assert {60, 61} <= model._known_real


def test_news_with_no_seeds_is_known_to_nobody():
    model = run(amm_config(n_steps=210, n_clues_total=62, n_news_seeds=0), 1)
    assert not any(60 in t.known for t in model.traders)
    assert 60 not in model._known_real


def test_p_star_counts_every_clue_some_agent_knows():
    model = run(amm_config(n_steps=210, n_clues_total=62), 2)
    known = set().union(*(t.known for t in model.traders))
    assert model.p_star == pytest.approx(model.world.p_star(known))


@pytest.mark.parametrize("seed", range(5))
def test_money_and_contracts_are_conserved(seed):
    # debug=True checks the invariants after every step; this also checks settlement.
    model = run(amm_config(n_steps=500), seed)
    agents_pnl = sum(t.pnl for t in model.traders)
    maker_pnl = model.lmsr.cash - math.ceil(100 * model.config.lmsr_b * math.log(2))
    assert agents_pnl + maker_pnl == 0
    assert all(t.position == 0 and t.cash >= 0 for t in model.traders)
    assert model.lmsr.cash >= 0


def test_invariant_check_catches_created_money():
    model = MarketModel(amm_config(n_steps=20), 0, debug=True)
    model.step()
    model.traders[0].cash += 1
    with pytest.raises(InvariantError):
        model.check_invariants()


def test_invariant_check_catches_unbacked_contracts():
    model = MarketModel(amm_config(n_steps=20), 0, debug=True)
    model.traders[0].position += 1
    with pytest.raises(InvariantError):
        model.check_invariants()


def test_fully_informed_market_converges_to_p_star():
    # Limit case: everyone holds every private clue and gets every news clue, with no
    # noise, weighting, or minimum edge. The price must end within a cent or two of P*.
    n = 40
    config = amm_config(n_agents=n, turns_per_step=10, n_news_seeds=n, lmsr_b=20.0)
    for seed in range(4):
        traits = make_traits(n, n_clues=60, kelly_frac=0.5, cash=10**6)
        model = run(config, seed, traits=traits)
        df = model.datacollector.get_model_vars_dataframe()
        assert abs(df.price.iloc[-1] - df.p_star.iloc[-1]) < 0.02


def test_uninformed_market_stays_at_fifty_cents():
    traits = make_traits(30, noise=0.0)
    model = run(amm_config(n_agents=30, turns_per_step=5, n_news_seeds=0), 0, traits=traits)
    assert model.price == 0.5
    assert model.datacollector.get_model_vars_dataframe().volume.sum() == 0


def test_default_price_moves_toward_p_star(default_runs):
    # Beliefs are averages, so the price is roughly a weighted average of agents'
    # posteriors: it moves toward P* but falls short of it (PLAN.md Section 2.2).
    final_price = np.array([m.price for m in default_runs])
    final_p_star = np.array([m.p_star for m in default_runs])
    slope = np.polyfit(
        np.log(final_p_star / (1 - final_p_star)), np.log(final_price / (1 - final_price)), 1
    )[0]
    assert 0.05 < slope < 1
    errors, fifty = [], []
    for model in default_runs:
        df = model.datacollector.get_model_vars_dataframe().iloc[1000:]
        errors.append(np.abs(df.price - df.p_star).mean())
        fifty.append(np.abs(0.5 - df.p_star).mean())
    assert np.mean(errors) < np.mean(fifty)


def test_informed_agents_earn_more_than_uninformed(default_runs):
    gaps = []
    for model in default_runs:
        pnl = np.array([t.pnl for t in model.traders])
        n_clues = model.traits.n_clues
        gaps.append(pnl[n_clues >= 30].mean() - pnl[n_clues == 0].mean())
    assert np.mean(gaps) > 0
