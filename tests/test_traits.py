from dataclasses import fields, replace

import numpy as np
import pytest

from kalshi_model.config import Config
from kalshi_model.rng import make_streams
from kalshi_model.traits import TRAIT_NAMES, Traits, draw_traits


def draw(config: Config | None = None, seed: int = 0) -> Traits:
    return draw_traits(config or Config(), make_streams(seed)["traits"])


@pytest.fixture(scope="module")
def big() -> Traits:
    return draw(Config(n_agents=20_000, turns_per_step=1), seed=1)


def test_one_value_per_agent_for_every_trait():
    traits = draw()
    assert len(traits) == 300
    assert [f.name for f in fields(Traits)] == list(TRAIT_NAMES)
    for name in TRAIT_NAMES:
        assert getattr(traits, name).shape == (300,)


SINGLE_VALUED = (
    "w_hearsay",
    "market_trust",
    "w_depth",
    "noise",
    "min_edge",
    "patience",
    "kelly_frac",
    "g",
)
"""Traits every agent shares, each set by the Config field of the same name."""


def test_only_informedness_and_cash_vary_across_agents(big):
    config = Config()
    for name in SINGLE_VALUED:
        assert np.all(getattr(big, name) == getattr(config, name)), name
    assert np.all(big.activity == 1.0)  # equal chances of a turn
    assert len(np.unique(big.n_clues)) > 30
    assert len(np.unique(big.cash)) > 10_000


def test_single_values_follow_the_config():
    config = replace(Config(), w_hearsay=0.2, market_trust=7.0, noise=0.4, g=1.0)
    traits = draw(config)
    assert np.all(traits.w_hearsay == 0.2)
    assert np.all(traits.market_trust == 7.0)
    assert np.all(traits.noise == 0.4)
    assert np.all(traits.g == 1.0)


def test_drawn_traits_follow_their_distributions(big):
    config = Config()
    assert np.all(big.n_clues >= 0)
    assert big.n_clues.mean() == pytest.approx(config.n_clues_mean, rel=0.05)
    assert np.all(big.cash >= 1)
    assert big.cash.dtype == np.int64
    assert np.median(big.cash) == pytest.approx(config.cash_median, rel=0.05)


def test_informedness_is_heavy_tailed(big):
    # Most agents know nothing or little; a few know most of the 60-clue pool.
    assert 0.3 < np.mean(big.n_clues == 0) < 0.5
    assert np.quantile(big.n_clues, 0.9) >= 40


def test_same_seed_gives_same_population():
    first, second = draw(seed=4), draw(seed=4)
    for name in TRAIT_NAMES:
        assert np.array_equal(getattr(first, name), getattr(second, name))


def test_changing_one_trait_setting_leaves_the_others_unchanged():
    base = draw(seed=4)
    changed = draw(replace(Config(), noise=0.5, cash_sigma=1.0), seed=4)
    assert not np.array_equal(base.noise, changed.noise)
    assert not np.array_equal(base.cash, changed.cash)
    for name in TRAIT_NAMES:
        if name not in ("noise", "cash"):
            assert np.array_equal(getattr(base, name), getattr(changed, name)), name


def test_traits_are_read_only():
    traits = draw()
    with pytest.raises(ValueError):
        traits.cash[0] = 5


def test_mismatched_lengths_are_rejected():
    traits = draw()
    arrays = {name: np.array(getattr(traits, name)) for name in TRAIT_NAMES}
    arrays["g"] = arrays["g"][:-1]
    with pytest.raises(ValueError):
        Traits(**arrays)


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.argsort().argsort(), b.argsort().argsort())[0, 1])


def test_cash_rises_with_informedness_by_cash_rho():
    config = Config(n_agents=20_000, turns_per_step=1, cash_rho=0.0)
    unlinked = draw(config, seed=1)
    linked = draw(replace(config, cash_rho=0.8), seed=1)
    # A Gaussian copula with correlation 0.8 has rank correlation (6 / pi) asin(0.4), about
    # 0.79; ties in n_clues (40% hold none) pull the measured value somewhat below it.
    assert 0.65 < rank_correlation(linked.n_clues, linked.cash) < 0.79
    assert abs(rank_correlation(unlinked.n_clues, unlinked.cash)) < 0.03
    none, top = linked.n_clues == 0, linked.n_clues >= np.quantile(linked.n_clues, 0.9)
    assert np.median(linked.cash[top]) > 5 * np.median(linked.cash[none])


def test_cash_link_only_reassigns_cash_and_leaves_other_traits_unchanged():
    config = Config(cash_rho=0.0)
    unlinked = draw(config, seed=3)
    linked = draw(replace(config, cash_rho=0.5), seed=3)
    assert np.array_equal(np.sort(linked.cash), np.sort(unlinked.cash))
    assert not np.array_equal(linked.cash, unlinked.cash)
    for name in TRAIT_NAMES:
        if name != "cash":
            assert np.array_equal(getattr(linked, name), getattr(unlinked, name))


@pytest.mark.parametrize("rho", [-1.0, 1.0, 1.5])
def test_cash_rho_must_lie_strictly_inside_minus_one_to_one(rho):
    with pytest.raises(ValueError):
        Config(cash_rho=rho)
