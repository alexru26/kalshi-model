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


def test_bounded_traits_stay_in_bounds(big):
    for name in ("w_hearsay", "w_depth", "patience", "kelly_frac"):
        values = getattr(big, name)
        assert np.all((values > 0) & (values < 1)), name
    assert np.all(big.n_clues >= 0)
    assert np.all(big.market_trust > 0)
    assert np.all(big.noise >= 0)
    assert np.all(big.min_edge >= 0)
    assert np.all((big.g >= 0.6) & (big.g <= 1.0))
    assert np.all(big.cash >= 1)
    assert big.cash.dtype == np.int64
    assert np.all(big.activity > 0)


def test_trait_means_follow_the_config(big):
    config = Config()
    assert big.n_clues.mean() == pytest.approx(config.n_clues_mean, rel=0.05)
    assert big.market_trust.mean() == pytest.approx(
        config.market_trust_shape * config.market_trust_scale, rel=0.03
    )
    kelly_mean = config.kelly_alpha / (config.kelly_alpha + config.kelly_beta)
    assert big.kelly_frac.mean() == pytest.approx(kelly_mean, abs=0.005)
    assert big.w_depth.mean() == pytest.approx(0.1, abs=0.005)
    assert big.g.mean() == pytest.approx(0.8, abs=0.005)
    assert np.median(big.cash) == pytest.approx(config.cash_median, rel=0.05)
    # Half-normal mean is scale x sqrt(2 / pi).
    assert big.min_edge.mean() == pytest.approx(
        config.min_edge_scale * np.sqrt(2 / np.pi), rel=0.03
    )


def test_informedness_is_heavy_tailed(big):
    # Most agents know nothing or little; a few know most of the 60-clue pool.
    assert 0.3 < np.mean(big.n_clues == 0) < 0.5
    assert np.quantile(big.n_clues, 0.9) >= 40


def test_hearsay_falls_and_patience_rises_with_informedness(big):
    informed = big.n_clues >= 30
    uninformed = big.n_clues == 0
    assert big.w_hearsay[informed].mean() < big.w_hearsay[uninformed].mean()
    assert big.patience[informed].mean() > big.patience[uninformed].mean()


def test_without_the_links_traits_are_independent_of_informedness(big):
    unlinked = draw(Config(n_agents=20_000, turns_per_step=1, hearsay_gamma=0.0), seed=1)
    informed = unlinked.n_clues >= 30
    assert unlinked.w_hearsay[informed].mean() == pytest.approx(
        unlinked.w_hearsay[~informed].mean(), abs=0.02
    )


def test_same_seed_gives_same_population():
    first, second = draw(seed=4), draw(seed=4)
    for name in TRAIT_NAMES:
        assert np.array_equal(getattr(first, name), getattr(second, name))


def test_changing_one_trait_distribution_leaves_the_others_unchanged():
    base = draw(seed=4)
    noisier = draw(replace(Config(), noise_scale=0.5, kelly_alpha=3.0), seed=4)
    assert not np.array_equal(base.noise, noisier.noise)
    for name in TRAIT_NAMES:
        if name not in ("noise", "kelly_frac"):
            assert np.array_equal(getattr(base, name), getattr(noisier, name)), name


def test_g_equal_to_one_for_everyone():
    traits = draw(replace(Config(), g_low=1.0, g_high=1.0))
    assert np.all(traits.g == 1.0)


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
    config = Config(n_agents=20_000, turns_per_step=1)
    unlinked = draw(config, seed=1)
    linked = draw(replace(config, cash_rho=0.8), seed=1)
    # A Gaussian copula with correlation 0.8 has rank correlation (6 / pi) asin(0.4), about
    # 0.79; ties in n_clues (40% hold none) pull the measured value somewhat below it.
    assert 0.65 < rank_correlation(linked.n_clues, linked.cash) < 0.79
    assert abs(rank_correlation(unlinked.n_clues, unlinked.cash)) < 0.03
    none, top = linked.n_clues == 0, linked.n_clues >= np.quantile(linked.n_clues, 0.9)
    assert np.median(linked.cash[top]) > 5 * np.median(linked.cash[none])


def test_cash_link_only_reassigns_cash_and_leaves_other_traits_unchanged():
    config = Config()
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


def test_noise_falls_with_informedness_by_noise_delta():
    config = Config(n_agents=20_000, turns_per_step=1)
    flat = draw(config, seed=2)
    linked = draw(replace(config, noise_delta=0.06), seed=2)
    assert np.allclose(linked.noise, flat.noise * np.exp(-0.06 * flat.n_clues))
    none, many = linked.n_clues == 0, linked.n_clues >= 30
    assert linked.noise[many].mean() < 0.2 * linked.noise[none].mean()
    for name in TRAIT_NAMES:
        if name != "noise":
            assert np.array_equal(getattr(linked, name), getattr(flat, name))
