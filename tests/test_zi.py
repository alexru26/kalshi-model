import numpy as np
import pytest

from kalshi_model.zi import othman_expected_price, simulate_zi_market

# Othman (2008), Section 4: expected trade prices for mean beliefs 0.8 and 0.2.
PAPER = [
    (0.8, 100, 0.781),
    (0.8, 25, 0.766),
    (0.8, 5, 0.742),
    (0.2, 100, 0.219),
    (0.2, 25, 0.234),
    (0.2, 5, 0.258),
]


@pytest.mark.parametrize(("mean", "concentration", "reported"), PAPER)
def test_formula_reproduces_the_papers_numbers(mean, concentration, reported):
    assert othman_expected_price(mean, concentration) == pytest.approx(reported, abs=6e-4)


def test_formula_is_symmetric_and_unbiased_at_one_half():
    assert othman_expected_price(0.5, 5) == pytest.approx(0.5, abs=1e-9)
    assert othman_expected_price(0.3, 25) == pytest.approx(1 - othman_expected_price(0.7, 25))


@pytest.mark.parametrize(("mean", "concentration", "reported"), PAPER)
def test_order_book_with_zi_traders_reproduces_othman(mean, concentration, reported):
    prices = simulate_zi_market(mean, concentration, 6000, np.random.default_rng(7)) / 100
    standard_error = prices.std() / np.sqrt(len(prices))
    expected = othman_expected_price(mean, concentration)
    # Within four standard errors plus the 0.2c that whole-cent prices can shift it.
    assert abs(prices.mean() - expected) < 4 * standard_error + 0.002


def test_zi_market_shows_the_favorite_longshot_bias_growing_with_diffuse_beliefs():
    rng = np.random.default_rng(3)
    favorite = {s: simulate_zi_market(0.8, s, 4000, rng).mean() / 100 for s in (5, 100)}
    longshot = {s: simulate_zi_market(0.2, s, 4000, rng).mean() / 100 for s in (5, 100)}
    assert favorite[5] < favorite[100] < 0.8
    assert longshot[5] > longshot[100] > 0.2


@pytest.mark.parametrize("seed", range(50))
def test_zi_runs_never_break_the_invariants(seed):
    # check_invariants=True re-checks the whole book after every order.
    simulate_zi_market(
        0.6, 5, 150, np.random.default_rng(seed), n_accounts=40, check_invariants=True
    )
