import math
from dataclasses import replace

import numpy as np
import pytest

from kalshi_model.config import Config
from kalshi_model.rng import make_streams
from kalshi_model.world import (
    clue_log_odds,
    deal_private_clues,
    make_world,
    news_schedule,
    p_star,
    sigmoid,
)

Q = 0.53


def by_hand(n_yes: int, n_no: int, q: float = Q) -> float:
    """Bayes' rule written out directly: 1 / (1 + ((1 - q) / q) ^ (n_yes - n_no))."""
    return 1 / (1 + ((1 - q) / q) ** (n_yes - n_no))


def world_for(seed: int, config: Config | None = None):
    return make_world(config or Config(), make_streams(seed)["world"])


# P* hand-made cases


@pytest.mark.parametrize(
    ("points_yes", "known", "expected"),
    [
        ([True, False, True], [], 0.5),
        ([True, False, True], [0], Q),
        ([True, False, True], [1], 1 - Q),
        ([True, False, True], [0, 1], 0.5),
        ([True, False, True], [0, 2], by_hand(2, 0)),
        ([True, True, True, False], [0, 1, 2, 3], by_hand(3, 1)),
        ([False, False, False], [0, 1, 2], by_hand(0, 3)),
    ],
)
def test_p_star_hand_cases(points_yes, known, expected):
    assert p_star(np.array(points_yes), known, Q) == pytest.approx(expected, rel=1e-12)


def test_p_star_with_two_yes_clues_matches_bayes_by_hand():
    # Prior 1/2. Two YES clues: P(clues | YES) = q^2, P(clues | NO) = (1 - q)^2.
    expected = Q**2 / (Q**2 + (1 - Q) ** 2)
    assert p_star(np.array([True, True]), [0, 1], Q) == pytest.approx(expected, rel=1e-12)


def test_p_star_counts_each_clue_once():
    points_yes = np.array([True, False])
    assert p_star(points_yes, [0, 0, 0], Q) == pytest.approx(Q)
    assert p_star(points_yes, [0, 0, 1, 1], Q) == pytest.approx(0.5)


def test_p_star_only_counts_known_clues():
    points_yes = np.array([True] * 5 + [False] * 5)
    assert p_star(points_yes, range(5), Q) == pytest.approx(by_hand(5, 0))
    assert p_star(points_yes, range(10), Q) == pytest.approx(0.5)


def test_p_star_is_symmetric_between_yes_and_no():
    points_yes = np.array([True, True, False, True])
    flipped = ~points_yes
    known = [0, 1, 2, 3]
    assert p_star(points_yes, known, Q) == pytest.approx(1 - p_star(flipped, known, Q))


def test_p_star_stays_finite_with_many_clues():
    many = 20_000
    assert p_star(np.ones(many, dtype=bool), range(many), Q) == 1.0
    assert p_star(np.zeros(many, dtype=bool), range(many), Q) == 0.0


@pytest.mark.parametrize("bad", [[-1], [3], [0, 5]])
def test_p_star_rejects_unknown_clue_ids(bad):
    with pytest.raises(ValueError):
        p_star(np.array([True, False, True]), bad, Q)


def test_one_clue_moves_a_fifty_cent_price_by_about_three_cents():
    assert clue_log_odds(Q) == pytest.approx(math.log(53 / 47))
    assert sigmoid(clue_log_odds(Q)) - 0.5 == pytest.approx(0.03)


@pytest.mark.parametrize("x", [-800.0, -30.0, -1.0, 0.0, 1.0, 30.0, 800.0])
def test_sigmoid_matches_formula_and_never_overflows(x):
    expected = 1 / (1 + math.exp(-x)) if x > -700 else 0.0
    assert sigmoid(x) == pytest.approx(expected, rel=1e-12, abs=1e-300)


# World draws


def test_world_shape_follows_config():
    world = world_for(1)
    assert world.n_clues == 80
    assert world.n_private == 60
    assert len(world.news_steps) == 20
    assert world.clue_accuracy == Q


def test_same_seed_gives_same_world():
    first, second = world_for(3), world_for(3)
    assert first.y == second.y
    assert np.array_equal(first.points_yes, second.points_yes)


def test_world_arrays_are_read_only():
    world = world_for(1)
    with pytest.raises(ValueError):
        world.points_yes[0] = not world.points_yes[0]
    with pytest.raises(ValueError):
        world.news_steps[0] = 0


def test_outcome_is_a_fair_coin_and_clues_point_to_truth_with_probability_q():
    worlds = [world_for(seed) for seed in range(4000)]
    yes_share = np.mean([w.y for w in worlds])
    correct = np.array([w.points_yes == w.y for w in worlds])
    # 4000 coin flips: sd 0.008. 320,000 clues: sd 0.0009.
    assert abs(yes_share - 0.5) < 0.03
    assert abs(correct.mean() - Q) < 0.004


def test_clue_directions_do_not_depend_on_the_news_split():
    # Moving the private/news boundary must not change the world drawn for a seed.
    base = world_for(5)
    more_news = world_for(5, replace(Config(), n_private=40))
    assert base.y == more_news.y
    assert np.array_equal(base.points_yes, more_news.points_yes)


# News schedule


def test_default_news_is_evenly_spaced_over_the_run():
    steps = news_schedule(20, 2000)
    assert len(steps) == 20
    assert steps[0] >= 1 and steps[-1] <= 2000
    assert set(np.diff(steps).tolist()) <= {95, 96}


def test_news_schedule_hand_case():
    assert news_schedule(3, 100).tolist() == [25, 50, 75]
    assert news_schedule(0, 100).tolist() == []


def test_news_schedule_never_releases_before_step_one():
    steps = news_schedule(10, 10)
    assert steps.min() == 1
    assert np.all(np.diff(steps) >= 0)


def test_news_released_by_step():
    world = world_for(0, Config(n_steps=100, n_clues_total=13, n_private=10))
    assert world.news_steps.tolist() == [25, 50, 75]
    assert world.news_released_by(0).tolist() == []
    assert world.news_released_by(24).tolist() == []
    assert world.news_released_by(25).tolist() == [10]
    assert world.news_released_by(74).tolist() == [10, 11]
    assert world.news_released_by(100).tolist() == [10, 11, 12]


def test_p_star_at_combines_held_private_clues_and_released_news():
    world = world_for(0, Config(n_steps=100, n_clues_total=13, n_private=10))
    held = [0, 3, 3, 7]
    assert world.p_star_at(0, held) == pytest.approx(world.p_star([0, 3, 7]))
    assert world.p_star_at(60, held) == pytest.approx(world.p_star([0, 3, 7, 10, 11]))
    assert world.p_star_at(100, range(10)) == pytest.approx(world.p_star(range(13)))


def test_p_star_at_rejects_news_ids_as_private():
    world = world_for(0)
    with pytest.raises(ValueError):
        world.p_star_at(0, [60])


# Dealing private clues


def test_deal_gives_distinct_private_clues_of_the_requested_size():
    hands = deal_private_clues([0, 1, 5, 60], 60, make_streams(0)["deal"])
    assert [len(hand) for hand in hands] == [0, 1, 5, 60]
    for hand in hands:
        assert len(set(hand.tolist())) == len(hand)
        assert all(0 <= i < 60 for i in hand)
    assert hands[3].tolist() == list(range(60))


def test_deal_caps_requests_at_the_pool_size():
    hands = deal_private_clues([61, 500], 60, make_streams(0)["deal"])
    assert [len(hand) for hand in hands] == [60, 60]


def test_deal_is_deterministic_per_seed():
    first = deal_private_clues([3, 4, 5], 60, make_streams(9)["deal"])
    second = deal_private_clues([3, 4, 5], 60, make_streams(9)["deal"])
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))


def test_deal_spreads_clues_evenly_over_the_pool():
    # 300 agents with 2 clues each from 60: each clue is held by about 10 agents.
    hands = deal_private_clues([2] * 300, 60, make_streams(4)["deal"])
    counts = np.bincount(np.concatenate(hands), minlength=60)
    assert counts.sum() == 600
    assert counts.mean() == 10
    assert counts.max() < 25


def test_deal_rejects_negative_counts():
    with pytest.raises(ValueError):
        deal_private_clues([-1], 60, make_streams(0)["deal"])


# Step 2 "done when": 10,000 worlds


@pytest.fixture(scope="module")
def final_p_stars():
    """Final P* (every private clue held, all news released) and Y for 10,000 seeds."""
    config = Config()
    worlds = [world_for(seed, config) for seed in range(10_000)]
    finals = np.array([w.p_star_at(config.n_steps, range(config.n_private)) for w in worlds])
    outcomes = np.array([w.y for w in worlds])
    return finals, outcomes


def test_most_final_p_stars_are_between_ten_and_ninety_percent(final_p_stars):
    finals, _ = final_p_stars
    share = np.mean((finals >= 0.1) & (finals <= 0.9))
    assert share >= 0.90


def test_final_p_star_is_calibrated(final_p_stars):
    finals, outcomes = final_p_stars
    bins = np.minimum((finals * 10).astype(int), 9)
    for b in range(10):
        in_bin = bins == b
        n = int(in_bin.sum())
        if n < 200:
            continue
        mean_p = finals[in_bin].mean()
        sd = math.sqrt(mean_p * (1 - mean_p) / n)
        assert abs(outcomes[in_bin].mean() - mean_p) < 4 * sd
