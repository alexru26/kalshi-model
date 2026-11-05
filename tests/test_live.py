"""The GUI's logic: presets, trait pins, building runs, the latest-clue view, and charts."""

from dataclasses import fields

import numpy as np
import pytest

from kalshi_model import live
from kalshi_model.config import Config, clob_config
from kalshi_model.custom import load_custom
from kalshi_model.model import MarketModel


def _built(name: str):
    preset = live.PRESETS[name]
    return live.build_model(preset.values, preset.pins, live.DEFAULT_SEED)


@pytest.mark.parametrize(
    ("name", "config"),
    [
        ("Level 2 (default)", Config()),
        ("E1: fee-free CLOB", clob_config()),
        *[(f"E2: disinformation ratio {r}", Config(disinfo_ratio=r)) for r in live.E2_RATIOS],
        ("E3: honest twin", Config(manipulator="twin")),
        ("E3: pump and dump", Config(manipulator="pump")),
        ("E3: spoofing", Config(manipulator="spoof")),
    ],
)
def test_experiment_presets_run_the_experiment_configs(name, config):
    model = _built(name)
    assert model.config == config
    assert model.seed == live.DEFAULT_SEED


@pytest.mark.parametrize(("name", "file"), [("Level 0", "level_0"), ("Level 1", "level_1")])
def test_level_presets_match_their_config_files(name, file):
    custom = load_custom(live.CONFIGS / f"{file}.toml")
    model = _built(name)
    assert model.config == custom.config
    # The same agents as `plot_run.py --config` builds from the file.
    direct = MarketModel(custom.config, live.DEFAULT_SEED, fixed_traits=custom.fixed_traits)
    for f in fields(model.traits):
        assert np.array_equal(getattr(model.traits, f.name), getattr(direct.traits, f.name))
    assert set(live.PRESETS[name].pins) == set(custom.fixed_traits)


def test_a_preset_run_matches_the_same_run_built_directly():
    model = _built("E3: pump and dump")
    direct = MarketModel(Config(manipulator="pump"), live.DEFAULT_SEED)
    for _ in range(50):
        model.step()
        direct.step()
    assert model.price == direct.price
    assert model.p_star == direct.p_star


@pytest.mark.parametrize(
    ("text", "value"),
    [("", None), ("  ", None), ("5", 5), ("0.5", 0.5), ("[20, 0]", [20, 0])],
)
def test_parse_pin_reads_blank_numbers_and_lists(text, value):
    assert live.parse_pin("n_clues", text) == value


@pytest.mark.parametrize("text", ["abc", "true", "[1, true]", "[]", '"5"', "[1, [2]]"])
def test_parse_pin_rejects_anything_but_numbers(text):
    with pytest.raises(ValueError, match="trait n_clues"):
        live.parse_pin("n_clues", text)


@pytest.mark.parametrize("value", [5, 0.25, [20, 0], [0.5, 1.0, 2.5]])
def test_format_pin_round_trips_through_parse_pin(value):
    assert live.parse_pin("x", live.format_pin(value)) == value


def test_build_model_applies_pins():
    preset = live.PRESETS["Level 0"]
    model = live.build_model({**preset.values}, {"n_clues": [10, 0]}, 1)
    assert model.traits.n_clues.tolist() == [10, 0]


@pytest.mark.parametrize(
    ("values", "pins", "match"),
    [
        ({"turns_per_step": 400}, {}, "turns_per_step cannot exceed n_agents"),
        ({}, {"n_clues": [1, 2]}, "has 2 values for 300 agents"),
        ({}, {"courage": 1}, "unknown trait"),
        ({"n_agents": 1.5}, {}, "n_agents must be an integer"),
    ],
)
def test_build_model_rejects_invalid_settings(values, pins, match):
    with pytest.raises(ValueError, match=match):
        live.build_model({**live.default_values(), **values}, pins, 1)


def _run_until(model, done, limit=2000):
    while not done(model):
        assert model.steps < limit, "condition never met"
        model.step()


def test_latest_clue_is_none_before_any_release():
    model = _built("Level 2 (default)")
    assert live.latest_clue(model) is None


def test_latest_clue_marks_seeds_then_neighbors():
    model = _built("Level 2 (default)")
    _run_until(model, lambda m: bool(m.release_step))
    spread = live.latest_clue(model)
    assert not spread.fake
    assert spread.released == model.release_step[spread.clue_id]
    assert (spread.state == live.AT_RELEASE).sum() == model.config.n_news_seeds
    assert (spread.state == live.FROM_NEIGHBOR).sum() == 0

    for _ in range(15):
        model.step()
    later = live.latest_clue(model)
    assert later.clue_id == spread.clue_id  # news comes about every 95 steps
    assert (later.state == live.AT_RELEASE).sum() == model.config.n_news_seeds
    assert (later.state == live.FROM_NEIGHBOR).sum() > 0
    knowers = set(model.learned_step[later.clue_id])
    assert set(np.flatnonzero(later.state != live.UNKNOWN).tolist()) == knowers


def test_latest_clue_follows_the_newest_release_and_flags_fakes():
    model = _built("E2: disinformation ratio 0.5")
    _run_until(model, lambda m: any(c in m.release_step for c in m.fake_ids))
    spread = live.latest_clue(model)
    assert spread.fake
    assert spread.clue_id in model.fake_ids
    assert spread.points_yes == (not model.world.y)  # fakes point against the truth


def test_series_turns_missing_quotes_into_nan_and_aligns_columns():
    model = _built("Level 2 (default)")
    for _ in range(5):
        model.step()
    model.datacollector.model_vars["price"].append(0.5)  # a step half recorded
    data = live.series(model)
    assert {len(v) for v in data.values()} == {5}
    assert all(v.dtype == float for v in data.values())


def _lines(fig):
    return [line for ax in fig.axes for line in ax.get_lines()]


def test_charts_draw_a_run_in_progress_on_its_full_step_axis():
    model = _built("Level 2 (default)")
    for _ in range(30):
        model.step()
    layout = live.network_layout_for(model)
    for draw in (
        live.price_figure,
        live.volume_figure,
        live.book_figure,
        live.market_maker_figure,
        lambda m: live.network_figure(m, layout),
    ):
        fig = draw(model)
        assert fig is not None
        fig.canvas.draw()
    price = live.price_figure(model)
    assert price.axes[0].get_xlim() == (0, model.config.n_steps + 1)
    price_line, p_star_line = _lines(price)[1:3]  # after the 50c guide
    assert len(price_line.get_xdata()) == 30
    assert np.allclose(p_star_line.get_ydata(), 100 * live.series(model)["p_star"])


def test_charts_draw_before_the_first_step():
    model = _built("Level 2 (default)")
    for fig in (live.price_figure(model), live.book_figure(model), live.market_maker_figure(model)):
        fig.canvas.draw()


def test_amm_has_no_order_book_charts():
    model = live.build_model(
        {**live.default_values(), "mechanism": "amm", "taker_fee_bp": 0}, {}, 1
    )
    model.step()
    assert live.book_figure(model) is None
    assert live.market_maker_figure(model) is None
    live.price_figure(model).canvas.draw()


def test_network_title_reports_the_latest_clue():
    model = _built("Level 2 (default)")
    layout = live.network_layout_for(model)
    assert "no clue released yet" in live.network_figure(model, layout).axes[0].get_title("left")
    _run_until(model, lambda m: bool(m.release_step))
    spread = live.latest_clue(model)
    title = live.network_figure(model, layout).axes[0].get_title("left")
    assert f"news #{spread.clue_id}" in title
    assert f"{model.config.n_news_seeds} of 300 agents know it" in title


def test_network_title_says_when_a_run_has_no_news():
    model = _built("Level 0")
    title = live.network_figure(model, live.network_layout_for(model)).axes[0].get_title("left")
    assert "releases no news or fakes" in title
