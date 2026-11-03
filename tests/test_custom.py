"""Custom configurations: TOML files, pinned traits, and the simple systems they allow."""

import subprocess
import sys
import tomllib
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from kalshi_model.config import Config
from kalshi_model.custom import build_custom, load_custom, parse_assignment
from kalshi_model.model import MarketModel
from kalshi_model.rng import make_streams
from kalshi_model.traits import TRAIT_NAMES, draw_traits, fixed_values

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs").glob("*.toml"))

# Two agents, no clues, no network, no market maker.
TINY = {
    "n_agents": 2,
    "turns_per_step": 2,
    "n_steps": 100,
    "n_clues_total": 0,
    "n_private": 0,
    "n_news_seeds": 0,
    "mean_degree": 0,
    "mm_size": 0,
}


def test_file_sets_market_fields_and_traits(tmp_path):
    path = tmp_path / "amm_pair.toml"
    path.write_text(
        'market = "amm"\nn_agents = 2\nturns_per_step = 2\nn_news_seeds = 1\n'
        "mean_degree = 0\n\n[traits]\nn_clues = [5, 0]\ncash = 50_000\n"
    )
    custom = load_custom(path)
    assert custom.config.mechanism == "amm"
    assert custom.config.taker_fee_bp == 0
    assert custom.config.n_agents == 2
    assert custom.config.n_steps == Config().n_steps  # unset fields keep the defaults
    assert custom.fixed_traits == {"n_clues": [5, 0], "cash": 50_000}


def test_overrides_beat_the_file_and_can_pin_traits():
    custom = build_custom(
        {"market": "amm", "n_steps": 100, "traits": {"cash": 10}},
        {"n_steps": 50, "traits.cash": 20, "traits.g": 1.0},
        market="clob",
    )
    assert custom.config.mechanism == "clob"
    assert custom.config.taker_fee_bp == 0
    assert custom.config.n_steps == 50
    assert custom.fixed_traits == {"cash": 20, "g": 1.0}


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"n_agent": 2}, "unknown setting"),
        ({"market": "lmsr"}, "market must be one of"),
        ({"n_agents": 0}, "n_agents must be positive"),
        ({"traits": 3}, "traits must be a table"),
        ({"traits": {"luck": 1}}, "unknown trait"),
        ({**TINY, "traits": {"cash": [1]}}, "1 values for 2 agents"),
    ],
)
def test_invalid_settings_fail_before_the_run(data, message):
    with pytest.raises(ValueError, match=message):
        build_custom(data)


def test_malformed_toml_is_a_value_error(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("n_agents = = 2\n")
    with pytest.raises(ValueError, match=r"bad\.toml"):
        load_custom(path)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("n_agents=2", ("n_agents", 2)),
        (" beta = 0.25 ", ("beta", 0.25)),
        ("traits.n_clues=[10, 0]", ("traits.n_clues", [10, 0])),
        ("network=sbm", ("network", "sbm")),
        ('network="sbm"', ("network", "sbm")),
    ],
)
def test_parse_assignment_keeps_types(text, expected):
    assert parse_assignment(text) == expected


@pytest.mark.parametrize("text", ["n_agents", "=2", ""])
def test_parse_assignment_rejects_missing_key_or_value(text):
    with pytest.raises(ValueError):
        parse_assignment(text)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("n_clues", -1),
        ("n_clues", 1.5),
        ("cash", 0),
        ("w_depth", 1.5),
        ("g", 0.0),
        ("activity", 0),
        ("patience", True),
        ("noise", "low"),
    ],
)
def test_pinned_trait_out_of_range_is_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        fixed_values(name, value, 3)


def test_pinning_one_trait_leaves_the_others_drawn_unchanged():
    config = Config(n_agents=50)
    base = draw_traits(config, make_streams(4)["traits"])
    pinned = draw_traits(config, make_streams(4)["traits"], {"w_depth": 0.0, "g": [1.0] * 50})
    assert np.all(pinned.w_depth == 0.0)
    assert np.all(pinned.g == 1.0)
    for name in set(TRAIT_NAMES) - {"w_depth", "g"}:
        assert np.array_equal(getattr(pinned, name), getattr(base, name)), name


def test_pinned_informedness_feeds_the_cash_link():
    # Cash rises with n_clues, so the agents pinned to many clues hold more of it.
    config = Config(n_agents=200, cash_sigma=2.0, cash_rho=0.95)
    pins = {"n_clues": [50] * 100 + [0] * 100}
    traits = draw_traits(config, make_streams(0)["traits"], pins)
    assert traits.n_clues.tolist() == pins["n_clues"]
    assert np.median(traits.cash[:100]) > 10 * np.median(traits.cash[100:])


def test_traits_and_fixed_traits_are_exclusive():
    custom = build_custom(TINY)
    model = MarketModel(custom.config, 0)
    with pytest.raises(ValueError, match="either"):
        MarketModel(custom.config, 0, traits=model.traits, fixed_traits={"g": 1.0})


def test_no_clues_and_no_market_maker():
    # The simplest market: two agents, nothing to know, no one quoting for them.
    model = MarketModel(build_custom(TINY).config, 1, fixed_traits={"noise": 0.5}, debug=True)
    model.run()
    assert model.p_star == 0.5
    assert model.book.fills, "noisy agents should trade with each other"
    assert all("market_maker" not in (f.maker, f.taker) for f in model.book.fills)
    assert model.market_maker.account.position == 0


def test_manipulator_clues_only_checked_with_a_manipulator():
    assert Config(n_private=0, n_clues_total=0).manipulator == "none"
    with pytest.raises(ValueError, match="manip_n_clues"):
        Config(n_private=0, n_clues_total=0, manipulator="twin")


def test_template_lists_every_setting_at_its_default():
    path = ROOT / "configs" / "template.toml"
    with path.open("rb") as f:
        settings = set(tomllib.load(f)) - {"traits"}
    assert settings == {f.name for f in fields(Config)}
    custom = load_custom(path)
    assert custom.config == Config()
    assert custom.fixed_traits == {}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_every_example_config_runs_with_invariants(path):
    custom = load_custom(path, {"n_steps": 60})
    model = MarketModel(custom.config, 0, fixed_traits=custom.fixed_traits, debug=True)
    model.run()
    assert model.settled


def test_plot_script_runs_a_config_file(tmp_path):
    out = tmp_path / "custom.png"
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments" / "plot_run.py"),
            "2",
            "--config",
            str(ROOT / "configs" / "test_1.toml"),
            "--set",
            "n_steps=80",
            "--set",
            "traits.n_clues=[0, 20]",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert str(out) in done.stdout
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_script_reports_a_bad_setting(tmp_path):
    done = subprocess.run(
        [sys.executable, str(ROOT / "experiments" / "plot_run.py"), "--set", "n_agent=2"],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 2
    assert "unknown setting(s): n_agent" in done.stderr
