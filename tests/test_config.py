from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from kalshi_model.config import Config, amm_config, clob_config


def test_defaults_match_plan():
    config = Config()
    assert config.n_agents == 300
    assert config.n_steps == 2000
    assert config.taker_fee_bp == 700
    assert config.maker_fee_bp == 0
    assert config.n_clues_total == 80
    assert config.n_private == 60
    assert config.clue_accuracy == 0.53


def test_config_is_frozen():
    config = Config()
    with pytest.raises(FrozenInstanceError):
        config.n_agents = 10


def test_replace_changes_one_field_and_keeps_the_rest():
    fee_free = replace(Config(), taker_fee_bp=0)
    assert fee_free.taker_fee_bp == 0
    assert asdict(fee_free) == {**asdict(Config()), "taker_fee_bp": 0}


def test_config_round_trips_through_a_dict():
    config = Config(n_agents=50, n_steps=100, taker_fee_bp=0)
    assert Config(**asdict(config)) == config


def test_fields_are_keyword_only():
    with pytest.raises(TypeError):
        Config(300)


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_agents": 0},
        {"n_agents": -1},
        {"n_agents": 2.5},
        {"n_agents": True},
        {"n_steps": 0},
        {"n_steps": "2000"},
        {"taker_fee_bp": -1},
        {"taker_fee_bp": 0.07},
        {"maker_fee_bp": -175},
        {"maker_fee_bp": False},
        {"n_clues_total": -1},
        {"n_private": -1},
        {"n_private": 81},
        {"n_clues_total": 30, "n_private": 0, "n_steps": 29},
        {"clue_accuracy": 0.5},
        {"clue_accuracy": 1.0},
        {"clue_accuracy": 0.3},
        {"clue_accuracy": 1},
        {"mechanism": "lmsr"},
        {"mechanism": "amm"},
        {"mechanism": "amm", "taker_fee_bp": 0, "maker_fee_bp": 175},
        {"turns_per_step": 301},
        {"n_news_seeds": 301},
        {"lmsr_b": 0.0},
        {"noise_scale": -0.1},
        {"g_low": 0.9, "g_high": 0.8},
        {"g_high": 1.2},
        {"n_clues_mean": "20"},
        {"mm_size": -1},
        {"mm_alpha": 1.5},
        {"mm_decay": 1.0},
        {"mm_half_spread": 12.0},
        {"spread_cap": 0},
        {"mm_cash": -1},
        {"network": "lattice"},
        {"mean_degree": 5},
        {"mean_degree": 300},
        {"beta": -0.1},
        {"disinfo_ratio": -0.05},
    ],
)
def test_invalid_values_are_rejected(overrides):
    with pytest.raises(ValueError):
        Config(**overrides)


def test_amm_config_is_the_fee_free_lmsr():
    config = amm_config(lmsr_b=10.0)
    assert (config.mechanism, config.taker_fee_bp, config.maker_fee_bp) == ("amm", 0, 0)
    assert config.lmsr_b == 10.0
    assert config.n_agents == Config().n_agents


def test_float_parameters_accept_integers():
    assert Config(lmsr_b=50).lmsr_b == 50


def test_clob_config_is_the_fee_free_order_book():
    config = clob_config()
    assert (config.mechanism, config.taker_fee_bp, config.maker_fee_bp) == ("clob", 0, 0)
    assert Config().mechanism == "clob" and Config().taker_fee_bp == 700
