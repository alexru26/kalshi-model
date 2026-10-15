from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from kalshi_model.config import Config


def test_defaults_match_plan():
    config = Config()
    assert config.n_agents == 300
    assert config.n_steps == 2000
    assert config.taker_fee_bp == 700
    assert config.maker_fee_bp == 0


def test_config_is_frozen():
    config = Config()
    with pytest.raises(FrozenInstanceError):
        config.n_agents = 10


def test_replace_changes_one_field_and_keeps_the_rest():
    fee_free = replace(Config(), taker_fee_bp=0)
    assert fee_free.taker_fee_bp == 0
    assert asdict(fee_free) == {**asdict(Config()), "taker_fee_bp": 0}


def test_config_round_trips_through_a_dict():
    config = Config(n_agents=50, n_steps=10, taker_fee_bp=0)
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
    ],
)
def test_invalid_values_are_rejected(overrides):
    with pytest.raises(ValueError):
        Config(**overrides)
