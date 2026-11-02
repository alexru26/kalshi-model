"""Custom configurations from TOML files, for exploring small or simple systems.

A file (examples in `configs/`) holds:

- `market = "kalshi" | "clob" | "amm"`: the market preset to start from (default
  "kalshi"). "kalshi" and "clob" are the two E1 markets; "amm" is a legacy feature no
  longer used by any experiment. The other defaults are the frozen `Config` defaults.
- Any `Config` field by name, such as `n_agents = 2`.
- An optional `[traits]` table pinning agent traits: one value for every agent, or a
  list with one value per agent (`n_clues = [10, 0]`). Unpinned traits are drawn from
  the config's distributions as usual (`traits.draw_traits`).

Command-line overrides (`plot_run.py --set key=value`) apply on top of the file; a key
`traits.<name>` pins a trait.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

from kalshi_model.config import Config, amm_config, clob_config
from kalshi_model.traits import fixed_values

MARKETS = {"kalshi": Config, "clob": clob_config, "amm": amm_config}
"""Market presets: the E1 markets (Kalshi rules, the default, and the fee-free CLOB) and
the legacy AMM."""

CONFIG_FIELDS = frozenset(f.name for f in fields(Config))


@dataclass(frozen=True)
class CustomConfig:
    """A run's settings: the config and any pinned traits (trait name -> value)."""

    config: Config
    fixed_traits: Mapping[str, object]


def load_custom(
    path: Path, overrides: Mapping[str, object] | None = None, market: str | None = None
) -> CustomConfig:
    """Read a TOML custom configuration. Raises ValueError for any invalid setting."""
    try:
        with Path(path).open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"{path}: {e}") from e
    return build_custom(data, overrides, market)


def build_custom(
    data: Mapping[str, object],
    overrides: Mapping[str, object] | None = None,
    market: str | None = None,
) -> CustomConfig:
    """Build a custom configuration from parsed settings.

    `overrides` take precedence over `data` (keys `traits.<name>` pin traits), and
    `market`, if given, over the file's `market`.
    """
    settings = dict(data)
    traits = settings.pop("traits", {})
    if not isinstance(traits, Mapping):
        raise ValueError("traits must be a table of trait name = value")
    traits = dict(traits)
    file_market = settings.pop("market", "kalshi")
    for key, value in (overrides or {}).items():
        if key.startswith("traits."):
            traits[key.removeprefix("traits.")] = value
        else:
            settings[key] = value

    market = market or file_market
    if market not in MARKETS:
        raise ValueError(f"market must be one of {', '.join(MARKETS)}, got {market!r}")
    unknown = sorted(set(settings) - CONFIG_FIELDS)
    if unknown:
        raise ValueError(f"unknown setting(s): {', '.join(unknown)} (see kalshi_model/config.py)")

    config = MARKETS[market](**settings)
    for name, value in traits.items():  # fail before the run, not during it
        fixed_values(name, value, config.n_agents)
    return CustomConfig(config, traits)


def parse_assignment(text: str) -> tuple[str, object]:
    """Split a command-line `key=value`. The value is read as TOML (so `2`, `0.5`,
    `true`, and `[10, 0]` keep their types); anything else is a string."""
    key, sep, raw = text.partition("=")
    key = key.strip()
    if not sep or not key:
        raise ValueError(f"expected key=value, got {text!r}")
    try:
        value = tomllib.loads(f"value = {raw}")["value"]
    except tomllib.TOMLDecodeError:
        value = raw.strip()
    return key, value
