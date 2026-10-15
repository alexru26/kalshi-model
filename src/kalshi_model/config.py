"""The single configuration object for a model run.

Every parameter lives here (PLAN.md Section 4). Values are starting defaults until
calibration (PLAN.md Section 4, Step 7) freezes them. Later build steps add their
parameters to this class.
"""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class Config:
    """Parameters of one run. Experiments vary one field at a time around the defaults."""

    n_agents: int = 300
    n_steps: int = 2000

    # Fee rates in basis points (700 = 0.07). Kalshi's default market charges takers
    # 0.07 and makers nothing; set both to 0 for the AMM and the fee-free CLOB.
    taker_fee_bp: int = 700
    maker_fee_bp: int = 0

    def __post_init__(self) -> None:
        for name in ("n_agents", "n_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        for name in ("taker_fee_bp", "maker_fee_bp"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
