"""The single configuration object for a model run.

Every parameter lives here (PLAN.md Section 4). Values are starting defaults until
calibration (PLAN.md Section 4, Step 7) freezes them. Later build steps add their
parameters to this class.
"""

from dataclasses import dataclass


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True, kw_only=True)
class Config:
    """Parameters of one run. Experiments vary one field at a time around the defaults."""

    n_agents: int = 300
    n_steps: int = 2000

    # Fee rates in basis points (700 = 0.07). Kalshi's default market charges takers
    # 0.07 and makers nothing; set both to 0 for the AMM and the fee-free CLOB.
    taker_fee_bp: int = 700
    maker_fee_bp: int = 0

    # World (PLAN.md Section 2.4): M clues, each pointing to the truth with probability
    # q. The first K are the private pool; the other M - K are released as news.
    n_clues_total: int = 80
    n_private: int = 60
    clue_accuracy: float = 0.53

    def __post_init__(self) -> None:
        for name in ("n_agents", "n_steps", "n_clues_total"):
            value = getattr(self, name)
            if not _is_int(value) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        for name in ("taker_fee_bp", "maker_fee_bp", "n_private"):
            value = getattr(self, name)
            if not _is_int(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
        if self.n_private > self.n_clues_total:
            raise ValueError(
                f"n_private ({self.n_private}) cannot exceed n_clues_total ({self.n_clues_total})"
            )
        if self.n_clues_total - self.n_private > self.n_steps:
            raise ValueError("there must be at least one step per news clue")
        q = self.clue_accuracy
        if not isinstance(q, float) or not 0.5 < q < 1:
            raise ValueError(f"clue_accuracy must be a float strictly between 0.5 and 1, got {q!r}")
