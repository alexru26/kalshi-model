import numpy as np

from kalshi_model.traits import TRAIT_NAMES, Traits

# A plain, deterministic trader: no noise, no weighting, no minimum edge.
PLAIN = {
    "n_clues": 0,
    "w_hearsay": 0.5,
    "market_trust": 3.0,
    "w_depth": 0.0,
    "noise": 0.0,
    "min_edge": 0.0,
    "patience": 0.5,
    "kelly_frac": 0.2,
    "g": 1.0,
    "cash": 10_000,
    "activity": 1.0,
}


def make_traits(n: int, **overrides: object) -> Traits:
    """A population of `n` plain traders; each override is a scalar or a length-n list."""
    values = {**PLAIN, **overrides}
    assert set(values) == set(TRAIT_NAMES)
    arrays = {}
    for name in TRAIT_NAMES:
        dtype = np.int64 if name in ("n_clues", "cash") else float
        arrays[name] = np.broadcast_to(np.asarray(values[name], dtype=dtype), (n,)).copy()
    return Traits(**arrays)
