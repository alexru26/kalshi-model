"""Independent random streams for each part of the model.

Every part of a run draws only from its own stream, spawned from the run seed. Changing
one part (the network in E2, the mechanism in E1) then cannot shift the draws of
another, so the same seed gives the same world and population in every configuration
and paired comparisons stay valid.
"""

import numpy as np

# Each name maps to a fixed child index of the run seed. Append new streams at the end;
# reordering or removing a name would change every stream after it.
STREAM_NAMES = (
    "world",
    "deal",
    "traits",
    "network",
    "placement",
    "turns",
    "disinformation",
    "manipulator",
    "news",
    "decisions",
    "patience",
    "spread",
)


def make_streams(seed: int) -> dict[str, np.random.Generator]:
    """One independent generator per model part, all derived from `seed`."""
    children = np.random.SeedSequence(seed).spawn(len(STREAM_NAMES))
    return {
        name: np.random.default_rng(child)
        for name, child in zip(STREAM_NAMES, children, strict=True)
    }
