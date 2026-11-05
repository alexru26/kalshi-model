"""Check news spread speed and price distortion from disinformation.

Usage: uv run python experiments/information_check.py [number of seeds]
"""

import sys
from functools import cache

import numpy as np

from kalshi_model.config import Config
from kalshi_model.model import MarketModel

RATIOS = (0.0, 0.05, 0.1, 0.2, 0.5)


@cache
def run(ratio: float, seed: int) -> MarketModel:
    model = MarketModel(Config(disinfo_ratio=ratio), seed)
    model.run()
    return model


def half_reach(model: MarketModel) -> list[float]:
    """Steps from release until half the agents know each real news clue (inf if never)."""
    news = [c for c in model.release_step if c not in model.fake_ids]
    return [model.half_reach_step.get(c, np.inf) - model.release_step[c] for c in news]


def gap(model: MarketModel) -> float:
    """Mean signed gap between price and P*, positive in the fakes' direction."""
    df = model.datacollector.get_model_vars_dataframe()
    sign = -1 if model.world.y else 1  # fakes point against the truth
    return float(sign * (df.price - df.p_star).mean())


def distortion(ratio: float, seeds: range) -> float:
    """Mean gap minus the same seed's gap without disinformation, in cents."""
    return 100 * float(np.mean([gap(run(ratio, s)) - gap(run(0.0, s)) for s in seeds]))


def main(n_seeds: int) -> None:
    seeds = range(n_seeds)
    reach = np.array([r for s in seeds for r in half_reach(run(0.1, s))])
    finite = reach[np.isfinite(reach)]
    steps = finite.mean() if finite.size else np.inf
    print("news spread at disinfo ratio 0.1: steps to half reach, share never reaching half")
    print(f"{steps:13.1f}  {1 - finite.size / reach.size:11.2f}")
    print("\ndisinfo ratio  fakes  distortion (c)")
    for ratio in RATIOS:
        n_fakes = Config(disinfo_ratio=ratio).n_fakes
        print(f"{ratio:13.2f}  {n_fakes:5d}  {distortion(ratio, seeds):14.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
