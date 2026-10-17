"""Step 6 check: news spread speed by network and price distortion from disinformation.

Usage: uv run python experiments/information_check.py [number of seeds]
"""

import sys
from dataclasses import replace

import numpy as np

from kalshi_model.config import NETWORKS, Config
from kalshi_model.model import MarketModel


def run(config: Config, seed: int) -> MarketModel:
    model = MarketModel(config, seed)
    model.run()
    return model


def half_reach(model: MarketModel) -> float:
    """Mean steps from release until half the agents know a real news clue (inf if never)."""
    news = [c for c in model.release_step if c not in model.fake_ids]
    return float(
        np.mean([model.half_reach_step.get(c, np.inf) - model.release_step[c] for c in news])
    )


def distortion(model: MarketModel) -> float:
    """Mean signed gap between price and P*, positive in the fakes' direction."""
    df = model.datacollector.get_model_vars_dataframe()
    sign = -1 if model.world.y else 1  # fakes point against the truth
    return float(sign * (df.price - df.p_star).mean())


def main(n_seeds: int) -> None:
    seeds = range(n_seeds)
    base = Config(disinfo_ratio=0.1)
    print("network (disinfo ratio 0.1)  steps to half  share never  distortion (c)")
    for network in NETWORKS:
        models = [run(replace(base, network=network), s) for s in seeds]
        reach = np.array([half_reach(m) for m in models])
        never = np.mean(~np.isfinite(reach))
        finite = reach[np.isfinite(reach)]
        print(
            f"{network:28s} {finite.mean():13.1f}  {never:11.2f}  "
            f"{100 * np.mean([distortion(m) for m in models]):14.2f}"
        )
    print("\ndisinfo ratio (holme_kim)  fakes  distortion (c)")
    for ratio in (0.0, 0.05, 0.1, 0.2):
        config = Config(disinfo_ratio=ratio)
        models = [run(config, s) for s in seeds]
        print(
            f"{ratio:25.2f}  {config.n_fakes:5d}  "
            f"{100 * np.mean([distortion(m) for m in models]):14.2f}"
        )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
