"""Check the frozen Config defaults against the Step 7 patterns on fresh seeds.

Usage: uv run python experiments/validate_defaults.py [first seed] [number of seeds]
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from kalshi_model.calibration import evaluate, measure, targets
from kalshi_model.config import Config
from kalshi_model.model import MarketModel


def run(seed: int) -> MarketModel:
    model = MarketModel(Config(), seed)
    model.run()
    return model


def main(first: int, n_seeds: int) -> None:
    with ProcessPoolExecutor() as pool:
        models = list(pool.map(run, range(first, first + n_seeds)))
    patterns = measure(models)
    result = evaluate(patterns, targets())
    print(f"seeds {first}..{first + n_seeds - 1}")
    print(json.dumps(patterns, indent=2))
    print(json.dumps({k: v for k, v in result.items()}, indent=2))
    # Reported, not required: how much more accurate the price is than the average
    # agent's own judgment, paired seed by seed.
    per_seed = [measure([m]) for m in models]
    gain = np.array([p["agent_brier"] - p["price_brier"] for p in per_seed])
    print(
        f"price Brier better than the average agent's own judgment by {gain.mean():.4f} "
        f"(se {gain.std(ddof=1) / np.sqrt(len(gain)):.4f}); price better in "
        f"{np.mean(gain > 0):.0%} of seeds"
    )


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(*(args + [2000, 48][len(args) :]))
