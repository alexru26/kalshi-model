"""Check the frozen Config defaults against the Step 7 patterns on fresh seeds.

Usage: uv run python experiments/validate_defaults.py [first seed] [number of seeds]
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor

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


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(*(args + [2000, 48][len(args) :]))
