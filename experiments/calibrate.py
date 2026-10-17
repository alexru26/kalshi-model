"""Step 7: pattern-oriented calibration of the Kalshi default (PLAN.md Section 4).

Random search over the trait distributions and market-maker settings that have a data
anchor (not `g`, `w_hearsay`, or `w_depth`; informedness stays as set in Step 3). Each
candidate runs the default Kalshi market on the search seeds; candidates meeting every
pattern (`kalshi_model.calibration`) are ranked by distance to Kalshi's spread and trade
sizes, and the best few are re-run on fresh seeds before one is chosen. The check uses
many seeds because informed agents' P&L swings with luck: they win big in worlds where
their clues point the right way and lose big where the clues mislead. Every candidate is
written to `results/calibration.csv` with the git commit, so the accepted set can be
reused (Step 11 robustness).

Usage: uv run python experiments/calibrate.py [candidates] [search seeds] [check seeds]
"""

import csv
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from kalshi_model.calibration import evaluate, measure, targets
from kalshi_model.config import Config
from kalshi_model.model import MarketModel

RESULTS = Path(__file__).resolve().parents[1] / "results" / "calibration.csv"
SEARCH_RNG_SEED = 7
CHECK_SEED_OFFSET = 1000
FINALISTS = 5


def sample(rng: np.random.Generator) -> dict:
    """One candidate: log-uniform for scales, uniform otherwise."""

    def log_uniform(low: float, high: float) -> float:
        return float(np.exp(rng.uniform(np.log(low), np.log(high))))

    # Ranges narrowed after a first 160-candidate pass over wider ones (cash $100 to
    # $5,000, kelly mean 0.05 to 0.4, noise 0.03 to 0.3, min_edge 0.5 to 4c, patience mu
    # -2 to 2, market-maker size 5 to 100 and half-spread 0.5 to 4c): small cash could not
    # match Kalshi's trade sizes, and high patience kept spreads at 1c.
    kelly_mean = rng.uniform(0.1, 0.4)
    return {
        "cash_median": round(log_uniform(30_000, 500_000)),
        "cash_sigma": rng.uniform(0.8, 2.0),
        "kelly_alpha": 10 * kelly_mean,
        "kelly_beta": 10 * (1 - kelly_mean),
        "noise_scale": log_uniform(0.05, 0.3),
        "min_edge_scale": log_uniform(0.5, 3.0),
        "patience_mu": rng.uniform(-2.0, 1.0),
        "market_trust_scale": log_uniform(0.5, 5.0),
        "activity_sigma": rng.uniform(0.5, 1.5),
        "mm_size": round(log_uniform(5, 60)),
        "mm_half_spread": rng.uniform(0.5, 3.0),
    }


def run_patterns(job: tuple[dict, tuple[int, ...]]) -> dict:
    params, seeds = job
    config = replace(Config(), **params)
    models = []
    for seed in seeds:
        model = MarketModel(config, seed)
        model.run()
        models.append(model)
    return measure(models)


def flatten(params: dict, patterns: dict, result: dict, stage: str) -> dict:
    row = {"stage": stage, **params}
    row |= {k: v for k, v in patterns.items() if k != "sizes"}
    row |= {f"size_{q}": v for q, v in patterns["sizes"].items()}
    row |= {f"met_{k}": v for k, v in result["met"].items()}
    row |= {k: result[k] for k in ("accepted", "size_error", "spread_error", "distance")}
    return row


def main(n_candidates: int, n_search: int, n_check: int) -> None:
    target = targets()
    rng = np.random.default_rng(SEARCH_RNG_SEED)
    candidates = [sample(rng) for _ in range(n_candidates)]
    search_seeds = tuple(range(n_search))
    check_seeds = tuple(range(CHECK_SEED_OFFSET, CHECK_SEED_OFFSET + n_check))

    with ProcessPoolExecutor() as pool:
        searched = list(pool.map(run_patterns, [(c, search_seeds) for c in candidates]))
        rows = []
        for params, patterns in zip(candidates, searched, strict=True):
            rows.append(flatten(params, patterns, evaluate(patterns, target), "search"))
        accepted = sorted((r for r in rows if r["accepted"]), key=lambda r: r["distance"])[
            :FINALISTS
        ]
        finalists = [{k: r[k] for k in candidates[0]} for r in accepted]
        checked = list(pool.map(run_patterns, [(f, check_seeds) for f in finalists]))
    for params, patterns in zip(finalists, checked, strict=True):
        rows.append(flatten(params, patterns, evaluate(patterns, target), "check"))

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    RESULTS.parent.mkdir(exist_ok=True)
    with RESULTS.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["commit", *rows[0]])
        writer.writeheader()
        writer.writerows({"commit": commit, **row} for row in rows)

    n_accepted = sum(r["accepted"] for r in rows if r["stage"] == "search")
    print(
        f"{n_accepted} of {n_candidates} candidates met every required pattern on the search seeds"
    )
    print(f"targets: spread {target['spread']}c, sizes {target['sizes']}")
    for row in (r for r in rows if r["stage"] == "check"):
        sizes = [round(row[f"size_{q}"], 1) for q in ("q25", "q50", "q75", "q90")]
        print(
            f"check: accepted={row['accepted']} distance={row['distance']:.3f} "
            f"spread={row['spread']} sizes={sizes} "
            f"informed={row['informed_pnl']:.1f} uninformed={row['uninformed_pnl']:.1f}"
        )
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(*(args + [200, 8, 48][len(args) :]))
