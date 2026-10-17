"""Step 7: pattern-oriented calibration of the Kalshi default (PLAN.md Section 4).

Pattern-oriented modeling looks for the region of settings consistent with the data, not
one best point (Grimm et al. 2005), so this is a rejection search:

1. Search: a Latin hypercube of candidates over `calibration.SPACE` (even coverage of
   every setting's range), each run on the search seeds; a candidate is accepted if it
   meets every required pattern (`kalshi_model.calibration`).
2. Select: the accepted candidates nearest the accepted region's median (central), and
   a spread-out set chosen by farthest-point sampling (the ensemble for Step 11).
3. Check: rerun those on fresh seeds. Many seeds are needed because informed agents'
   P&L swings with luck: they win big in worlds where their clues point the right way
   and lose big where the clues mislead.

The most central candidate that passes the check is the one to freeze in `Config`; the
spread-out ones that pass form the robustness ensemble. Everything is written to
`results/calibration.csv` with the git commit.

Usage: uv run python experiments/calibrate.py [candidates] [search seeds] [check seeds]
"""

import csv
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from kalshi_model.calibration import (
    SPACE,
    central_order,
    config_overrides,
    evaluate,
    latin_hypercube,
    measure,
    spread_out,
    targets,
)
from kalshi_model.config import Config
from kalshi_model.model import MarketModel

RESULTS = Path(__file__).resolve().parents[1] / "results" / "calibration.csv"
SEARCH_RNG_SEED = 7
CHECK_SEED_OFFSET = 1000
N_CENTRAL = 5
N_ENSEMBLE = 10


def run_patterns(job: tuple[dict, tuple[int, ...]]) -> dict:
    overrides, seeds = job
    config = replace(Config(), **overrides)
    models = []
    for seed in seeds:
        model = MarketModel(config, seed)
        model.run()
        models.append(model)
    return measure(models)


def row_for(index: int, u: np.ndarray, patterns: dict, result: dict, stage: str) -> dict:
    row = {"stage": stage, "candidate": index}
    row |= {f"u_{d.name}": float(x) for d, x in zip(SPACE, u, strict=True)}
    row |= config_overrides(u)
    row |= {k: v for k, v in patterns.items() if k != "sizes"}
    row |= {f"size_{q}": v for q, v in patterns["sizes"].items()}
    row |= {f"met_{k}": v for k, v in result["met"].items()}
    row |= {k: result[k] for k in ("accepted", "size_error", "spread_error", "distance")}
    return row


def main(n_candidates: int, n_search: int, n_check: int) -> None:
    target = targets()
    points = latin_hypercube(n_candidates, len(SPACE), np.random.default_rng(SEARCH_RNG_SEED))
    search_seeds = tuple(range(n_search))
    check_seeds = tuple(range(CHECK_SEED_OFFSET, CHECK_SEED_OFFSET + n_check))

    with ProcessPoolExecutor() as pool:
        jobs = [(config_overrides(u), search_seeds) for u in points]
        searched = list(pool.map(run_patterns, jobs, chunksize=4))
        rows = [
            row_for(i, u, p, evaluate(p, target), "search")
            for i, (u, p) in enumerate(zip(points, searched, strict=True))
        ]
        accepted = np.array([i for i, r in enumerate(rows) if r["accepted"]])
        if accepted.size == 0:
            raise SystemExit("no candidate met every required pattern; widen the ranges")
        order = accepted[central_order(points[accepted])]
        central = [int(i) for i in order[:N_CENTRAL]]
        ensemble = [
            int(accepted[j])
            for j in spread_out(
                points[accepted], int(np.flatnonzero(accepted == order[0])[0]), N_ENSEMBLE
            )
        ]
        to_check = list(dict.fromkeys(central + ensemble))
        checked = list(
            pool.map(run_patterns, [(config_overrides(points[i]), check_seeds) for i in to_check])
        )

    frozen = None
    for i, patterns in zip(to_check, checked, strict=True):
        result = evaluate(patterns, target)
        row = row_for(i, points[i], patterns, result, "check")
        row["central_rank"] = central.index(i) if i in central else ""
        row["in_ensemble"] = i in ensemble and result["accepted"]
        if frozen is None and i in central and result["accepted"]:
            frozen = i
        rows.append(row)
    for row in rows:
        row["frozen"] = row["stage"] == "check" and row["candidate"] == frozen

    commit = subprocess.run(
        ["git", "describe", "--always", "--dirty", "--abbrev=40"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()  # ends in "-dirty" if the code had uncommitted changes
    RESULTS.parent.mkdir(exist_ok=True)
    fields = ["commit", *dict.fromkeys(k for row in rows for k in row)]
    with RESULTS.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows({"commit": commit, **row} for row in rows)

    print(f"{accepted.size} of {n_candidates} candidates met every required pattern")
    print(f"targets: spread {target['spread']}c, sizes {target['sizes']}")
    for row in (r for r in rows if r["stage"] == "check"):
        sizes = [round(row[f"size_{q}"], 1) for q in ("q25", "q50", "q75", "q90")]
        role = "frozen" if row["frozen"] else "ensemble" if row["in_ensemble"] else "checked"
        print(
            f"{role:8s} #{row['candidate']:4d} accepted={row['accepted']!s:5s} "
            f"spread={row['spread']} empty={row['empty_share']:.3f} sizes={sizes} "
            f"informed={row['informed_pnl']:.1f} uninformed={row['uninformed_pnl']:.1f}"
        )
    print(f"frozen candidate: {frozen}; wrote {RESULTS}")


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(*(args + [1000, 8, 48][len(args) :]))
