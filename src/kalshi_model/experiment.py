"""Running an experiment and comparing its configurations.

Every configuration runs on the same seeds with the conservation invariants checked
every step, and each run is one row of the results CSV. A JSON file beside the CSV
records the git commit, the seeds, and every configuration in full. Comparisons are
paired differences on shared seeds with 95% confidence intervals; statistics of the
mean news-response curve (its half-life) use a bootstrap that resamples seeds.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from kalshi_model.config import Config
from kalshi_model.metrics import RESPONSE_LAGS, half_life, response_column, run_metrics
from kalshi_model.model import MarketModel

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
SEEDS = range(20000, 20100)
"""The experiment seeds: 100, none used by calibration or earlier checks."""
Z95 = 1.96
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 8


def run_one(job: tuple[str, Config, int]) -> dict:
    """Run one configuration on one seed with invariants checked; return its row."""
    label, config, seed = job
    model = MarketModel(config, seed, debug=True)
    model.run()
    return {"config": label, "seed": seed, **run_metrics(model)}


def git_commit() -> str:
    """The commit the code ran at, ending in "-dirty" if files outside results/ had
    uncommitted changes (rewritten results do not make the code dirty)."""

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False, cwd=RESULTS_DIR.parent
        )

    commit = git("rev-parse", "HEAD").stdout.strip()
    dirty = git("diff", "--quiet", "HEAD", "--", ".", f":(exclude){RESULTS_DIR.name}").returncode
    return f"{commit}-dirty" if dirty else commit


def run_experiment(
    name: str,
    configs: dict[str, Config],
    seeds: range = SEEDS,
    results_dir: Path = RESULTS_DIR,
    workers: int | None = None,
) -> list[dict]:
    """Run every configuration on every seed; write results/<name>.csv and .json."""
    commit = git_commit()
    jobs = [(label, config, seed) for label, config in configs.items() for seed in seeds]
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(run_one, jobs, chunksize=2))
    write_results(name, rows, configs, seeds, commit, results_dir)
    return rows


def write_results(
    name: str,
    rows: list[dict],
    configs: dict[str, Config],
    seeds: range,
    commit: str,
    results_dir: Path = RESULTS_DIR,
) -> None:
    """Write the rows to results/<name>.csv, and to results/<name>.json the commit the
    code ran at (`git_commit()`, taken before the runs), the seeds, and the configurations."""
    results_dir.mkdir(exist_ok=True)
    with (results_dir / f"{name}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    meta = {
        "commit": commit,
        "seeds": [seeds.start, seeds.stop],
        "configs": {label: asdict(config) for label, config in configs.items()},
    }
    (results_dir / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n")


def load(name: str, results_dir: Path = RESULTS_DIR) -> dict[str, dict[int, dict]]:
    """Read results/<name>.csv as config label -> seed -> row of floats."""
    out: dict[str, dict[int, dict]] = {}
    with (results_dir / f"{name}.csv").open() as f:
        for row in csv.DictReader(f):
            label, seed = row.pop("config"), int(row.pop("seed"))
            out.setdefault(label, {})[seed] = {k: float(v) for k, v in row.items()}
    return out


def mean_ci(values) -> tuple[float, float]:
    """Mean of the finite values and the half-width of its 95% interval."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    if values.size == 1:
        return float(values[0]), math.nan
    return float(values.mean()), float(Z95 * values.std(ddof=1) / math.sqrt(values.size))


def paired(results: dict, metric: str, a: str, b: str) -> tuple[float, float]:
    """Mean and 95% half-width of (b - a) in `metric`, over the seeds both ran."""
    seeds = sorted(set(results[a]) & set(results[b]))
    return mean_ci([results[b][s][metric] - results[a][s][metric] for s in seeds])


def values(results: dict, label: str, metric: str) -> np.ndarray:
    """One configuration's `metric` on every seed, in seed order."""
    return np.array([row[metric] for _, row in sorted(results[label].items())])


def response_curve(rows: list[dict]) -> np.ndarray:
    """Mean news-response curve over runs (runs with no news events are skipped)."""
    curves = np.array([[row[response_column(k)] for k in RESPONSE_LAGS] for row in rows])
    return curves[np.isfinite(curves).all(axis=1)].mean(axis=0)


def curve_half_life(rows: list[dict]) -> float:
    """Half-life of the mean news-response curve over runs."""
    return half_life(response_curve(rows))


def mean_of(metric: str) -> Callable[[list[dict]], float]:
    """A statistic: the mean of `metric` over runs."""
    return lambda rows: float(np.nanmean([row[metric] for row in rows]))


def bootstrap(
    results: dict, labels: list[str], statistic: Callable[..., float]
) -> tuple[float, float, float]:
    """`statistic(*rows per label)` and its 95% bootstrap interval over seeds.

    Each resample draws the same seeds for every label, so the interval of a
    comparison between labels stays paired.
    """
    seeds = sorted(set.intersection(*(set(results[label]) for label in labels)))

    def at(sample: list[int]) -> float:
        return statistic(*([results[label][s] for s in sample] for label in labels))

    estimate = at(seeds)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.array(
        [at(list(rng.choice(seeds, size=len(seeds)))) for _ in range(BOOTSTRAP_SAMPLES)]
    )
    draws = draws[np.isfinite(draws)]
    if draws.size < BOOTSTRAP_SAMPLES / 2:
        return estimate, math.nan, math.nan
    low, high = np.quantile(draws, [0.025, 0.975])
    return estimate, float(low), float(high)
