import json
import math
from dataclasses import asdict

import pytest

from kalshi_model.config import amm_config, clob_config
from kalshi_model.experiment import bootstrap, load, mean_ci, mean_of, paired, run_experiment


def test_mean_ci_is_the_normal_95_interval_of_the_finite_values():
    mean, half = mean_ci([1.0, 2.0, 3.0, math.nan])
    assert mean == 2.0
    assert half == pytest.approx(1.96 * 1.0 / math.sqrt(3))
    assert math.isnan(mean_ci([math.nan])[0])


def test_paired_differences_use_only_shared_seeds():
    results = {
        "a": {1: {"x": 1.0}, 2: {"x": 2.0}, 3: {"x": 10.0}},
        "b": {1: {"x": 2.0}, 2: {"x": 4.0}},
    }
    mean, half = paired(results, "x", "a", "b")
    assert mean == 1.5  # (2 - 1 + 4 - 2) / 2; seed 3 ran only in "a"
    assert half == pytest.approx(1.96 * math.sqrt(0.5) / math.sqrt(2))


def test_bootstrap_resamples_the_same_seeds_for_every_label():
    results = {"a": {s: {"x": float(s)} for s in range(30)}}
    results["b"] = {s: {"x": float(s) + 1} for s in range(30)}
    # b - a is 1 on every seed, so a paired bootstrap has no spread at all.
    estimate, low, high = bootstrap(
        results, ["a", "b"], lambda a, b: mean_of("x")(b) - mean_of("x")(a)
    )
    assert (estimate, low, high) == pytest.approx((1.0, 1.0, 1.0))


def test_run_experiment_writes_one_row_per_run_and_the_full_configs(tmp_path):
    configs = {
        "amm": amm_config(n_steps=120, n_clues_total=62),
        "clob": clob_config(n_steps=120, n_clues_total=62),
    }
    rows = run_experiment("tiny", configs, range(3, 5), tmp_path, workers=2)
    assert [(r["config"], r["seed"]) for r in rows] == [
        ("amm", 3),
        ("amm", 4),
        ("clob", 3),
        ("clob", 4),
    ]
    results = load("tiny", tmp_path)
    assert set(results) == {"amm", "clob"} and set(results["amm"]) == {3, 4}
    assert results["clob"][4]["error"] == pytest.approx(rows[3]["error"])
    meta = json.loads((tmp_path / "tiny.json").read_text())
    assert meta["seeds"] == [3, 5]
    assert meta["configs"] == {label: asdict(c) for label, c in configs.items()}
    assert meta["commit"]
