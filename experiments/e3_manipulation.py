"""E3: manipulation. Honest twin vs pump and dump vs spoofing on shared seeds.

All three configurations add the same manipulator agent to the Kalshi default: the
twin trades honestly throughout, the others add their manipulation layer from
`manip_start`. Each seed runs all three together, so distortion
is measured step by step against the twin's price on the same seed (`metrics.
manipulation_metrics`), and the manipulation premium is the manipulator's P&L minus the
twin's. P&L is valued at final P* (no fakes in E3, so it has the outcome's expectation
with far less luck; `metrics.expected_pnl`).

Writes results/e3_manipulation.csv (one row per run) and .json (commit, seeds,
configs), then prints each prediction against the data. `summary`
reprints from the saved CSV without rerunning; it also compares the twin's market with
the Kalshi default of E1 (`results/e1_rules.csv`), which has no manipulator.

Usage: uv run python experiments/e3_manipulation.py [number of seeds | summary]
"""

import math
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from kalshi_model.config import Config
from kalshi_model.experiment import (
    RESULTS_DIR,
    SEEDS,
    bootstrap,
    git_commit,
    load,
    mean_ci,
    paired,
    write_results,
)
from kalshi_model.metrics import (
    AFTERMATH,
    COUNTERPARTIES,
    DISTORTION_LAGS,
    distortion_column,
    manipulation_metrics,
    run_metrics,
)
from kalshi_model.model import MarketModel

NAME = "e3_manipulation"
CONFIGS = {s: Config(manipulator=s) for s in ("twin", "pump", "spoof")}
STRATEGIES = ("pump", "spoof")


def run_seed(seed: int) -> list[dict]:
    """Run the twin and both manipulations on one seed with invariants checked."""
    models = {}
    for label, config in CONFIGS.items():
        models[label] = MarketModel(config, seed, debug=True)
        models[label].run()
    twin = models["twin"]
    return [
        {
            "config": label,
            "seed": seed,
            **run_metrics(model),
            **manipulation_metrics(model, None if label == "twin" else twin),
        }
        for label, model in models.items()
    ]


def run(seeds: range) -> None:
    commit = git_commit()
    with ProcessPoolExecutor() as pool:
        per_seed = list(pool.map(run_seed, seeds))
    rows = [row for label in CONFIGS for rows in per_seed for row in rows if row["config"] == label]
    write_results(NAME, rows, CONFIGS, seeds, commit)


def fmt(mean: float, half: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f} +- {half:.{digits}f}"


def fmt_boot(estimate: float, low: float, high: float, digits: int = 2) -> str:
    return f"{estimate:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def column(rows: list[dict], metric: str) -> np.ndarray:
    return np.array([r[metric] for r in rows], dtype=float)


def distortion_curve(rows: list[dict]) -> np.ndarray:
    """Mean distortion (c) over seeds at each lag after `manip_start`."""
    curves = np.array([[r[distortion_column(k)] for k in DISTORTION_LAGS] for r in rows])
    return np.nanmean(curves, axis=0)


def peak(rows: list[dict]) -> float:
    """Peak of the mean distortion curve, in cents."""
    return float(np.max(distortion_curve(rows)))


def fall_lag(rows: list[dict]) -> float:
    """Steps after `manip_start` until the mean distortion curve, past its peak, first
    falls below half the peak (linear interpolation); NaN if it never does."""
    curve = distortion_curve(rows)
    top = int(np.argmax(curve))
    half = curve[top] / 2
    if not half > 0:
        return math.nan
    for i in range(top + 1, len(curve)):
        if curve[i] < half:
            lo, hi = DISTORTION_LAGS[i - 1], DISTORTION_LAGS[i]
            return float(lo + (curve[i - 1] - half) / (curve[i - 1] - curve[i]) * (hi - lo))
    return math.nan


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return math.nan
    rx, ry = (np.argsort(np.argsort(v[keep])) for v in (x, y))
    return float(np.corrcoef(rx, ry)[0, 1])


def per_config_table(results: dict) -> None:
    metrics = [
        ("error", "mean |price - P*| (c)"),
        ("brier", "final Brier"),
        ("volume", "volume (contracts)"),
        ("fees", "fees ($)"),
        ("maker_pnl", "market maker P&L ($)"),
        ("manip_pnl", "manipulator P&L ($)"),
        ("manip_pnl_at_p_star", "same at P* ($)"),
        ("depth_before", "depth before start"),
        ("layer_steps", "steps the layer acted"),
        ("bought", "contracts it bought (layer)"),
        ("sold", "contracts it sold (layer)"),
    ]
    print(f"{'metric':30s}" + "".join(f"{label:>24s}" for label in CONFIGS))
    for metric, name in metrics:
        cells = []
        for label in CONFIGS:
            m, h = mean_ci([row[metric] for row in results[label].values()])
            cells.append(f"{fmt(m, h):>24s}" if np.isfinite(m) else f"{'-':>24s}")
        print(f"{name:30s}" + "".join(cells))


def twin_vs_default(results: dict) -> None:
    """How much the twin's presence alone changes the Kalshi default (E1, same seeds)."""
    if not (RESULTS_DIR / "e1_rules.csv").exists():
        return
    e1 = load("e1_rules")
    if "kalshi" not in e1:
        return
    both = {"default": e1["kalshi"], "twin": results["twin"]}
    print("\nTwin - Kalshi default without a manipulator (E1), paired")
    for metric in ("error", "brier", "volume", "pnl_top_at_p_star", "pnl_none_at_p_star"):
        print(f"  {metric:22s} {fmt(*paired(both, metric, 'default', 'twin'), 3)}")


def counterparties(rows: list[dict], kind: str) -> str:
    """Each counterparty's share of the contracts and its by-chance share, pooled."""
    total = sum(column(rows, f"{kind}_{c}").sum() for c in COUNTERPARTIES)
    if not total:
        return "none traded"
    cells = []
    for c in COUNTERPARTIES:
        share = column(rows, f"{kind}_{c}").sum() / total
        weights = sum(column(rows, f"{kind}_{d}") for d in COUNTERPARTIES)
        chance = np.nansum(weights * column(rows, f"share_{c}")) / weights.sum()
        cells.append(f"{c} {share:.2f} ({chance:.2f})")
    return ", ".join(cells)


def share_gap(rows: list[dict], kind: str, tier: str) -> tuple[float, float]:
    """Per seed, `tier`'s share of the contracts of `kind` minus its by-chance share."""
    gaps = []
    for r in rows:
        total = sum(r[f"{kind}_{c}"] for c in COUNTERPARTIES)
        if total:
            gaps.append(r[f"{kind}_{tier}"] / total - r[f"share_{tier}"])
    return mean_ci(gaps)


def predictions(results: dict) -> None:
    for strategy, title in (
        ("pump", "E3.1 pump and dump moves the price but loses money"),
        ("spoof", "E3.2 spoofing lifts the price and profits, through agents reading depth"),
    ):
        rows = list(results[strategy].values())
        print(f"\n{title}")
        print(f"  distortion while acting (c):   {fmt(*mean_ci(column(rows, 'distortion')))}")
        after = fmt(*mean_ci(column(rows, "distortion_after")))
        print(f"  distortion {AFTERMATH} steps after (c): {after}")
        print(f"  premium at P* ($):             {fmt(*mean_ci(column(rows, 'premium')), 0)}")
        print(
            f"  premium realized ($):          {fmt(*mean_ci(column(rows, 'premium_realized')), 0)}"
        )
        print(f"  peak of mean distortion (c):   {fmt_boot(*bootstrap(results, [strategy], peak))}")
        print(
            "  steps until below half peak:   "
            f"{fmt_boot(*bootstrap(results, [strategy], fall_lag), 0)}"
        )
        curve = distortion_curve(rows)
        print(
            "  mean distortion by lag:        "
            + " ".join(f"{k}:{v:.2f}" for k, v in zip(DISTORTION_LAGS, curve, strict=True))
        )
        for metric, name in (
            ("taken_from_agents", "taken from agents ($)"),
            ("taken_from_mm", "taken from market maker ($)"),
            ("fees_change", "change in fees ($)"),
        ):
            print(f"  {name:31s}{fmt(*mean_ci(column(rows, metric)), 0)}")
        print(f"  sold to it, share (by chance): {counterparties(rows, 'sold_to')}")
        print(f"  bought from it:                {counterparties(rows, 'bought_from')}")
        for kind in ("sold_to", "bought_from"):
            print(
                f"  top tenth's {kind} share - chance, per seed: "
                f"{fmt(*share_gap(rows, kind, 'top'), 3)}"
            )
        gap = column(rows, "buyer_w_depth") - column(rows, "buyer_w_depth_chance")
        print(
            f"  buyers' w_depth: {fmt(*mean_ci(column(rows, 'buyer_w_depth')), 3)}, "
            f"by chance {fmt(*mean_ci(column(rows, 'buyer_w_depth_chance')), 3)}, "
            f"gap {fmt(*mean_ci(gap), 3)}"
        )
        if strategy == "pump":
            print(f"  share reaching the target:     {np.mean(column(rows, 'pump_reached')):.2f}")
        else:
            fake = column(rows, "fake_filled")
            print(
                f"  fake-bid contracts filled:     {fmt(*mean_ci(fake), 0)} "
                f"(runs with any: {np.mean(fake > 0):.2f})"
            )

    print("\nE3.3 more liquid runs distort less and for less time")
    print("  Spearman correlation across seeds of depth before the start (at the last step,")
    print("  and its mean over the 100 steps before) with distortion and the premium")
    for depth in ("depth_before", "depth_before_mean"):
        for strategy in STRATEGIES:
            for metric in ("distortion", "distortion_after", "premium"):

                def stat(rows: list[dict], depth: str = depth, metric: str = metric) -> float:
                    return spearman(column(rows, depth), column(rows, metric))

                corr = fmt_boot(*bootstrap(results, [strategy], stat))
                print(f"  {depth:18s} {strategy:6s} {metric:18s} {corr}")


def main(argv: list[str]) -> None:
    if argv[:1] != ["summary"]:
        n_seeds = int(argv[0]) if argv else len(SEEDS)
        run(range(SEEDS.start, SEEDS.start + n_seeds))
    results = load(NAME)
    n = len(next(iter(results.values())))
    print(f"E3: {n} seeds per configuration, invariants checked every step\n")
    per_config_table(results)
    twin_vs_default(results)
    predictions(results)


if __name__ == "__main__":
    main(sys.argv[1:])
