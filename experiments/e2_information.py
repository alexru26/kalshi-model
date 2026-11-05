"""E2: information spread. A disinformation sweep on the Holme-Kim network.

Ratios 0, 0.05, 0.1, 0.2, 0.5 (0, 1, 2, 4, 10 fakes). Distortion is the signed gap to
P* in the fakes' direction minus the same seed's gap without disinformation (ratio 0).

Writes results/e2_information.csv (one row per run) and .json (commit, seeds,
configs), then prints each prediction against the data. `summary`
reprints from the saved CSV without rerunning.

Usage: uv run python experiments/e2_information.py [number of seeds | summary]
"""

import sys
from itertools import pairwise

from kalshi_model.config import Config
from kalshi_model.experiment import (
    SEEDS,
    bootstrap,
    curve_half_life,
    load,
    mean_ci,
    paired,
    run_experiment,
)

NAME = "e2_information"
RATIOS = (0.0, 0.05, 0.1, 0.2, 0.5)
DEGREE_RATIO = 0.1
"""Ratio at which E2.3 compares well- and poorly-connected agents' realized P&L."""


def label(ratio: float) -> str:
    return f"ratio {ratio}"


CONFIGS = {label(r): Config(disinfo_ratio=r) for r in RATIOS}


def fmt(mean: float, half: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f} +- {half:.{digits}f}"


def fmt_boot(estimate: float, low: float, high: float, digits: int = 1) -> str:
    return f"{estimate:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def per_seed(results: dict, ratio: float, metric: str = "signed_gap") -> dict:
    """Paired distortion per seed: the metric with fakes minus the same seed's without."""
    clean, dirty = results[label(0.0)], results[label(ratio)]
    return {s: dirty[s][metric] - clean[s][metric] for s in sorted(set(clean) & set(dirty))}


def ratio_table(results: dict) -> None:
    print("Disinformation sweep on Holme-Kim")
    head = (
        f"{'ratio':>6s}{'fakes':>7s}{'steps to half':>16s}{'news half-life':>22s}"
        f"{'distortion (c)':>16s}{'last quarter':>16s}{'error (c)':>16s}{'Brier':>16s}"
    )
    print(head)
    for ratio in RATIOS:
        lab = label(ratio)
        rows = results[lab].values()
        if ratio:
            dist = fmt(*paired(results, "signed_gap", label(0.0), lab))
            last = fmt(*paired(results, "signed_gap_last_quarter", label(0.0), lab))
        else:
            dist = last = "-"
        print(
            f"{ratio:6.2f}{Config(disinfo_ratio=ratio).n_fakes:7d}"
            f"{fmt(*mean_ci([r['half_reach'] for r in rows]), 1):>16s}"
            f"{fmt_boot(*bootstrap(results, [lab], curve_half_life)):>22s}"
            f"{dist:>16s}{last:>16s}"
            f"{fmt(*mean_ci([r['error'] for r in rows])):>16s}"
            f"{fmt(*mean_ci([r['brier'] for r in rows]), 3):>16s}"
        )


def predictions(results: dict) -> None:
    print(f"\nE2.3 top vs bottom quarter by degree (ratio {DEGREE_RATIO})")
    rows = results[label(DEGREE_RATIO)].values()
    delay = [r["learn_delay_top_degree"] - r["learn_delay_bottom_degree"] for r in rows]
    pnl = [r["pnl_top_degree"] - r["pnl_bottom_degree"] for r in rows]
    print(f"  learn delay top - bottom (steps): {fmt(*mean_ci(delay), 1)}")
    print(f"  realized P&L top - bottom ($):    {fmt(*mean_ci(pnl), 1)}")
    # Without fakes, P&L can be valued at P*: the same expectation without outcome luck.
    rows = results[label(0.0)].values()
    pnl = [r["pnl_top_degree_at_p_star"] - r["pnl_bottom_degree_at_p_star"] for r in rows]
    print(f"  ratio 0, P&L top - bottom at P* ($): {fmt(*mean_ci(pnl), 1)}")

    print("\nE2.5 distortion per fake (c)")
    per_fake = {}
    for ratio in RATIOS[1:]:
        n_fakes = Config(disinfo_ratio=ratio).n_fakes
        d = per_seed(results, ratio)
        per_fake[ratio] = {s: v / n_fakes for s, v in d.items()}
        print(
            f"  ratio {ratio:4.2f} ({n_fakes:2d} fakes): distortion "
            f"{fmt(*mean_ci(list(d.values())))}, "
            f"per fake {fmt(*mean_ci(list(per_fake[ratio].values())))}"
        )
    for low, high in pairwise(RATIOS[1:]):
        gaps = [per_fake[high][s] - per_fake[low][s] for s in per_fake[low]]
        print(f"  per fake at {high} - at {low}: {fmt(*mean_ci(gaps))}")


def main(argv: list[str]) -> None:
    if argv[:1] != ["summary"]:
        n_seeds = int(argv[0]) if argv else len(SEEDS)
        run_experiment(NAME, CONFIGS, range(SEEDS.start, SEEDS.start + n_seeds))
    results = load(NAME)
    n = len(next(iter(results.values())))
    print(f"E2: {len(results)} configurations x {n} seeds, invariants checked every step\n")
    ratio_table(results)
    predictions(results)


if __name__ == "__main__":
    main(sys.argv[1:])
