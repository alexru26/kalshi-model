"""E2: information spread. Network and disinformation sweeps around the default.

Network sweep: the 5 networks at disinformation ratio 0.1. Disinformation sweep:
ratios 0, 0.05, 0.1, 0.2 on Holme-Kim (the default network). Distortion is the signed
gap to P* in the fakes' direction minus the same seed's gap in the same network
without disinformation, so every network also runs at ratio 0 (PLAN.md Section 3).

Writes results/e2_information.csv (one row per run) and .json (commit, seeds,
configs), then prints each prediction in PREDICTIONS.md against the data. `summary`
reprints from the saved CSV without rerunning.

Usage: uv run python experiments/e2_information.py [number of seeds | summary]
"""

import sys

from kalshi_model.config import NETWORKS, Config
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
SWEEP_RATIO = 0.1
RATIOS = (0.0, 0.05, 0.1, 0.2)
DEFAULT_NETWORK = Config().network


def label(network: str, ratio: float) -> str:
    return f"{network}@{ratio}"


CONFIGS = {
    label(network, ratio): Config(network=network, disinfo_ratio=ratio)
    for network in NETWORKS
    for ratio in (0.0, SWEEP_RATIO)
} | {label(DEFAULT_NETWORK, r): Config(disinfo_ratio=r) for r in RATIOS}


def fmt(mean: float, half: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f} +- {half:.{digits}f}"


def fmt_boot(estimate: float, low: float, high: float, digits: int = 1) -> str:
    return f"{estimate:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def distortion(results: dict, network: str, ratio: float, metric: str = "signed_gap"):
    """Paired distortion in cents: gap with fakes minus the same seed's gap without."""
    return paired(results, metric, label(network, 0.0), label(network, ratio))


def per_seed(results: dict, network: str, ratio: float, metric: str = "signed_gap") -> dict:
    clean, dirty = results[label(network, 0.0)], results[label(network, ratio)]
    return {s: dirty[s][metric] - clean[s][metric] for s in sorted(set(clean) & set(dirty))}


def network_table(results: dict) -> None:
    print(f"Network sweep at disinformation ratio {SWEEP_RATIO}")
    head = (
        f"{'network':16s}{'steps to half':>16s}{'news half-life':>22s}"
        f"{'distortion (c)':>16s}{'last quarter':>16s}{'error (c)':>16s}{'Brier':>16s}"
    )
    print(head)
    for network in NETWORKS:
        lab = label(network, SWEEP_RATIO)
        rows = results[lab].values()
        print(
            f"{network:16s}"
            f"{fmt(*mean_ci([r['half_reach'] for r in rows]), 1):>16s}"
            f"{fmt_boot(*bootstrap(results, [lab], curve_half_life)):>22s}"
            f"{fmt(*distortion(results, network, SWEEP_RATIO)):>16s}"
            f"{fmt(*distortion(results, network, SWEEP_RATIO, 'signed_gap_last_quarter')):>16s}"
            f"{fmt(*mean_ci([r['error'] for r in rows])):>16s}"
            f"{fmt(*mean_ci([r['brier'] for r in rows]), 3):>16s}"
        )
    print("\nWithout disinformation (ratio 0)")
    for network in NETWORKS:
        lab = label(network, 0.0)
        rows = results[lab].values()
        print(
            f"{network:16s}"
            f"{fmt(*mean_ci([r['half_reach'] for r in rows]), 1):>16s}"
            f"{fmt_boot(*bootstrap(results, [lab], curve_half_life)):>22s}"
            f"{fmt(*mean_ci([r['error'] for r in rows])):>16s}"
            f"{fmt(*mean_ci([r['brier'] for r in rows]), 3):>16s}"
        )


def predictions(results: dict) -> None:
    sweep = {n: label(n, SWEEP_RATIO) for n in NETWORKS}
    ws = "watts_strogatz"

    print("\nE2.1 Watts-Strogatz spreads news slowest (WS - other, steps to half reach)")
    for network in NETWORKS:
        if network != ws:
            print(
                f"  vs {network:16s} "
                f"{fmt(*paired(results, 'half_reach', sweep[network], sweep[ws]), 1)}"
            )

    print("\nE2.2 slower spread means slower pricing (WS - other, news half-life, steps)")
    for network in NETWORKS:
        if network != ws:
            diff = bootstrap(
                results,
                [sweep[network], sweep[ws]],
                lambda o, w: curve_half_life(w) - curve_half_life(o),
            )
            print(f"  vs {network:16s} {fmt_boot(*diff)}")

    print("\nE2.3 top vs bottom quarter by degree (ratio 0.1)")
    print(f"  {'network':16s}{'learn delay top-bottom':>26s}{'P&L top-bottom ($)':>24s}")
    for network in NETWORKS:
        rows = results[sweep[network]].values()
        delay = [r["learn_delay_top_degree"] - r["learn_delay_bottom_degree"] for r in rows]
        pnl = [r["pnl_top_degree"] - r["pnl_bottom_degree"] for r in rows]
        print(f"  {network:16s}{fmt(*mean_ci(delay), 1):>26s}{fmt(*mean_ci(pnl), 1):>24s}")
    every = [r for n in NETWORKS for r in results[sweep[n]].values()]
    pnl = [r["pnl_top_degree"] - r["pnl_bottom_degree"] for r in every]
    print(f"  {'all five':16s}{'':>26s}{fmt(*mean_ci(pnl), 1):>24s}")
    # Without fakes, P&L can be valued at P*: the same expectation without outcome luck.
    print("  ratio 0, P&L top - bottom valued at P* ($)")
    for network in NETWORKS:
        rows = results[label(network, 0.0)].values()
        pnl = [r["pnl_top_degree_at_p_star"] - r["pnl_bottom_degree_at_p_star"] for r in rows]
        print(f"  {network:16s}{fmt(*mean_ci(pnl), 1):>50s}")
    every = [r for n in NETWORKS for r in results[label(n, 0.0)].values()]
    pnl = [r["pnl_top_degree_at_p_star"] - r["pnl_bottom_degree_at_p_star"] for r in every]
    print(f"  {'all five':16s}{fmt(*mean_ci(pnl), 1):>50s}")

    print("\nE2.4 SBM - other network, distortion (c), whole run and last quarter")
    for metric in ("signed_gap", "signed_gap_last_quarter"):
        sbm = per_seed(results, "sbm", SWEEP_RATIO, metric)
        for network in NETWORKS:
            if network == "sbm":
                continue
            other = per_seed(results, network, SWEEP_RATIO, metric)
            print(
                f"  {metric:24s} vs {network:16s} {fmt(*mean_ci([sbm[s] - other[s] for s in sbm]))}"
            )

    print("\nE2.5 distortion per fake on Holme-Kim (c)")
    per_fake = {}
    for ratio in RATIOS[1:]:
        n_fakes = Config(disinfo_ratio=ratio).n_fakes
        d = per_seed(results, DEFAULT_NETWORK, ratio)
        per_fake[ratio] = {s: v / n_fakes for s, v in d.items()}
        print(
            f"  ratio {ratio:4.2f} ({n_fakes} fakes): distortion "
            f"{fmt(*mean_ci(list(d.values())))}, "
            f"per fake {fmt(*mean_ci(list(per_fake[ratio].values())))}"
        )
    for low, high in ((0.05, 0.1), (0.1, 0.2)):
        gaps = [per_fake[high][s] - per_fake[low][s] for s in per_fake[low]]
        print(f"  per fake at {high} - at {low}: {fmt(*mean_ci(gaps))}")


def main(argv: list[str]) -> None:
    if argv[:1] != ["summary"]:
        n_seeds = int(argv[0]) if argv else len(SEEDS)
        run_experiment(NAME, CONFIGS, range(SEEDS.start, SEEDS.start + n_seeds))
    results = load(NAME)
    n = len(next(iter(results.values())))
    print(f"E2: {len(results)} configurations x {n} seeds, invariants checked every step\n")
    network_table(results)
    predictions(results)


if __name__ == "__main__":
    main(sys.argv[1:])
