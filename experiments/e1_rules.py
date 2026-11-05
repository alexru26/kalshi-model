"""E1: Kalshi rules. CLOB vs Kalshi on shared seeds.

Writes results/e1_rules.csv (one row per run) and .json (commit, seeds, configs), then
prints each prediction against the data. `summary` reprints from the
saved CSV without rerunning.

Usage: uv run python experiments/e1_rules.py [number of seeds | summary]
"""

import sys

import numpy as np

from kalshi_model.config import Config, clob_config
from kalshi_model.experiment import (
    SEEDS,
    bootstrap,
    curve_half_life,
    load,
    mean_ci,
    mean_of,
    paired,
    run_experiment,
)
from kalshi_model.metrics import NEWS_WINDOW, response_column

NAME = "e1_rules"
CONFIGS = {"clob": clob_config(), "kalshi": Config()}
END = response_column(NEWS_WINDOW)


def fmt(mean: float, half: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} +- {half:.{digits}f}"


def fmt_boot(estimate: float, low: float, high: float, digits: int = 1) -> str:
    return f"{estimate:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def per_config_table(results: dict) -> None:
    metrics = [
        ("error", "mean |price - P*| (c)"),
        ("error_late", "same, second half (c)"),
        ("brier", "final Brier"),
        ("aggregation_loss", "final (price - P*)^2"),
        (END, f"news move by lag {NEWS_WINDOW} (c)"),
        ("p_star_jump", "P* jump per news (c)"),
        ("volume", "volume (contracts)"),
        ("no_trade_share", "no-trade steps"),
        ("spread", "median spread (c)"),
        ("depth", "depth within 2c"),
        ("empty_share", "empty-book steps"),
        ("fees", "fees ($)"),
        ("maker_pnl", "market maker P&L ($)"),
        ("agent_liquidity_share", "contracts with agent maker"),
        ("pnl_none", "P&L, no clues ($/agent)"),
        ("pnl_top", "P&L, top tenth ($/agent)"),
        ("pnl_none_at_p_star", "same at P*, no clues"),
        ("pnl_top_at_p_star", "same at P*, top tenth"),
    ]
    print(f"{'metric':32s}" + "".join(f"{label:>22s}" for label in CONFIGS))
    for metric, name in metrics:
        cells = []
        for label in CONFIGS:
            m, h = mean_ci([row[metric] for row in results[label].values()])
            cells.append(f"{fmt(m, h):>22s}" if np.isfinite(m) else f"{'-':>22s}")
        print(f"{name:32s}" + "".join(cells))
    print(f"{'news half-life (steps)':32s}", end="")
    for label in CONFIGS:
        print(f"{fmt_boot(*bootstrap(results, [label], curve_half_life)):>22s}", end="")
    print()


def predictions(results: dict) -> None:
    def diff(metric: str, a: str, b: str, digits: int = 3) -> str:
        return fmt(*paired(results, metric, a, b), digits)

    print("\nE1.1 fee slows news pricing, most near 50c (Kalshi - CLOB)")
    print(f"  news move by lag {NEWS_WINDOW} (c):         {diff(END, 'clob', 'kalshi')}")
    print(f"    news arriving at 40-60c:       {diff('response_near50', 'clob', 'kalshi')}")
    print(f"    news arriving outside 40-60c:  {diff('response_far', 'clob', 'kalshi')}")

    def near_minus_far(clob: list[dict], kalshi: list[dict]) -> float:
        d = [k["response_near50"] - c["response_near50"] for c, k in zip(clob, kalshi, strict=True)]
        e = [k["response_far"] - c["response_far"] for c, k in zip(clob, kalshi, strict=True)]
        return mean_ci(d)[0] - mean_ci(e)[0]

    print(
        "    near minus outside:            "
        f"{fmt_boot(*bootstrap(results, ['clob', 'kalshi'], near_minus_far), 3)}"
    )
    hl = bootstrap(
        results, ["clob", "kalshi"], lambda c, k: curve_half_life(k) - curve_half_life(c)
    )
    print(f"  half-life difference (steps):    {fmt_boot(*hl)}")
    n_near = np.mean([r["n_near50"] for r in results["kalshi"].values()])
    n_far = np.mean([r["n_far"] for r in results["kalshi"].values()])
    print(f"  news events per Kalshi run: {n_near:.1f} near 50c, {n_far:.1f} outside")

    def pct_change(metric: str):
        def stat(clob: list[dict], kalshi: list[dict]) -> float:
            return 100 * (mean_of(metric)(kalshi) / mean_of(metric)(clob) - 1)

        return stat

    print("\nE1.2 fees cut volume much more than accuracy (Kalshi vs CLOB, % change)")
    vol = bootstrap(results, ["clob", "kalshi"], pct_change("volume"))
    err = bootstrap(results, ["clob", "kalshi"], pct_change("error"))
    print(f"  volume:            {fmt_boot(*vol)}%")
    print(f"  aggregation error: {fmt_boot(*err)}%")

    print("\nE1.3 share of contracts whose maker is an agent, not the market maker")
    for label in ("clob", "kalshi"):
        share = [r["agent_liquidity_share"] for r in results[label].values()]
        print(f"  {label:8s} {fmt(*mean_ci(share))}")
    print(f"  Kalshi - CLOB: {diff('agent_liquidity_share', 'clob', 'kalshi')}")

    print("\nE1.4 makers out-earn takers on Kalshi (cents per contract, after fees)")
    k = results["kalshi"].values()
    print(f"  maker return:          {fmt(*mean_ci([r['maker_return'] for r in k]))}")
    print(f"  taker return:          {fmt(*mean_ci([r['taker_return'] for r in k]))}")
    gaps = [r["maker_return"] - r["taker_return"] for r in k]
    print(f"  maker - taker (paired): {fmt(*mean_ci(gaps))}")
    # Valued at P* (no fakes in E1): the same expectation without the outcome's luck.
    print(f"  maker return at P*:     {fmt(*mean_ci([r['maker_return_at_p_star'] for r in k]))}")
    print(f"  taker return at P*:     {fmt(*mean_ci([r['taker_return_at_p_star'] for r in k]))}")
    gaps = [r["maker_return_at_p_star"] - r["taker_return_at_p_star"] for r in k]
    print(f"  maker - taker at P*:    {fmt(*mean_ci(gaps))}")

    print("\nE1.5 fee per contract traded on Kalshi (cents)")
    print(f"  no clues:     {fmt(*mean_ci([r['fee_per_contract_none'] for r in k]))}")
    print(f"  top tenth:    {fmt(*mean_ci([r['fee_per_contract_top'] for r in k]))}")
    gaps = [r["fee_per_contract_none"] - r["fee_per_contract_top"] for r in k]
    print(f"  none - top:   {fmt(*mean_ci(gaps))}")

    print("\nReported: paired differences (Kalshi - CLOB)")
    for metric in ("error", "brier", "aggregation_loss", "volume"):
        print(f"  {metric:18s} {diff(metric, 'clob', 'kalshi'):>20s}")


def main(argv: list[str]) -> None:
    if argv[:1] != ["summary"]:
        n_seeds = int(argv[0]) if argv else len(SEEDS)
        run_experiment(NAME, CONFIGS, range(SEEDS.start, SEEDS.start + n_seeds))
    results = load(NAME)
    n = len(next(iter(results.values())))
    print(f"E1: {n} seeds per configuration, invariants checked every step\n")
    per_config_table(results)
    predictions(results)


if __name__ == "__main__":
    main(sys.argv[1:])
