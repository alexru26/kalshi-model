"""Step 5 check: solvency and empty-book steps in the three E1 markets.

Runs each market on the same seeds with the conservation invariants checked every step,
then reports means across seeds.

Usage: uv run python experiments/clob_check.py [number of seeds]
"""

import sys

import numpy as np

from kalshi_model.config import Config, amm_config, clob_config
from kalshi_model.model import MarketModel


def check(label: str, config: Config, seeds: range) -> None:
    rows = []
    for seed in seeds:
        model = MarketModel(config, seed, debug=True)  # raises on any invariant violation
        model.run()
        df = model.datacollector.get_model_vars_dataframe()
        late = df.iloc[len(df) // 2 :]
        row = {
            "error": (late.price - late.p_star).abs().mean(),
            "volume": df.volume.sum(),
            "empty": model.empty_book_steps,
            "broke": sum(t.cash == 0 and t.position == 0 for t in model.traders),
        }
        if model.book is not None:
            maker = model.market_maker.account
            row |= {
                "spread": df.spread.median(),
                "quoting": df.mm_quoting.mean(),
                "maker_pnl": (maker.cash - config.mm_cash) / 100,
                "max_inventory": df.mm_position.abs().max(),
                "fees": model.book.fees_collected / 100,
            }
        rows.append(row)
    means = {key: np.mean([row[key] for row in rows]) for key in rows[0]}
    print(f"{label}: {len(rows)} seeds, invariants held every step")
    for key, value in means.items():
        print(f"  {key:14s} {value:10.3f}")


def main(n_seeds: int) -> None:
    seeds = range(n_seeds)
    check("Kalshi (CLOB + fees)", Config(), seeds)
    check("CLOB (no fees)", clob_config(), seeds)
    check("AMM (LMSR)", amm_config(), seeds)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
