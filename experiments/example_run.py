"""Print one AMM run's price next to P*, to see how far the price moves toward the benchmark.

Usage: uv run python experiments/example_run.py [seed]
"""

import sys

from kalshi_model.config import amm_config
from kalshi_model.model import MarketModel


def main(seed: int) -> None:
    model = MarketModel(amm_config(), seed, debug=True)
    model.run()
    df = model.datacollector.get_model_vars_dataframe()
    print(f"seed {seed}, outcome {'YES' if model.world.y else 'NO'}")
    print(" step  price     P*   volume (last 200 steps)")
    for end in range(200, len(df) + 1, 200):
        window = df.iloc[end - 200 : end]
        row = window.iloc[-1]
        print(f"{end:5d}  {row.price:5.3f}  {row.p_star:5.3f}  {window.volume.sum():6d}")
    pnl = sorted(model.traders, key=lambda t: t.pnl)
    print(f"mean |price - P*|: {(df.price - df.p_star).abs().mean():.3f}")
    print(f"mean |0.5 - P*|:   {(0.5 - df.p_star).abs().mean():.3f}")
    print(f"agent P&L range: {pnl[0].pnl / 100:.2f} to {pnl[-1].pnl / 100:.2f} dollars")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
