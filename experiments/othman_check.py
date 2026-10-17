"""Compare zero-intelligence trading in our order book with Othman (2008).

Usage: uv run python experiments/othman_check.py [trades per cell]
"""

import sys

import numpy as np

from kalshi_model.zi import othman_expected_price, simulate_zi_market


def main(n_trades: int) -> None:
    rng = np.random.default_rng(2008)
    print("    S  mean belief  Othman formula  order book (+- 1 s.e.)")
    for concentration in (5, 25, 100):
        for mean in (0.2, 0.5, 0.8):
            prices = simulate_zi_market(mean, concentration, n_trades, rng) / 100
            se = prices.std() / np.sqrt(n_trades)
            formula = othman_expected_price(mean, concentration)
            row = f"{concentration:5d}  {mean:11.1f}  {formula:14.4f}"
            print(f"{row}  {prices.mean():.4f} +- {se:.4f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20_000)
