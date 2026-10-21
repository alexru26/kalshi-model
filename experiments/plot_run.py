"""Plot one run's price over time against P*, with clue releases marked.

Writes a PNG (default `figures/price_<market>_seed<seed>.png`). The market is the
Kalshi default unless `--market` picks the E1 fee-free CLOB or the AMM; `--disinfo`
and `--network` set the E2 variables.

Usage: uv run python experiments/plot_run.py [seed] [--market kalshi|clob|amm]
       [--disinfo RATIO] [--network NAME] [--out PATH]
"""

import argparse
from pathlib import Path

from kalshi_model.config import Config, amm_config, clob_config
from kalshi_model.model import MarketModel
from kalshi_model.plot import price_chart

MARKETS = {"kalshi": Config, "clob": clob_config, "amm": amm_config}
FIGURES = Path(__file__).resolve().parents[1] / "figures"


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("seed", type=int, nargs="?", default=20000)
    parser.add_argument("--market", choices=MARKETS, default="kalshi")
    parser.add_argument("--disinfo", type=float, default=0.0, help="fake clues per true news clue")
    parser.add_argument("--network", default=None, help="network type (default: Config's)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    overrides: dict[str, object] = {"disinfo_ratio": args.disinfo}
    if args.network is not None:
        overrides["network"] = args.network
    model = MarketModel(MARKETS[args.market](**overrides), args.seed, debug=True)
    model.run()

    out = args.out or FIGURES / f"price_{args.market}_seed{args.seed}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    price_chart(model).savefig(out, dpi=150)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
