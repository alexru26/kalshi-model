"""Plot one run's price over time against P*, with clue releases marked.

Writes a PNG (default `figures/price_<market or config>_seed<seed>.png`). The market is
the Kalshi default unless `--market` picks the E1 fee-free CLOB or the AMM; `--disinfo`
and `--network` set the E2 variables. `--config` reads a custom configuration from a
TOML file (see `configs/` and `kalshi_model/custom.py`), and each `--set key=value`
overrides one setting on top of it (`traits.<name>` pins a trait).

Usage: uv run python experiments/plot_run.py [seed] [--market kalshi|clob|amm]
       [--disinfo RATIO] [--network NAME] [--config FILE] [--set KEY=VALUE ...]
       [--out PATH]
"""

import argparse
from pathlib import Path

from kalshi_model.custom import MARKETS, build_custom, load_custom, parse_assignment
from kalshi_model.model import MarketModel
from kalshi_model.plot import price_chart, run_label

FIGURES = Path(__file__).resolve().parents[1] / "figures"


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("seed", type=int, nargs="?", default=20000)
    parser.add_argument("--market", choices=MARKETS, default=None, help="default: kalshi")
    parser.add_argument("--disinfo", type=float, default=None, help="fake clues per true news clue")
    parser.add_argument("--network", default=None, help="network type (default: Config's)")
    parser.add_argument("--config", type=Path, default=None, help="TOML custom configuration")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one setting, e.g. --set n_agents=2 or --set traits.n_clues=[10,0]",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    overrides: dict[str, object] = {}
    if args.disinfo is not None:
        overrides["disinfo_ratio"] = args.disinfo
    if args.network is not None:
        overrides["network"] = args.network
    try:
        overrides.update(parse_assignment(text) for text in args.set)
        if args.config is not None:
            custom = load_custom(args.config, overrides, args.market)
        else:
            custom = build_custom({}, overrides, args.market)
    except (OSError, ValueError) as e:
        parser.error(str(e))

    model = MarketModel(custom.config, args.seed, fixed_traits=custom.fixed_traits, debug=True)
    model.run()

    name = args.config.stem if args.config is not None else args.market or "kalshi"
    out = args.out or FIGURES / f"price_{name}_seed{args.seed}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    title = f"{args.config.stem}: {run_label(model)}" if args.config is not None else None
    price_chart(model, title).savefig(out, dpi=150)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
