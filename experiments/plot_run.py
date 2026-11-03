"""Plot one run's price over time against P*, with clue releases marked, and its network.

Writes two PNGs: the price chart (default `figures/price_<market or config>_seed<seed>.png`)
and, beside it with `_network` added to the name, the agents' social network colored by
each agent's P&L, with P&L against degree and private clues. The market is
the Kalshi default unless `--market` picks the E1 fee-free CLOB or the legacy AMM; `--disinfo`
sets the E2 variable. `--config` reads a custom configuration from a
TOML file (see `configs/` and `kalshi_model/custom.py`), and each `--set key=value`
overrides one setting on top of it (`traits.<name>` pins a trait).

Usage: uv run python experiments/plot_run.py [seed] [--market kalshi|clob|amm]
       [--disinfo RATIO] [--config FILE] [--set KEY=VALUE ...]
       [--out PATH]
"""

import argparse
from pathlib import Path

from kalshi_model.custom import MARKETS, build_custom, load_custom, parse_assignment
from kalshi_model.model import MarketModel
from kalshi_model.network_plot import network_chart
from kalshi_model.plot import price_chart, run_label

FIGURES = Path(__file__).resolve().parents[1] / "figures"


def main(argv: list[str] | None = None) -> tuple[Path, Path]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("seed", type=int, nargs="?", default=20000)
    parser.add_argument("--market", choices=MARKETS, default=None, help="default: kalshi")
    parser.add_argument("--disinfo", type=float, default=None, help="fake clues per true news clue")
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
    network_out = out.with_name(f"{out.stem}_network{out.suffix}")
    network_chart(model, title).savefig(network_out, dpi=150)
    print(f"wrote {network_out}")
    return out, network_out


if __name__ == "__main__":
    main()
