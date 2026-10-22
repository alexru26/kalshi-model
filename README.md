# kalshi-model

An agent-based model of one Kalshi binary prediction market. See `PLAN.md` for the design and build order.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest
uv run ruff check && uv run ruff format --check
```

## Experiments

Predictions are written before each experiment runs (`PREDICTIONS.md`) and judged in `RESULTS.md`. Each script writes one row per run to `results/` and prints the comparisons; add `summary` to reprint them from the saved results.

```sh
uv run python experiments/e1_rules.py         # E1: AMM vs CLOB vs Kalshi
uv run python experiments/e2_information.py   # E2: networks and disinformation
uv run python experiments/e3_manipulation.py  # E3: honest twin vs pump and dump vs spoofing
```

To see one run, `experiments/plot_run.py` draws its price over time against `P*`, with each news and fake clue's release marked, plus volume and (order book only) the spread. It writes a PNG to `figures/` (gitignored):

```sh
uv run python experiments/plot_run.py 20000                       # Kalshi default, seed 20000
uv run python experiments/plot_run.py 20000 --market amm          # or clob
uv run python experiments/plot_run.py 20000 --disinfo 0.2 --network watts_strogatz
```

### Custom configurations

To explore small or simple systems, write a TOML file and pass it with `--config`. It may set `market` (`kalshi`, `clob`, or `amm`), any `Config` field (`src/kalshi_model/config.py`), and a `[traits]` table that pins agent traits to exact values, either one value for every agent or a list with one value per agent. Unpinned traits are drawn as usual. `n_clues_total = 0` removes all information, `mean_degree = 0` removes the network, and `mm_size = 0` turns off the market maker. Examples are in `configs/`:

```sh
uv run python experiments/plot_run.py --config configs/two_agents_no_info.toml
uv run python experiments/plot_run.py 7 --config configs/informed_vs_uninformed.toml
uv run python experiments/plot_run.py --config configs/small_network_news.toml
```

`--set key=value` overrides one setting on top of the file (or the defaults), and `traits.<name>` pins a trait:

```sh
uv run python experiments/plot_run.py --config configs/two_agents_no_info.toml --set traits.noise=0.5
uv run python experiments/plot_run.py --config configs/informed_vs_uninformed.toml --set "traits.n_clues=[10, 10]"
```
