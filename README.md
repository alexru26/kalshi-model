# kalshi-model

An agent-based model of one Kalshi binary prediction market.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest
uv run ruff check && uv run ruff format --check
```

## Experiments

Predictions are written before each experiment runs, and each script prints them against the data. Each script writes one row per run to `results/` and prints the comparisons; add `summary` to reprint them from the saved results.

```sh
uv run python experiments/e1_rules.py         # E1: CLOB vs Kalshi
uv run python experiments/e2_information.py   # E2: disinformation ratios
uv run python experiments/e3_manipulation.py  # E3: honest twin vs pump and dump vs spoofing
uv run python experiments/figures.py          # redraw results/figures/ from the saved results
```

To see one run, `experiments/plot_run.py` draws its price over time against `P*`, with each news and fake clue's release marked, plus volume and (order book only) the spread. A second PNG, with `_network` added to its name, draws the agents' social network with each agent colored by its P&L, and plots P&L against each agent's links and private clues. Both go to `figures/` (gitignored):

```sh
uv run python experiments/plot_run.py 20000                       # Kalshi default, seed 20000
uv run python experiments/plot_run.py 20000 --market clob         # fee-free order book
uv run python experiments/plot_run.py 20000 --disinfo 0.2           # with disinformation
```

### Custom configurations

To explore small or simple systems, write a TOML file and pass it with `--config`. It may set `market` (`kalshi`, `clob`, or the legacy `amm`, which no experiment uses), any `Config` field, and a `[traits]` table that pins agent traits to exact values, either one value for every agent or a list with one value per agent. Unpinned traits keep their defaults (informedness and cash drawn, the rest one value for all). `n_clues_total = 0` removes all information, `mean_degree = 0` removes the network, and `mm_size = 0` turns off the market maker. `configs/default.toml` holds the frozen defaults: every setting, with a short note on each. `Config` reads it at import, so editing a value there changes the default everywhere (the model refuses to start if a setting is missing, unknown, or invalid). Copy it as a starting point for a custom configuration, which needs to list only what differs. `configs/level_*.toml` are small demos of the model, and `configs/test_*.toml` are fixtures used by the tests:

```sh
uv run python experiments/plot_run.py --config configs/level_1.toml
uv run python experiments/plot_run.py 7 --config configs/test_1.toml
uv run python experiments/plot_run.py --config configs/test_2.toml
```

`--set key=value` overrides one setting on top of the file (or the defaults), and `traits.<name>` pins a trait:

```sh
uv run python experiments/plot_run.py --config configs/level_1.toml --set noise=0.5
uv run python experiments/plot_run.py --config configs/test_1.toml --set "traits.n_clues=[10, 10]"
```
