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
