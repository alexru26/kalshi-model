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
```
