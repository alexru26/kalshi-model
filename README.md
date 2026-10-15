# kalshi-model

An agent-based model of one Kalshi binary prediction market. See `PLAN.md` for the design and build order.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest
uv run ruff check && uv run ruff format --check
```
