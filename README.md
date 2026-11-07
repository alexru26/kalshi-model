# kalshi-model

An agent-based model of one Kalshi binary prediction market. 300 traders hold private clues about a YES/NO outcome, hear news over a social network, and trade one YES contract in a CLOB with Kalshi's fee rule and a market-maker component. Every run has an exact benchmark, `P*`: the probability of YES given every real clue any agent holds. Prices are scored against it.

Three experiments each change one variable around one shared default (`configs/default.toml`), on the same 100 seeds:

- **E1 Kalshi rules:** fee-free order book vs Kalshi's taker fee.
- **E2 Information spread:** share of fake clues (disinformation) on a Holme-Kim network.
- **E3 Manipulation:** an honest trader vs the same trader running pump and dump or spoofing.

The defaults are calibrated against public Kalshi market data: spreads, trade sizes, a rarely empty book, informed traders out-earning uninformed ones, and price calibration.

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync                                       # install
uv run pytest                                 # 334 tests
uv run ruff check && uv run ruff format --check
uv run solara run kalshi_model.gui            # live GUI at http://localhost:8765
uv run python experiments/plot_run.py 20000   # chart one run (seed 20000) to figures/
```

## Architecture

```
            configs/default.toml -> Config (config.py)
                         |
                  MarketModel (model.py)  - one run, one Mesa model
   ________________________|_____________________________________
  |            |              |               |                  |
World       network.py     traits.py      OrderBook          MarketMaker
(world.py)  Holme-Kim      per-agent      (orderbook.py)     (market_maker.py)
outcome,    graph, clue    n_clues, cash  price-time         requotes each step
clues, P*   spreading                     priority, fees.py
                                              ^
                                              | orders
                             Trader (agents.py)  +  Manipulator (manipulator.py, E3)
```

Each step (`MarketModel.step`, `src/kalshi_model/model.py:297`):

1. Release scheduled news and fake clues to random seed agents (`_release_news`, line 352).
2. Spread every clue one hop along the network (`_spread`, line 370).
3. The market maker cancels and requotes around a moving price estimate (`market_maker.py:52`).
4. 20 agents, sampled each step, take a turn: form a belief, cancel stale orders, take any quote with enough edge after fees, otherwise maybe post a resting order sized by fractional Kelly. In E3 the manipulator also acts in a random slot.
5. Record the step. At the last step the market settles.

Conventions: money is integer cents, positions are signed YES counts, every account is fully collateralized, and each part of the model draws from its own random stream (`rng.py`) so paired comparisons across configurations stay valid.

## Key files

| File | What to look at |
| --- | --- |
| `src/kalshi_model/model.py` | `MarketModel`: setup (180), the scored price (273), `P*` (293), the step loop (297), invariants (436) |
| `src/kalshi_model/agents.py` | Trader decision rules: belief in log-odds (`belief_log_odds`, 36), probability weighting (65), Kelly sizing (70), one order-book turn (`take_turn_clob`, 246) |
| `src/kalshi_model/fees.py` | Kalshi fee `ceil(0.07 x C x P x (1 - P))` in exact integer arithmetic |
| `src/kalshi_model/world.py` | True outcome, clues, news schedule, and the benchmark `P*` (`p_star`, 102) |
| `src/kalshi_model/manipulator.py` | E3 strategies: phase machine (`_advance`, 122), pump and dump (`take_turn`, 106), spoofing (`_spoof`, 157) |
| `src/kalshi_model/network.py` | Holme-Kim social network and agent placement |
| `src/kalshi_model/traits.py` | Trait population: drawn informedness and cash, linked by a copula |
| `src/kalshi_model/experiment.py` | Runs configurations on shared seeds; paired differences with 95% intervals |
| `src/kalshi_model/config.py` | `Config`: every parameter, validated, defaults read from `configs/default.toml` |
| `experiments/e1_rules.py`, `e2_information.py`, `e3_manipulation.py` | The three experiments; each prints its written predictions against the data |

Other files: `configs/` (frozen defaults with a note per setting, small demo levels, test fixtures), `data/` (Kalshi data fetcher and the saved calibration targets), and `results/` (saved experiment CSVs, run metadata, and figures).

## Running the experiments

Each script writes one row per run to `results/<name>.csv` and the git commit, seeds, and full configurations to `results/<name>.json`, then prints each prediction against the data. Add `summary` to reprint from the saved results without rerunning.

```sh
uv run python experiments/e1_rules.py           # E1, 2 configs x 100 seeds
uv run python experiments/e2_information.py     # E2, 5 disinformation ratios x 100 seeds
uv run python experiments/e3_manipulation.py    # E3, twin vs pump vs spoof x 100 seeds
uv run python experiments/figures.py            # redraw results/figures/ from the CSVs
```

## Exploring the model

**Live GUI** (`uv run solara run kalshi_model.gui`; from WSL, open the address in the Windows browser). The sidebar has play/pause and step controls, presets (demo Levels 0 to 2 and every experiment configuration), the seed, every `Config` field, and pins for per-agent traits. Charts show the price against `P*`, volume, the bid and ask, the market maker, the network colored by who knows the latest clue, and in E3 the manipulator's orders and its price against the honest twin's.

**One run as PNGs** (`experiments/plot_run.py`): price vs `P*` with clue releases marked, plus the network colored by each agent's P&L.

```sh
uv run python experiments/plot_run.py 20000 --market clob     # fee-free order book
uv run python experiments/plot_run.py 20000 --disinfo 0.2     # with disinformation
```

**Custom configurations:** a TOML file passed with `--config` may set `market` (`kalshi` or `clob`), any `Config` field, and a `[traits]` table pinning agent traits (one value, or a list with one per agent). It only needs to list what differs from `configs/default.toml`. `--set key=value` overrides one setting. Setting `n_clues_total = 0`, `mean_degree = 0`, or `mm_size = 0` removes information, the network, or the market maker.

```sh
uv run python experiments/plot_run.py --config configs/level_1.toml --set noise=0.5
uv run python experiments/plot_run.py 20000 --config configs/test_1.toml --set "traits.n_clues=[10, 10]"
```
