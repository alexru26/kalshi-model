"""Per-run metrics for the experiments.

`run_metrics` turns one finished run into a flat dict of numbers: one row of
`results/*.csv`, since each run is one data point. Prices and gaps are in cents and
P&L in dollars. Metrics that a market does not have (fees and maker fills under the
legacy AMM, for example) are NaN.

News response: for each real news clue whose 300-step window fits in the run, the
price move toward the clue (up for a YES clue, down for a NO clue) from the step
before its release, at each lag in `RESPONSE_LAGS`. A run's response is the mean over
its clues. The price can still stay short of `P*` (underconfident, calibration slope
1.10-1.18), so the gap to the new `P*` rarely halves; the half-life is instead the
lag at which the mean move reaches half its size at the window's end (`half_life`).
Each run records the half-life of its own mean curve (`news_half_life`), so its spread
over seeds can be shown; the experiments also report the half-life of the mean curve
over seeds, which is less noisy.

E3 adds `manipulation_metrics`, which compares a manipulated run with its honest twin
on the same seed.
"""

from __future__ import annotations

import math

import numpy as np

from kalshi_model.lmsr import subsidy_cents
from kalshi_model.manipulator import OWNER as MANIPULATOR
from kalshi_model.market_maker import OWNER as MM
from kalshi_model.model import MarketModel
from kalshi_model.orderbook import BUY

NEWS_WINDOW = 300
"""Steps after a news release over which the price response is measured. News comes
about every 95 steps, so later clues fall inside the window, but their directions are
independent of this clue's, so they add only noise to the mean move. Earlier runs found the
mean move still growing at 90 steps."""
RESPONSE_LAGS = (0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 120, 150, 200, 250, 300)
NEAR_FIFTY = (40.0, 60.0)
"""Prices, in cents, at which the Kalshi fee per contract is near its 50c peak."""
ERROR_BLOCKS = 10
"""Equal blocks of the run over which the aggregation error is also averaged."""


def _mean(values) -> float:
    """Mean of the finite values; NaN if there are none."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else math.nan


def response_column(lag: int) -> str:
    return f"response_{lag:02d}"


def run_metrics(model: MarketModel) -> dict[str, float]:
    """Every metric of one finished run."""
    if not model.settled:
        raise ValueError("the run has not finished")
    df = model.datacollector.get_model_vars_dataframe()
    price = 100 * df.price.to_numpy(dtype=float)
    p_star = 100 * df.p_star.to_numpy(dtype=float)
    y = float(model.world.y)
    row: dict[str, float] = {
        "y": y,
        "final_price": price[-1],
        "final_p_star": p_star[-1],
        "brier": (price[-1] / 100 - y) ** 2,
        "brier_p_star": (p_star[-1] / 100 - y) ** 2,
        "aggregation_loss": ((price[-1] - p_star[-1]) / 100) ** 2,
    }
    row |= _accuracy(price, p_star, fakes_sign=1.0 if y == 0 else -1.0)
    volume = df.volume.to_numpy()
    row |= {"volume": float(volume.sum()), "no_trade_share": float(np.mean(volume == 0))}
    row |= _news_response(model)
    row |= _pnl_by_informedness(model)
    row |= _market(model, df)
    row |= _network(model)
    return row


def _accuracy(price: np.ndarray, p_star: np.ndarray, fakes_sign: float) -> dict[str, float]:
    """Aggregation error |price - P*| and the signed gap in the fakes' direction.

    Fakes point against the truth, so the fakes' direction is down when YES is true.
    Steps before the CLOB has any price (NaN) are left out.
    """
    gap = price - p_star
    late = len(gap) // 2
    last_quarter = 3 * len(gap) // 4
    row = {
        "error": _mean(np.abs(gap)),
        "error_late": _mean(np.abs(gap[late:])),
        "signed_gap": _mean(fakes_sign * gap),
        "signed_gap_last_quarter": _mean(fakes_sign * gap[last_quarter:]),
    }
    for b, block in enumerate(np.array_split(np.abs(gap), ERROR_BLOCKS)):
        row[f"error_block_{b}"] = _mean(block)
    return row


def news_events(model: MarketModel) -> list[dict]:
    """Each real news clue with a full window: its start price, P* jump, and response.

    Row i of the step data is step i + 1, so the price just before the release at step
    t is row t - 2 and the price at lag k is row t - 1 + k.
    """
    df = model.datacollector.get_model_vars_dataframe()
    price = 100 * df.price.to_numpy(dtype=float)
    p_star = 100 * df.p_star.to_numpy(dtype=float)
    events = []
    for clue_id, t in sorted(model.release_step.items()):
        if clue_id in model.fake_ids or t < 2 or t - 1 + NEWS_WINDOW >= len(price):
            continue
        before = price[t - 2]
        if math.isnan(before):
            continue
        sign = 1.0 if model.points_yes(clue_id) else -1.0
        lags = np.array(RESPONSE_LAGS)
        events.append(
            {
                "before": before,
                "p_star_jump": sign * (p_star[t - 1] - p_star[t - 2]),
                "response": sign * (price[t - 1 + lags] - before),
            }
        )
    return events


def _news_response(model: MarketModel) -> dict[str, float]:
    events = news_events(model)
    row: dict[str, float] = {"n_news": float(len(events))}
    responses = np.array([e["response"] for e in events]).reshape(len(events), len(RESPONSE_LAGS))
    mean = responses.mean(axis=0) if events else np.full(len(RESPONSE_LAGS), math.nan)
    row |= {response_column(lag): float(v) for lag, v in zip(RESPONSE_LAGS, mean, strict=True)}
    row["news_half_life"] = half_life(mean) if events else math.nan
    row["p_star_jump"] = _mean([e["p_star_jump"] for e in events])
    near = [NEAR_FIFTY[0] <= e["before"] <= NEAR_FIFTY[1] for e in events]
    for name, keep in (("near50", True), ("far", False)):
        chosen = [e["response"][-1] for e, n in zip(events, near, strict=True) if n == keep]
        row[f"n_{name}"] = float(len(chosen))
        row[f"response_{name}"] = _mean(chosen)
    return row


def half_life(curve: np.ndarray, lags: tuple[int, ...] = RESPONSE_LAGS) -> float:
    """First lag at which `curve` reaches half its final value (linear interpolation).

    NaN if the final value is not positive (the price did not move toward the news).
    """
    curve = np.asarray(curve, dtype=float)
    half = curve[-1] / 2
    if not half > 0:
        return math.nan
    for i in range(len(curve)):
        if curve[i] >= half:
            if i == 0:
                return float(lags[0])
            lo, hi = curve[i - 1], curve[i]
            return float(lags[i - 1] + (half - lo) / (hi - lo) * (lags[i] - lags[i - 1]))
    return math.nan  # unreachable: the last point always reaches half of itself


def _tiers(model: MarketModel) -> dict[str, np.ndarray]:
    """Agents with no private clue, the best-informed tenth, and everyone else."""
    n_clues = model.traits.n_clues
    top = n_clues >= np.quantile(n_clues, 0.9)
    none = n_clues == 0
    return {"none": none, "mid": ~top & ~none, "top": top}


def _pnl_by_informedness(model: MarketModel) -> dict[str, float]:
    pnl = np.array([t.pnl for t in model.traders]) / 100
    expected = expected_pnl(model)
    row = {f"pnl_{name}": _mean(pnl[mask]) for name, mask in _tiers(model).items()}
    return row | {
        f"pnl_{name}_at_p_star": _mean(expected[mask]) for name, mask in _tiers(model).items()
    }


def expected_pnl(model: MarketModel) -> np.ndarray:
    """Each agent's P&L in dollars with its closing position valued at final P* instead
    of the outcome.

    Without disinformation P* is the exact probability of YES given everything the
    market knows, so this has the same expectation as the realized P&L with far less
    outcome luck. With fakes, trades carry information about the outcome
    (fakes always point against it), so this would be biased: use it only without them.
    """
    y, p_star = float(model.world.y), model.p_star
    realized = np.array([t.pnl for t in model.traders], dtype=float)
    positions = np.array(model.closing_positions, dtype=float)
    return (realized - 100 * positions * (y - p_star)) / 100


def fill_values(model: MarketModel, payout: float | None = None) -> dict[str, np.ndarray]:
    """Per agent: contracts and settlement value (cents, net of fees) of maker and taker
    fills, and fees paid. An agent's maker plus taker value is its P&L.

    `payout` is what one YES contract is worth, in cents: the outcome by default, or
    100 x P* to value fills at the benchmark (see `expected_pnl`).
    """
    n = len(model.traders)
    out = {k: np.zeros(n) for k in ("maker_qty", "taker_qty", "maker_value", "taker_value", "fees")}
    payout = 100 * int(model.world.y) if payout is None else payout
    for f in model.book.fills:
        buy_value = f.qty * (payout - f.price)  # what the buyer gains at settlement
        taker_value = buy_value if f.taker_side == BUY else -buy_value
        if isinstance(f.taker, int):
            out["taker_qty"][f.taker] += f.qty
            out["taker_value"][f.taker] += taker_value - f.taker_fee
            out["fees"][f.taker] += f.taker_fee
        if isinstance(f.maker, int):
            out["maker_qty"][f.maker] += f.qty
            out["maker_value"][f.maker] += -taker_value - f.maker_fee
            out["fees"][f.maker] += f.maker_fee
    return out


def _market(model: MarketModel, df) -> dict[str, float]:
    if model.lmsr is not None:  # legacy AMM, unused by the experiments
        nan = math.nan
        return {
            "maker_pnl": (model.lmsr.cash - subsidy_cents(model.config.lmsr_b)) / 100,
            "fees": 0.0,
            "spread": nan,
            "depth": nan,
            "empty_share": nan,
            "agent_liquidity_share": nan,
            "maker_return": nan,
            "taker_return": nan,
            "maker_return_at_p_star": nan,
            "taker_return_at_p_star": nan,
            "fee_per_contract_none": nan,
            "fee_per_contract_top": nan,
        }
    v = fill_values(model)
    at_p_star = fill_values(model, payout=100 * model.p_star)
    maker_qty, taker_qty = v["maker_qty"].sum(), v["taker_qty"].sum()
    total_qty = sum(f.qty for f in model.book.fills)
    tiers = _tiers(model)
    traded = v["maker_qty"] + v["taker_qty"]

    def fee_per_contract(mask: np.ndarray) -> float:
        qty = traded[mask].sum()
        return float(v["fees"][mask].sum() / qty) if qty else math.nan

    return {
        "maker_pnl": (model.market_maker.account.cash - model.config.mm_cash) / 100,
        "fees": model.book.fees_collected / 100,
        "spread": float(df.spread.median()),
        "depth": float(df.depth.mean()),
        "empty_share": float(df.empty_book.mean()),
        "agent_liquidity_share": float(maker_qty / total_qty) if total_qty else math.nan,
        "maker_return": float(v["maker_value"].sum() / maker_qty) if maker_qty else math.nan,
        "taker_return": float(v["taker_value"].sum() / taker_qty) if taker_qty else math.nan,
        "maker_return_at_p_star": float(at_p_star["maker_value"].sum() / maker_qty)
        if maker_qty
        else math.nan,
        "taker_return_at_p_star": float(at_p_star["taker_value"].sum() / taker_qty)
        if taker_qty
        else math.nan,
        "fee_per_contract_none": fee_per_contract(tiers["none"]),
        "fee_per_contract_top": fee_per_contract(tiers["top"]),
    }


def degree_quarters(model: MarketModel) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the best- and worst-connected quarters of agents: exactly
    `n_agents // 4` each, disjoint, ranked by degree with ties broken by agent index.

    Agents sit on nodes at random and their traits are i.i.d. per index, so breaking
    ties by index is unbiased.
    """
    degree = np.array([len(n) for n in model.neighbors])
    order = np.argsort(degree, kind="stable")
    quarter = len(degree) // 4
    return order[len(order) - quarter :], order[:quarter]


def _network(model: MarketModel) -> dict[str, float]:
    """News spread speed, and how soon and how well the best- and worst-connected
    quarters of agents (see `degree_quarters`) learn news and trade."""
    news = [c for c in model.release_step if c not in model.fake_ids]
    reach = [model.half_reach_step.get(c, math.inf) - model.release_step[c] for c in news]
    finite = [r for r in reach if math.isfinite(r)]
    delay = np.full(len(model.traders), math.nan)
    for i in range(len(model.traders)):
        waits = [
            model.learned_step[c][i] - model.release_step[c]
            for c in news
            if i in model.learned_step[c]
        ]
        if waits:
            delay[i] = np.mean(waits)
    pnl = np.array([t.pnl for t in model.traders]) / 100
    top, bottom = degree_quarters(model)
    return {
        "half_reach": _mean(finite),
        "half_reach_never": float(1 - len(finite) / len(reach)) if reach else math.nan,
        "learn_delay_top_degree": _mean(delay[top]),
        "learn_delay_bottom_degree": _mean(delay[bottom]),
        "pnl_top_degree": _mean(pnl[top]),
        "pnl_bottom_degree": _mean(pnl[bottom]),
        "pnl_top_degree_at_p_star": _mean(expected_pnl(model)[top]),
        "pnl_bottom_degree_at_p_star": _mean(expected_pnl(model)[bottom]),
    }


# E3: manipulation.

DISTORTION_LAGS = (0, 1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000)
"""Steps after `manip_start` at which the paired distortion is recorded."""
AFTERMATH = 300
"""Steps after the manipulation layer stops over which lasting distortion is averaged."""
DEPTH_SPAN = 100
"""Steps before `manip_start` over which book depth is also averaged (`depth_before_mean`)."""
COUNTERPARTIES = ("none", "mid", "top", "mm")
"""The manipulator's counterparties: agents by informedness tier, and the market maker."""


def distortion_column(lag: int) -> str:
    return f"distortion_{lag:04d}"


def manipulator_pnl_at_p_star(model: MarketModel) -> float:
    """The manipulator's P&L in dollars, its closing position valued at final P*."""
    man = model.manipulator
    y, p_star = float(model.world.y), model.p_star
    return (man.pnl - 100 * man.closing_position * (y - p_star)) / 100


def manipulation_metrics(model: MarketModel, twin: MarketModel | None = None) -> dict[str, float]:
    """The manipulator's P&L and, against the honest twin on the same seed, what its
    manipulation did (NaN for the twin itself, `twin=None`).

    Distortion is the price gap to the twin's run step by step, `price - twin price` in
    cents (both strategies push the price up). Both runs share the world, the population,
    and so P*, so this is also the change in the signed gap to P*. It is averaged over the
    steps on which the layer acted (`distortion`), over the `AFTERMATH` steps after
    (`distortion_after`), and recorded at each lag in `DISTORTION_LAGS` after
    `manip_start`. `depth_before` is the book depth within `depth_window` at the end of
    the step before `manip_start`, and `depth_before_mean` its mean over the `DEPTH_SPAN`
    steps before. Money taken from others is their P&L at P* in the twin's run minus in
    this run. Counterparties are split by informedness tier (`_tiers`) and the market
    maker: contracts each sold to the manipulator (`sold_to_*`) and bought from it
    (`bought_from_*`) while the layer acted, and each one's share of all contracts traded
    in the twin's run over the same steps (`share_*`), the share it would have by chance.
    `buyer_w_depth` is the contract-weighted `w_depth` of the agents who bought from the
    manipulator, and `buyer_w_depth_chance` the same for every agent buyer in the twin's
    run over those steps.
    """
    config = model.config
    df = model.datacollector.get_model_vars_dataframe()
    start = config.manip_start
    row: dict[str, float] = {
        "manip_pnl": model.manipulator.pnl / 100,
        "manip_pnl_at_p_star": manipulator_pnl_at_p_star(model),
        "depth_before": float(df.depth.iloc[start - 2]) if start >= 2 else math.nan,
        "depth_before_mean": _mean(
            df.depth.to_numpy(dtype=float)[max(0, start - 1 - DEPTH_SPAN) : start - 1]
        ),
    }
    names = (
        ["premium", "premium_realized", "distortion", "distortion_after", "layer_steps"]
        + [distortion_column(k) for k in DISTORTION_LAGS]
        + ["taken_from_agents", "taken_from_mm", "fees_change", "pump_reached"]
        + ["bought", "sold", "fake_filled", "buyer_w_depth", "buyer_w_depth_chance"]
        + [f"{kind}_{c}" for kind in ("sold_to", "bought_from", "share") for c in COUNTERPARTIES]
    )
    if twin is None:
        return row | dict.fromkeys(names, math.nan)

    price = 100 * df.price.to_numpy(dtype=float)
    twin_price = 100 * twin.datacollector.get_model_vars_dataframe().price.to_numpy(dtype=float)
    gap = price - twin_price  # row i is step i + 1
    steps = model.manipulator.layer_steps
    end = steps[-1] + 1 if steps else start
    row |= {
        "premium": row["manip_pnl_at_p_star"] - manipulator_pnl_at_p_star(twin),
        "premium_realized": (model.manipulator.pnl - twin.manipulator.pnl) / 100,
        "distortion": _mean(gap[[t - 1 for t in steps]]),
        "distortion_after": _mean(gap[end - 1 : end - 1 + AFTERMATH]),
        "layer_steps": float(len(steps)),
    }
    for lag in DISTORTION_LAGS:
        i = start - 1 + lag
        row[distortion_column(lag)] = float(gap[i]) if i < len(gap) else math.nan

    def mm_at_p_star(m: MarketModel) -> float:
        account = m.market_maker.account
        y, p_star = float(m.world.y), m.p_star
        realized = account.cash - m.config.mm_cash
        return (realized - 100 * m.market_maker.closing_position * (y - p_star)) / 100

    row |= {
        "taken_from_agents": float(expected_pnl(twin).sum() - expected_pnl(model).sum()),
        "taken_from_mm": mm_at_p_star(twin) - mm_at_p_star(model),
        "fees_change": (model.book.fees_collected - twin.book.fees_collected) / 100,
        "pump_reached": float(model.manipulator.pump_reached)
        if config.manipulator == "pump"
        else math.nan,
    }
    row |= _counterparties(model, twin, set(steps))
    return row


def _counterparties(model: MarketModel, twin: MarketModel, steps: set[int]) -> dict[str, float]:
    """Who traded with the manipulator while its layer acted, and by-chance shares."""
    tier_of = np.empty(len(model.traders), dtype=object)
    for name, mask in _tiers(model).items():
        tier_of[mask] = name

    def side(owner) -> str | None:
        if owner == MANIPULATOR:
            return None
        return "mm" if owner == MM else tier_of[owner]

    kinds = ("sold_to", "bought_from", "share")
    row = {f"{kind}_{c}": 0.0 for kind in kinds for c in COUNTERPARTIES}
    bought = sold = fake = 0
    w_depth = model.traits.w_depth
    depth_weighted = 0.0
    for fill, step in _fills_with_steps(model):
        if step not in steps or MANIPULATOR not in (fill.taker, fill.maker):
            continue
        manip_buys = (fill.taker == MANIPULATOR) == (fill.taker_side == BUY)
        other = fill.maker if fill.taker == MANIPULATOR else fill.taker
        if manip_buys:
            bought += fill.qty
            row[f"sold_to_{side(other)}"] += fill.qty
            fake += fill.qty if fill.maker == MANIPULATOR else 0
        else:
            sold += fill.qty
            row[f"bought_from_{side(other)}"] += fill.qty
            if isinstance(other, int):
                depth_weighted += fill.qty * w_depth[other]
    agent_bought = sum(row[f"bought_from_{c}"] for c in ("none", "mid", "top"))
    total = chance_bought = 0
    chance_weighted = 0.0
    for fill, step in _fills_with_steps(twin):
        if step not in steps:
            continue
        for owner in (fill.taker, fill.maker):
            if side(owner) is not None:
                row[f"share_{side(owner)}"] += fill.qty
                total += fill.qty
        buyer = fill.taker if fill.taker_side == BUY else fill.maker
        if isinstance(buyer, int):
            chance_bought += fill.qty
            chance_weighted += fill.qty * w_depth[buyer]
    for c in COUNTERPARTIES:
        row[f"share_{c}"] = row[f"share_{c}"] / total if total else math.nan
    return row | {
        "bought": float(bought),
        "sold": float(sold),
        "fake_filled": float(fake) if model.config.manipulator == "spoof" else math.nan,
        "buyer_w_depth": float(depth_weighted / agent_bought) if agent_bought else math.nan,
        "buyer_w_depth_chance": float(chance_weighted / chance_bought)
        if chance_bought
        else math.nan,
    }


def _fills_with_steps(model: MarketModel):
    """Each order-book fill with the step it happened on."""
    start = 0
    for step, mark in enumerate(model.fill_marks, start=1):
        for fill in model.book.fills[start:mark]:
            yield fill, step
        start = mark
