"""Pattern-oriented calibration (Grimm et al. 2005; PLAN.md Sections 2.2 and 4, Step 7).

The default Kalshi market should reproduce several real patterns at once:

1. Spreads of a few cents: the median bid-ask spread lies within the middle half of
   liquid Kalshi markets' spreads (their 25th to 75th percentiles). Real markets differ,
   and the model is one market, so any value in that range is consistent with Kalshi.
   This only rules out wide spreads: Kalshi's 25th percentile is the 1c tick, and the
   model's median spread is 1c, Kalshi's tightest quartile (Kalshi median 2c, IQR
   1-6c), so the pattern does not pin the spread down.
2. Kalshi trade sizes: the quantiles of fill sizes match Kalshi's trades.
3. The book is rarely empty: at most 2% of steps end with no bid or no offer (the
   Step 5 criterion, kept while tuning so larger orders cannot empty the book).
4. Informed agents earn more: the best-informed tenth out-earns agents with no clues,
   who lose on average (they pay spreads and fees for noise trading).

Reported, not required: whether the price is more accurate than the average agent's
own judgment (lower Brier score against the outcome, smaller gap to P*). Each agent is
scored on its own-evidence posterior (no market term, no depth signal, no noise), so
agents with no clues score 0.5. Beliefs are averages, so the price is roughly a
weighted average of agents' beliefs and comes out only slightly more accurate than the
average agent's own judgment: over 144 seeds its Brier score was better by 0.0042 (se
0.0023), and the price was better in 55% of seeds (decided with the user in Step 7).
The `beats_agents` columns in `results/calibration.csv` predate this and compared the
price with agents' beliefs after averaging in the market price.

`measure` reads these off finished runs; `targets` loads the real values fetched by
`data/fetch_kalshi_patterns.py`; `evaluate` scores a run set against them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from kalshi_model.agents import sigmoid
from kalshi_model.model import MarketModel

PATTERNS_FILE = Path(__file__).resolve().parents[2] / "data" / "kalshi_patterns.json"
SIZE_QUANTILES = ("q25", "q50", "q75", "q90")
REQUIRED = ("spread", "sizes", "book_filled", "informed_earn")
"""Patterns a calibration must meet; `beats_agents` is reported only."""
MAX_EMPTY_SHARE = 0.02
"""Most steps that may end with one side of the book empty."""
SIZE_TOLERANCE = math.log(2)
"""Accepted mean absolute log ratio of fill-size quantiles: within a factor of 2."""


def targets(path: Path = PATTERNS_FILE) -> dict:
    data = json.loads(path.read_text())
    return {
        "spread": data["spread_cents"]["q50"],
        "spread_band": (data["spread_cents"]["q25"], data["spread_cents"]["q75"]),
        "sizes": {q: data["trade_size_contracts"][q] for q in SIZE_QUANTILES},
    }


def measure(models: list[MarketModel]) -> dict:
    """Pool the calibration patterns over finished CLOB runs."""
    spreads, sizes, empty = [], [], []
    price_brier, agent_brier, price_gap, agent_gap = [], [], [], []
    informed_pnl, uninformed_pnl = [], []
    for model in models:
        df = model.datacollector.get_model_vars_dataframe()
        spreads.append(float(df.spread.median()))
        empty.append(float(df.empty_book.mean()))
        sizes += [f.qty for f in model.book.fills]
        y = float(model.world.y)
        price, p_star = model.price, model.p_star
        price_brier.append((price - y) ** 2)
        price_gap.append(abs(price - p_star))
        for trader in model.traders:
            belief = sigmoid(trader.belief(None, 0.0, 0.0))
            agent_brier.append((belief - y) ** 2)
            agent_gap.append(abs(belief - p_star))
        n_clues = model.traits.n_clues
        pnl = np.array([t.pnl for t in model.traders]) / 100
        informed_pnl.append(pnl[n_clues >= np.quantile(n_clues, 0.9)].mean())
        uninformed_pnl.append(pnl[n_clues == 0].mean())
    return {
        "spread": float(np.median(spreads)),
        "empty_share": float(np.mean(empty)),
        "sizes": {q: float(np.quantile(sizes, int(q[1:]) / 100)) for q in SIZE_QUANTILES},
        "price_brier": float(np.mean(price_brier)),
        "agent_brier": float(np.mean(agent_brier)),
        "price_gap": float(np.mean(price_gap)),
        "agent_gap": float(np.mean(agent_gap)),
        "informed_pnl": float(np.mean(informed_pnl)),
        "uninformed_pnl": float(np.mean(uninformed_pnl)),
    }


def evaluate(patterns: dict, target: dict) -> dict:
    """Which patterns are met, and a distance to rank configurations (0 is a perfect fit)."""
    size_error = float(
        np.mean([abs(math.log(patterns["sizes"][q] / target["sizes"][q])) for q in SIZE_QUANTILES])
    )
    spread_error = abs(math.log(patterns["spread"] / target["spread"]))
    met = {
        "spread": target["spread_band"][0] <= patterns["spread"] <= target["spread_band"][1],
        "sizes": size_error <= SIZE_TOLERANCE,
        "book_filled": patterns["empty_share"] <= MAX_EMPTY_SHARE,
        "beats_agents": patterns["price_brier"] < patterns["agent_brier"],
        "informed_earn": patterns["informed_pnl"] > patterns["uninformed_pnl"]
        and patterns["uninformed_pnl"] < 0,
    }
    return {
        "met": met,
        "accepted": all(met[k] for k in REQUIRED),
        "size_error": size_error,
        "spread_error": spread_error,
        "distance": size_error + spread_error,
    }
