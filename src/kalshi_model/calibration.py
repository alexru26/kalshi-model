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
   who lose on average (they pay spreads and fees for noise trading). P&L is valued at
   final P* (`metrics.expected_pnl`): calibration runs have no disinformation, so this
   has the realized P&L's expectation with about a sixth of its spread across seeds,
   which the outcome's luck otherwise dominates (Step 8b).
5. The price is calibrated: regressing logit(P*) on logit(price), over the second half
   of every run, gives a slope between 0.8 and 1.4. Without disinformation P* is the
   outcome's exact posterior, so it stands in for the outcome with far less noise; a
   slope above 1 means the price is underconfident (too close to 50c), below 1
   overconfident. Real prediction-market prices are roughly calibrated close to
   resolution (Page & Clemen 2013), and miss in the underconfident direction of the
   favorite-longshot bias (Burgi et al. 2025), so the band reaches further above 1.
   The informed earn only when the price leaves them an edge (Grossman & Stiglitz
   1980), so in the third Step 8b search every setting that met the other patterns
   had a slope of 1.2 to 1.3. Once relayed news counted as own evidence, 68 settings
   met them within 0.8-1.25, and the frozen one's slope is 1.10-1.18 on fresh seeds.
   Only the overall slope is fitted, so the favorite-longshot bias at the extremes
   stays an out-of-sample check.
6. The price beats the agents: its Brier score against the outcome is lower than the
   average agent's own judgment. Each agent is scored on its own-evidence posterior
   (no market term, no depth signal, no noise); one who knows no clue at all scores
   0.5, which is rare by the end of a run because news and hearsay reach nearly every
   agent. Under the Step 3 averaging belief rule the price was barely better, so this
   was reported rather than required until Step 8b (PLAN.md).

`measure` reads these off finished runs; `targets` loads the real values fetched by
`data/fetch_kalshi_patterns.py`; `evaluate` scores a run set against them. The second
half of the module samples and selects candidates for `experiments/calibrate.py`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kalshi_model.agents import sigmoid
from kalshi_model.metrics import expected_pnl
from kalshi_model.model import MarketModel

PATTERNS_FILE = Path(__file__).resolve().parents[2] / "data" / "kalshi_patterns.json"
SIZE_QUANTILES = ("q25", "q50", "q75", "q90")
REQUIRED = ("spread", "sizes", "book_filled", "informed_earn", "calibrated", "beats_agents")
"""Patterns a calibration must meet."""
MAX_EMPTY_SHARE = 0.02
"""Most steps that may end with one side of the book empty."""
SIZE_TOLERANCE = math.log(2)
"""Accepted mean absolute log ratio of fill-size quantiles: within a factor of 2."""
CALIBRATION_BAND = (0.8, 1.4)
"""Accepted slope of logit(P*) on logit(price)."""
CALIBRATION_STRIDE = 50
"""Steps between the (price, P*) pairs taken from the second half of each run."""


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
    pairs = []
    for model in models:
        df = model.datacollector.get_model_vars_dataframe()
        late = df.iloc[len(df) // 2 :: CALIBRATION_STRIDE]
        pairs += list(zip(late.price, late.p_star, strict=True))
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
        pnl = expected_pnl(model)
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
        "calibration_slope": calibration_slope(pairs),
    }


def calibration_slope(pairs: list[tuple[float, float]]) -> float:
    """Least-squares slope of logit(P*) on logit(price) over (price, P*) pairs."""
    both = np.clip(np.array(pairs, dtype=float), 1e-4, 1 - 1e-4)
    logits = np.log(both / (1 - both))
    x, y = logits[:, 0], logits[:, 1]
    return float(np.polyfit(x, y, 1)[0])


def evaluate(patterns: dict, target: dict) -> dict:
    """Which patterns are met, and whether every required one is.

    `size_error`, `spread_error`, and their sum `distance` (0 is a perfect fit) describe
    how closely a setting matches Kalshi's trade sizes and spread; they are reported in
    results/calibration.csv but not used for acceptance or selection. Acceptance uses
    the `met` flags of the required patterns, and the frozen pick is the most central
    accepted candidate.
    """
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
        "calibrated": CALIBRATION_BAND[0] <= patterns["calibration_slope"] <= CALIBRATION_BAND[1],
    }
    return {
        "met": met,
        "accepted": all(met[k] for k in REQUIRED),
        "size_error": size_error,
        "spread_error": spread_error,
        "distance": size_error + spread_error,
    }


# Searching the parameter space (experiments/calibrate.py)


@dataclass(frozen=True)
class Dimension:
    """One searched setting: its range, whether it is searched on a log scale, and
    whether it is a whole number."""

    name: str
    low: float
    high: float
    log: bool = False
    integer: bool = False

    def value(self, u: float) -> float | int:
        """Map u in [0, 1] onto the range."""
        if self.log:
            x = math.exp(math.log(self.low) + u * (math.log(self.high) - math.log(self.low)))
        else:
            x = self.low + u * (self.high - self.low)
        return round(x) if self.integer else x


SPACE = (
    Dimension("cash_median", 20_000, 1_000_000, log=True, integer=True),
    Dimension("cash_sigma", 0.5, 2.5),
    Dimension("cash_rho", 0.0, 0.95),
    Dimension("kelly_frac", 0.05, 0.5),
    Dimension("noise", 0.01, 1.0, log=True),
    Dimension("min_edge", 0.25, 4.0, log=True),
    Dimension("patience", 0.05, 0.95),
    Dimension("market_trust", 2.0, 120.0, log=True),
    Dimension("mm_size", 3, 100, log=True, integer=True),
    Dimension("mm_half_spread", 0.5, 4.0),
)
"""The calibrated settings and their prior ranges. Since Step 8c the cash distribution
and its link to informedness are the only searched trait distributions; the other
searched traits are single population values, with ranges covering what the Step 8b
distributions spanned on average. Not searched: g (lab estimates), informedness
(Step 3), and w_hearsay and w_depth (no data anchor)."""


def config_overrides(u: np.ndarray) -> dict:
    """Config fields for one point u in the unit cube (one coordinate per SPACE entry)."""
    return {d.name: d.value(float(x)) for d, x in zip(SPACE, u, strict=True)}


def latin_hypercube(n: int, dims: int, rng: np.random.Generator) -> np.ndarray:
    """n points in the unit cube with exactly one point in each of n slices per axis."""
    slices = np.stack([rng.permutation(n) for _ in range(dims)], axis=1)
    return (slices + rng.random((n, dims))) / n


def central_order(points: np.ndarray) -> np.ndarray:
    """Indices of `points` from nearest to farthest from their coordinate-wise median."""
    distance = np.linalg.norm(points - np.median(points, axis=0), axis=1)
    return np.argsort(distance, kind="stable")


def spread_out(points: np.ndarray, start: int, k: int) -> list[int]:
    """k indices chosen by farthest-point sampling from `start`: each new point is the one
    farthest from all points chosen so far, so the set covers the region."""
    chosen = [start]
    nearest = np.linalg.norm(points - points[start], axis=1)
    while len(chosen) < min(k, len(points)):
        nxt = int(np.argmax(nearest))
        chosen.append(nxt)
        nearest = np.minimum(nearest, np.linalg.norm(points - points[nxt], axis=1))
    return chosen
