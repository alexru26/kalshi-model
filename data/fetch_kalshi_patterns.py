"""Fetch the real Kalshi patterns that `experiments/calibrate.py` calibrates against.

Pulls open binary markets from Kalshi's public API (no key needed), keeps liquid ones
quoted near the middle, and records:

- their bid-ask spreads, in cents;
- the sizes of their recent trades (each trade is one match, like the model's fills).

Combination (multivariate) markets and block trades are skipped. The summary is saved
to `data/kalshi_patterns.json` with the fetch time, so calibration does not depend on
the live API.

Usage: uv run python data/fetch_kalshi_patterns.py [max markets]
"""

import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

API = "https://api.elections.kalshi.com/trade-api/v2"
OUT = Path(__file__).with_name("kalshi_patterns.json")
MIN_VOLUME_24H = 500  # contracts traded in the last day
MID_RANGE = (0.20, 0.80)
TRADES_PER_MARKET = 1000
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)


def get(path: str, **params: object) -> dict:
    url = f"{API}{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return json.load(response)
        except OSError:
            time.sleep(2**attempt)
    raise RuntimeError(f"could not fetch {url}")


def liquid_markets(max_markets: int) -> list[dict]:
    """Open single (non-combination) markets with real volume, quoted near the middle."""
    chosen, cursor = [], None
    while len(chosen) < max_markets:
        params = {"limit": 1000, "status": "open", "mve_filter": "exclude"}
        if cursor:
            params["cursor"] = cursor
        page = get("/markets", **params)
        for market in page["markets"]:
            bid, ask = float(market["yes_bid_dollars"]), float(market["yes_ask_dollars"])
            volume = float(market["volume_24h_fp"])
            if bid <= 0 or ask <= bid or volume < MIN_VOLUME_24H:
                continue
            if MID_RANGE[0] <= (bid + ask) / 2 <= MID_RANGE[1]:
                chosen.append(market)
        cursor = page.get("cursor")
        if not cursor:
            break
        time.sleep(0.2)
    return chosen[:max_markets]


def trade_sizes(ticker: str) -> list[float]:
    trades = get("/markets/trades", ticker=ticker, limit=TRADES_PER_MARKET)["trades"]
    return [float(t["count_fp"]) for t in trades if not t.get("is_block_trade")]


def quantiles(values: list[float]) -> dict[str, float]:
    return {f"q{int(100 * q)}": float(np.quantile(values, q)) for q in QUANTILES}


def main(max_markets: int) -> None:
    markets = liquid_markets(max_markets)
    spreads = [
        round(100 * (float(m["yes_ask_dollars"]) - float(m["yes_bid_dollars"]))) for m in markets
    ]
    sizes: list[float] = []
    for market in markets:
        sizes += trade_sizes(market["ticker"])
        time.sleep(0.1)
    summary = {
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": API,
        "filters": {
            "status": "open",
            "combination_markets": "excluded",
            "min_volume_24h_contracts": MIN_VOLUME_24H,
            "midpoint_range": MID_RANGE,
            "trades_per_market": TRADES_PER_MARKET,
            "block_trades": "excluded",
        },
        "n_markets": len(markets),
        "spread_cents": {"mean": float(np.mean(spreads)), **quantiles(spreads)},
        "n_trades": len(sizes),
        "trade_size_contracts": {
            "mean": float(np.mean(sizes)),
            "share_at_most_1": float(np.mean(np.array(sizes) <= 1)),
            **quantiles(sizes),
        },
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
