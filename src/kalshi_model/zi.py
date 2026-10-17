"""Zero-intelligence traders in the order book: the Othman (2008) sanity check.

Othman's agents hold beliefs drawn from Beta(mean x S, (1 - mean) x S), where S sets how
diffuse beliefs are (5 low accuracy, 25 middle, 100 high). Each call, a random agent bids
for one share uniformly below its belief or offers one uniformly above it, never reading
the market. Trades happen at the price of the order placed first, and the market
restarts (all orders cancelled) after each trade. Othman derives the expected trade
price in closed form and finds a favorite-longshot bias: a 0.8 mean belief trades at
0.742, 0.766, and 0.781 for S = 5, 25, 100.

His expected price integrates over the belief distribution, so each order here comes
from a fresh belief draw (an unlimited pool of agents); a fixed pool of 1,000 would
shift the mean belief by up to half a cent at S = 5. Prices here are whole cents, so
bids are rounded down and offers up to stay on the agent's side of its belief; the
discreteness moves the mean trade price by about a tenth of a cent.
"""

import math

import numpy as np

from kalshi_model.orderbook import BUY, SELL, OrderBook


def beta_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    log_norm = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    return np.exp(log_norm + (a - 1) * np.log(x) + (b - 1) * np.log1p(-x))


def othman_expected_price(mean: float, concentration: float, grid: int = 4001) -> float:
    """Othman's expected trade price, E(T) = integral of (1 - F_B) / (F_A + 1 - F_B).

    F_A(x) = integral_0^x (x - b) / (1 - b) f(b) db is the chance an offer is below x and
    1 - F_B(x) = integral_x^1 (b - x) / b f(b) db the chance a bid is above x.
    """
    # Midpoint grid on (0, 1) avoids the Beta density's endpoints.
    b = (np.arange(grid) + 0.5) / grid
    weight = beta_pdf(b, mean * concentration, (1 - mean) * concentration) / grid
    x = b[:, None]
    offer_below = np.sum(np.where(b <= x, (x - b) / (1 - b), 0.0) * weight, axis=1)
    bid_above = np.sum(np.where(b >= x, (b - x) / b, 0.0) * weight, axis=1)
    integrand = bid_above / (offer_below + bid_above)
    return float(np.mean(integrand))


def simulate_zi_market(
    mean: float,
    concentration: float,
    n_trades: int,
    rng: np.random.Generator,
    *,
    n_accounts: int = 1000,
    check_invariants: bool = False,
) -> np.ndarray:
    """Run Othman's ZI market until `n_trades` trades; return their prices in cents.

    Orders cycle through `n_accounts` accounts, far more than the orders placed between
    restarts, so no agent meets its own order.
    """
    a, b = mean * concentration, (1 - mean) * concentration
    book = OrderBook()
    for owner in range(n_accounts):
        book.open_account(owner, 10**12)

    prices = np.empty(n_trades, dtype=np.int64)
    trades = 0
    owner = 0
    while trades < n_trades:
        belief = rng.beta(a, b)
        if rng.random() < 0.5:
            side, price = BUY, math.floor(100 * rng.uniform(0.0, belief))
        else:
            side, price = SELL, math.ceil(100 * rng.uniform(belief, 1.0))
        if not 1 <= price <= 99:
            continue
        owner = (owner + 1) % n_accounts
        fills = book.submit(owner, side, price, 1).fills
        if check_invariants:
            book.check_invariants()
        if fills:
            prices[trades] = fills[0].price
            trades += 1
            book.cancel_everything()
    return prices
