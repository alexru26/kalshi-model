"""Kalshi trading fees in integer cents.

Kalshi charges `rate x C x P x (1 - P)` dollars on C contracts at price P, rounded up
to the next cent. Rates are given in basis points (700 = 0.07) so the whole computation
stays in exact integer arithmetic.

When one order fills in pieces, Kalshi keeps a fee accumulator per order so that the
total converges to the fee of one equivalent fill. The order book does the same: it
adds each fill's exact fee (`exact_fee`, in millionths of a cent) to the order's running
total and charges the order up to `ceil_cents(running total)`.
"""

from operator import index

# A price is in cents and P x (1 - P) uses dollars, so rate_bp / 10_000 x C x
# (P / 100) x ((100 - P) / 100) x 100 cents = rate_bp x C x P x (100 - P) / 1_000_000.
MICRO_PER_CENT = 1_000_000


def exact_fee(contracts: int, price_cents: int, rate_bp: int) -> int:
    """The exact fee for `contracts` at `price_cents`, in millionths of a cent."""
    contracts = index(contracts)
    price_cents = index(price_cents)
    rate_bp = index(rate_bp)
    if contracts < 1:
        raise ValueError(f"contracts must be at least 1, got {contracts}")
    if not 1 <= price_cents <= 99:
        raise ValueError(f"price_cents must be between 1 and 99, got {price_cents}")
    if rate_bp < 0:
        raise ValueError(f"rate_bp must be non-negative, got {rate_bp}")
    return rate_bp * contracts * price_cents * (100 - price_cents)


def ceil_cents(micro_cents: int) -> int:
    """Round an amount in millionths of a cent up to whole cents."""
    return -(-micro_cents // MICRO_PER_CENT)


def fee_cents(contracts: int, price_cents: int, rate_bp: int) -> int:
    """Fee in cents for `contracts` at `price_cents`, at `rate_bp` basis points.

    The fee is charged on the whole quantity, not per contract, and rounded up to the
    cent. A zero rate (the AMM, the fee-free CLOB, and Kalshi maker orders) costs nothing.
    """
    return ceil_cents(exact_fee(contracts, price_cents, rate_bp))
