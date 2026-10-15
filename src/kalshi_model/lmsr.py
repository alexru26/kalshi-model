"""Hanson's logarithmic market scoring rule (LMSR) as the AMM's market maker.

The maker's state is `net`, the number of YES contracts it has sold net of NO (positive
when agents hold YES on balance). Its cost function is `C(n) = b ln(1 + e^(n / b))`, the
two-outcome LMSR with q_yes - q_no = n, so the YES price is `sigmoid(n / b)`. Contracts
pay 100c, so costs in cents are `100 x (C(n') - C(n))`. Money moves in integer cents,
rounded in the maker's favor, so its loss stays within `100 b ln 2` cents.
"""

import math
from operator import index

from kalshi_model.world import sigmoid

# Guards ceil/floor against float error on costs that are exact integers in theory.
_ROUNDING_TOLERANCE = 1e-9


def lmsr_cost(net: float, b: float) -> float:
    """C(n) = b ln(1 + e^(n / b)) in contracts, without overflow for large |n|."""
    x = net / b
    if x > 0:
        return net + b * math.log1p(math.exp(-x))
    return b * math.log1p(math.exp(x))


def lmsr_price(net: float, b: float) -> float:
    """Marginal YES price in (0, 1)."""
    return sigmoid(net / b)


def subsidy_cents(b: float) -> int:
    """Worst-case maker loss, 100 b ln 2 cents, rounded up: the maker's starting cash."""
    return math.ceil(100 * b * math.log(2))


class LMSR:
    """An LMSR market maker that trades whole contracts for integer cents."""

    def __init__(self, b: float) -> None:
        if not b > 0:
            raise ValueError(f"b must be positive, got {b!r}")
        self.b = float(b)
        self.net = 0
        self.cash = subsidy_cents(self.b)

    @property
    def price(self) -> float:
        return lmsr_price(self.net, self.b)

    def quote(self, contracts: int) -> int:
        """Cents an agent pays to buy `contracts` YES (negative `contracts` sells YES).

        A buy costs `ceil(100 (C(n + k) - C(n)))`; a sale pays out
        `floor(100 (C(n) - C(n - k)))`, returned as a negative number.
        """
        contracts = index(contracts)
        exact = 100 * (lmsr_cost(self.net + contracts, self.b) - lmsr_cost(self.net, self.b))
        if exact >= 0:
            return math.ceil(exact - _ROUNDING_TOLERANCE)
        return -math.floor(-exact + _ROUNDING_TOLERANCE)

    def trade(self, contracts: int) -> int:
        """Sell `contracts` YES to an agent (buy them back if negative); return its cost."""
        cost = self.quote(contracts)
        self.net += contracts
        self.cash += cost
        return cost

    def max_buy_below(self, limit: float) -> int:
        """Most YES the maker can sell before its price would rise above `limit`."""
        return max(0, math.floor(self.b * _logit(limit) - self.net))

    def max_sell_above(self, limit: float) -> int:
        """Most YES the maker can buy back before its price would fall below `limit`."""
        return max(0, math.floor(self.net - self.b * _logit(limit)))

    def settle(self, y: bool) -> int:
        """Pay out 100c per net YES sold if YES wins, then close out; return the cents paid."""
        payout = 100 * self.net if y else 0
        self.cash -= payout
        self.net = 0
        return payout


def _logit(p: float) -> float:
    if not 0 < p < 1:
        raise ValueError(f"limit must be strictly between 0 and 1, got {p!r}")
    return math.log(p / (1 - p))
