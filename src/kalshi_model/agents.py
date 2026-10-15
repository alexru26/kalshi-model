"""Traders: belief, perceived probability, Kelly sizing, and trading (PLAN.md Section 2.2).

The pure functions here hold the decision rules so they can be tested by hand; `Trader`
wires them to one agent's traits, clues, cash, and position. Money is integer cents and
positions are signed YES counts. Every agent stays fully collateralized: whichever way
the market resolves, its cash after settlement is never negative.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mesa

from kalshi_model.world import clue_log_odds, sigmoid

if TYPE_CHECKING:
    from kalshi_model.lmsr import LMSR
    from kalshi_model.model import MarketModel

# Guards the market-trust weight against 0 / 0 (PLAN.md Section 2.2).
EPS = 1e-9
# Keeps float error from flooring an exact contract count one short (499.999... -> 4).
_FLOOR_TOLERANCE = 1e-9


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def belief_log_odds(
    *,
    l_own: float,
    n_own: int,
    l_heard: float,
    n_heard: int,
    w_hearsay: float,
    market_trust: float,
    reference_price: float | None,
    w_depth: float,
    imbalance: float,
    noise: float,
    z: float,
) -> float:
    """logit(b) = (1 - w_m)(L_own + w_h L_heard) + w_m logit(mid) + w_depth imb + noise z.

    `w_m = market_trust / (market_trust + n_eff + eps)` with
    `n_eff = n_own + w_h n_heard`, and `w_m = 0` when there is no reference price.
    """
    if reference_price is None:
        w_market = 0.0
        market_term = 0.0
    else:
        n_eff = n_own + w_hearsay * n_heard
        w_market = market_trust / (market_trust + n_eff + EPS)
        market_term = w_market * logit(reference_price)
    clue_term = (1 - w_market) * (l_own + w_hearsay * l_heard)
    return clue_term + market_term + w_depth * imbalance + noise * z


def perceived_yes(belief_log_odds_value: float, g: float) -> float:
    """Symmetric linear-in-log-odds weighting: logit(p_yes) = g logit(b)."""
    return sigmoid(g * belief_log_odds_value)


def kelly_contracts(p_win: float, price: float, kelly_frac: float, wealth_cents: int) -> int:
    """Fractional-Kelly position, in contracts, on one side bought at `price` (in (0, 1)).

    Full Kelly stakes `f = (p_win - price) / (1 - price)` of wealth; the agent stakes
    `kelly_frac x f x wealth`, buys whole contracts, and takes at least one.
    """
    if not p_win > price:
        return 0
    f = (p_win - price) / (1 - price)
    stake_cents = kelly_frac * f * wealth_cents
    return max(1, math.floor(stake_cents / (100 * price) + _FLOOR_TOLERANCE))


def solvent(cash: int, position: int) -> bool:
    """Fully collateralized: settlement cash is non-negative whether YES or NO wins."""
    return cash >= 0 and cash + 100 * position >= 0


class Trader(mesa.Agent):
    """One trader. Traits are fixed; clues, cash, and position change during a run."""

    model: MarketModel

    def __init__(self, model: MarketModel, index: int) -> None:
        super().__init__(model)
        self.index = index
        traits = model.traits
        self.w_hearsay = float(traits.w_hearsay[index])
        self.market_trust = float(traits.market_trust[index])
        self.w_depth = float(traits.w_depth[index])
        self.noise = float(traits.noise[index])
        self.min_edge = float(traits.min_edge[index])
        self.patience = float(traits.patience[index])
        self.kelly_frac = float(traits.kelly_frac[index])
        self.g = float(traits.g[index])
        self.initial_cash = int(traits.cash[index])

        self.cash = self.initial_cash
        self.position = 0
        self.known: set[int] = set()
        """Ids of every clue the agent knows, so each counts once."""
        self.own_net = 0
        """# YES - # NO among clues seen directly (private clues and news as a seed)."""
        self.n_own = 0
        self.heard_net = 0
        self.n_heard = 0

    def learn_own(self, clue_id: int, points_yes: bool) -> None:
        """See a clue directly. A clue already known is ignored."""
        if clue_id in self.known:
            return
        self.known.add(clue_id)
        self.n_own += 1
        self.own_net += 1 if points_yes else -1

    def belief(self, reference_price: float | None, imbalance: float, z: float) -> float:
        """This turn's belief, in log-odds."""
        weight = clue_log_odds(self.model.world.clue_accuracy)
        return belief_log_odds(
            l_own=self.own_net * weight,
            n_own=self.n_own,
            l_heard=self.heard_net * weight,
            n_heard=self.n_heard,
            w_hearsay=self.w_hearsay,
            market_trust=self.market_trust,
            reference_price=reference_price,
            w_depth=self.w_depth,
            imbalance=imbalance,
            noise=self.noise,
            z=z,
        )

    def take_turn_amm(self, lmsr: LMSR, z: float) -> int:
        """Trade with the LMSR toward the Kelly target; return contracts bought (sold < 0).

        The agent buys YES while its perceived probability beats the price by more than
        `min_edge`, stopping at its Kelly target or once its trades push the price to
        `p_yes - min_edge`; it sells YES (buys NO) in the mirror case. Inside the
        no-trade band it holds. It only trades toward a target on the side it now
        favors, never trimming a position because its cash, and so its target, fell.
        """
        price = lmsr.price
        p_yes = perceived_yes(self.belief(price, 0.0, z), self.g)
        edge = self.min_edge / 100

        if p_yes - price > edge:
            target = kelly_contracts(p_yes, price, self.kelly_frac, self.cash)
            wanted = max(0, min(target - self.position, lmsr.max_buy_below(p_yes - edge)))
        elif price - p_yes > edge:
            target = -kelly_contracts(1 - p_yes, 1 - price, self.kelly_frac, self.cash)
            wanted = -max(0, min(self.position - target, lmsr.max_sell_above(p_yes + edge)))
        else:
            return 0

        contracts = self._largest_affordable(lmsr, wanted)
        if contracts:
            cost = lmsr.trade(contracts)
            self.cash -= cost
            self.position += contracts
        return contracts

    def _largest_affordable(self, lmsr: LMSR, wanted: int) -> int:
        """The largest trade toward `wanted` (same sign, no larger) that keeps us solvent."""
        if wanted == 0:
            return 0
        sign = 1 if wanted > 0 else -1
        lo, hi = 0, abs(wanted)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if solvent(self.cash - lmsr.quote(sign * mid), self.position + sign * mid):
                lo = mid
            else:
                hi = mid - 1
        return sign * lo

    def settle(self, y: bool) -> None:
        """Receive 100c per YES held if YES wins (pay it per YES short), then close out."""
        if y:
            self.cash += 100 * self.position
        self.position = 0

    @property
    def pnl(self) -> int:
        """Cash gained over the run, in cents (final once the market has settled)."""
        return self.cash - self.initial_cash
