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

from kalshi_model.fees import MICRO_PER_CENT, exact_fee
from kalshi_model.orderbook import BUY, SELL, Account, OrderBook
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
    """logit(b) = L_own + w_h L_heard + w_m logit(mid) + w_depth imb + noise z.

    The price counts as evidence from other traders' clues, added to the agent's own.
    `w_m = market_trust / (market_trust + n_eff + eps)` with `n_eff = n_own + w_h n_heard`
    discounts it by the share of the market's information the agent thinks it holds,
    and `w_m = 0` when there is no reference price (PLAN.md Section 2.2).
    """
    market_term = 0.0
    if reference_price is not None:
        n_eff = n_own + w_hearsay * n_heard
        w_market = market_trust / (market_trust + n_eff + EPS)
        market_term = w_market * logit(reference_price)
    return l_own + w_hearsay * l_heard + market_term + w_depth * imbalance + noise * z


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

        self.account = Account(cash=self.initial_cash, deposited=self.initial_cash)
        """Cash and position. In a CLOB run the model swaps in the order book's account."""
        self.pending: set[int] = set()
        """News or fake clues learned since the agent's last turn: not relayed yet."""
        self.spreading: set[int] = set()
        """Clues the agent is retelling to its neighbors (until it becomes a stifler)."""
        self._stale_since: dict[int, int] = {}
        """Resting order id -> the step on which the agent first saw it stale."""
        self.known: set[int] = set()
        """Ids of every clue the agent knows, so each counts once."""
        self.own_net = 0
        """# YES - # NO among clues seen directly (private clues and news as a seed)."""
        self.n_own = 0
        self.heard_net = 0
        self.n_heard = 0

    @property
    def cash(self) -> int:
        return self.account.cash

    @cash.setter
    def cash(self, value: int) -> None:
        self.account.cash = value

    @property
    def position(self) -> int:
        return self.account.position

    @position.setter
    def position(self, value: int) -> None:
        self.account.position = value

    def learn_own(self, clue_id: int, points_yes: bool, *, spreadable: bool = False) -> bool:
        """See a clue directly; return False if it was already known (then it is ignored).

        Private clues are not `spreadable`; news and fakes delivered to a seed are, and
        the agent relays them after its next turn.
        """
        if clue_id in self.known:
            return False
        self.known.add(clue_id)
        self.n_own += 1
        self.own_net += 1 if points_yes else -1
        if spreadable:
            self.pending.add(clue_id)
        return True

    def learn_heard(self, clue_id: int, points_yes: bool) -> bool:
        """Hear a clue from a neighbor; return False if it was already known (ignored)."""
        if clue_id in self.known:
            return False
        self.known.add(clue_id)
        self.n_heard += 1
        self.heard_net += 1 if points_yes else -1
        self.pending.add(clue_id)
        return True

    def finish_turn(self) -> None:
        """After trading, the agent starts retelling what it learned since its last turn."""
        self.spreading |= self.pending
        self.pending.clear()

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

    def take_turn_clob(self, book: OrderBook, z: float, u: float, step: int) -> int:
        """One turn in the order book; return contracts taken (bought > 0, sold < 0).

        The agent updates its belief, then tends its resting orders (cancelling those
        no longer worth their edge or stuck behind the best quote), then takes any quote
        that beats its value by more than `min_edge` after the taker fee. If it took
        nothing, with probability `patience` (`u < patience`) it posts a resting order
        priced `min_edge` inside its value, sized to the gap to its Kelly target.
        """
        config = self.model.config
        reference = book.reference_price(config.spread_cap)
        imbalance = book.imbalance(config.depth_window)
        value = 100 * perceived_yes(self.belief(reference, imbalance, z), self.g)
        self._tend_orders(book, value, step)
        taken = self._take(book, value)
        if taken == 0 and u < self.patience:
            self._post(book, value, reference)
        return taken

    def _fee_per_contract(self, price: int, rate_bp: int) -> float:
        return exact_fee(1, price, rate_bp) / MICRO_PER_CENT

    def _tend_orders(self, book: OrderBook, value: float, step: int) -> None:
        """Cancel orders whose edge fell below `min_edge` or that sat stale too long."""
        config = self.model.config
        open_ids = set()
        for order in book.open_orders(self.index):
            maker_fee = self._fee_per_contract(order.price, book.maker_fee_bp)
            if order.side == BUY:
                edge = value - order.price - maker_fee
                behind = book.best_bid - order.price
            else:
                edge = order.price - value - maker_fee
                behind = order.price - book.best_ask
            if edge < self.min_edge:
                book.cancel(order.order_id)
                continue
            if behind >= config.stale_ticks:
                since = self._stale_since.setdefault(order.order_id, step)
                if step - since >= config.stale_steps:
                    book.cancel(order.order_id)
                    continue
            else:
                self._stale_since.pop(order.order_id, None)
            open_ids.add(order.order_id)
        for order_id in set(self._stale_since) - open_ids:
            del self._stale_since[order_id]

    def _take(self, book: OrderBook, value: float) -> int:
        """Buy or sell at once (immediate-or-cancel) if the best quote beats `min_edge`."""
        rate = book.taker_fee_bp
        best_ask, best_bid = book.best_ask, book.best_bid
        if best_ask is not None and value - best_ask - self._fee_per_contract(best_ask, rate) > (
            self.min_edge
        ):
            cost = (best_ask + self._fee_per_contract(best_ask, rate)) / 100
            target = kelly_contracts(value / 100, cost, self.kelly_frac, self.cash)
            wanted = target - self.position
            # Highest price still worth paying, walking up the book.
            limit = max(
                x
                for x in range(best_ask, 100)
                if value - x - self._fee_per_contract(x, rate) > self.min_edge
            )
            return self._submit_largest(book, BUY, limit, wanted, ioc=True)
        if best_bid is not None and best_bid - value - self._fee_per_contract(best_bid, rate) > (
            self.min_edge
        ):
            cost = (100 - best_bid + self._fee_per_contract(best_bid, rate)) / 100
            target = -kelly_contracts(1 - value / 100, cost, self.kelly_frac, self.cash)
            wanted = self.position - target
            limit = min(
                x
                for x in range(1, best_bid + 1)
                if x - value - self._fee_per_contract(x, rate) > self.min_edge
            )
            return -self._submit_largest(book, SELL, limit, wanted, ioc=True)
        return 0

    def _post(self, book: OrderBook, value: float, reference: float | None) -> None:
        """Rest an order `min_edge` (plus maker fee) inside value, never crossing the book."""
        rate = book.maker_fee_bp
        buy = value >= (50.0 if reference is None else 100 * reference)
        resting = sum(
            o.remaining for o in book.open_orders(self.index) if o.side == (BUY if buy else SELL)
        )
        if buy:
            price = math.floor(value - self.min_edge)
            while (
                price >= 1 and value - price - self._fee_per_contract(price, rate) < self.min_edge
            ):
                price -= 1
            if book.best_ask is not None:
                price = min(price, book.best_ask - 1)
            if not 1 <= price <= 99:
                return
            cost = (price + self._fee_per_contract(price, rate)) / 100
            target = kelly_contracts(value / 100, cost, self.kelly_frac, self.cash)
            self._submit_largest(book, BUY, price, target - self.position - resting, ioc=False)
        else:
            price = math.ceil(value + self.min_edge)
            while (
                price <= 99 and price - value - self._fee_per_contract(price, rate) < self.min_edge
            ):
                price += 1
            if book.best_bid is not None:
                price = max(price, book.best_bid + 1)
            if not 1 <= price <= 99:
                return
            cost = (100 - price + self._fee_per_contract(price, rate)) / 100
            target = -kelly_contracts(1 - value / 100, cost, self.kelly_frac, self.cash)
            self._submit_largest(book, SELL, price, self.position - target - resting, ioc=False)

    def _submit_largest(
        self, book: OrderBook, side: str, price: int, wanted: int, *, ioc: bool
    ) -> int:
        """Submit the largest affordable order up to `wanted`; return contracts filled."""
        lo, hi = 0, max(0, wanted)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if book.can_submit(self.index, side, price, mid, ioc=ioc):
                lo = mid
            else:
                hi = mid - 1
        if lo == 0:
            return 0
        fills = book.submit(self.index, side, price, lo, ioc=ioc).fills
        return sum(f.qty for f in fills)

    def settle(self, y: bool) -> None:
        """AMM settlement: receive 100c per YES held if YES wins (pay it per YES short).

        In a CLOB run the order book settles the agent's account instead.
        """
        if y:
            self.cash += 100 * self.position
        self.position = 0

    @property
    def pnl(self) -> int:
        """Cash gained over the run, in cents (final once the market has settled)."""
        return self.cash - self.initial_cash
