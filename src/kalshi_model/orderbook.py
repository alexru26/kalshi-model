"""A central limit order book for one YES contract.

Plain Python, no Mesa. Money is integer cents, prices are integer cents from 1 to 99,
and positions are signed YES counts: buying NO at x is selling YES at 100 - x.

Rules:
- Price-time priority; an incoming order trades at the resting order's price, and
  partial fills keep their place in the queue.
- Every account is fully collateralized at all times. Settlement wealth is
  `cash` if NO wins and `cash + 100 x position` if YES wins. Bids can only lower the
  first and asks the second, so each account keeps two reserves: its open bids' full
  cost plus worst-case fee (`bid_reserve`, covered by cash) and its open asks' 100 - p
  per contract plus worst-case fee (`ask_reserve`, covered by cash + 100 x position).
  An order that would break either cover is rejected whole.
- Fees follow Kalshi's per-order accumulator: each order adds every fill's exact fee to
  a running total and pays up to its ceiling in cents, so an order filled in pieces
  pays what one equivalent fill would. Takers pay the taker rate, makers the maker rate.
- Cancellation is free and releases the reserve at once.
- Self-trade prevention: an incoming order cancels the owner's own resting orders it
  would otherwise trade against.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable
from dataclasses import dataclass, field
from operator import index

from kalshi_model.fees import ceil_cents, exact_fee

BUY = "buy"
"""Bid for YES (equivalently, offer NO)."""
SELL = "sell"
"""Offer YES (equivalently, bid for NO)."""
SIDES = (BUY, SELL)


class OrderRejected(ValueError):
    """The order breaks a market rule (collateral or post-only) and was not placed."""


class InvariantError(AssertionError):
    """The book's accounting is inconsistent: a bug, never a market outcome."""


@dataclass
class Account:
    """One trader's money in the book: cash and position, plus the collateral its open
    orders hold (`bid_reserve` against cash, `ask_reserve` against YES-wins wealth)."""

    cash: int
    position: int = 0
    bid_reserve: int = 0
    ask_reserve: int = 0
    deposited: int = 0
    """Starting cash, for the conservation check."""

    @property
    def free_cash(self) -> int:
        """Cash not reserved for open bids: what a new bid can use."""
        return self.cash - self.bid_reserve

    @property
    def free_yes_cover(self) -> int:
        """Cover left for new asks: settlement wealth if YES wins, less open asks' risk."""
        return self.cash + 100 * self.position - self.ask_reserve


@dataclass
class Order:
    """A limit order; while it rests, `remaining` is its unfilled size."""

    order_id: int
    owner: Hashable
    side: str
    price: int
    remaining: int
    fee_micro: int = 0
    """Exact fees accrued over all of this order's fills, in millionths of a cent."""
    fee_paid: int = 0
    """Fees charged so far: always ceil_cents(fee_micro)."""
    reserve: int = 0
    """Cents of the owner's bid or ask reserve held for the unfilled part."""


@dataclass(frozen=True)
class Fill:
    """One trade between an incoming (taker) and a resting (maker) order, at the maker's
    price. `taker_side` is the taker's side; the maker was on the other."""

    price: int
    qty: int
    taker: Hashable
    maker: Hashable
    taker_side: str
    taker_fee: int
    maker_fee: int
    taker_order: int
    maker_order: int


@dataclass
class Submission:
    """What `OrderBook.submit` did: the fills, any self-cancelled orders, and the id of
    the part left resting."""

    order_id: int | None
    """Id of the resting remainder, or None if nothing rests."""
    fills: list[Fill] = field(default_factory=list)
    self_cancelled: list[int] = field(default_factory=list)


@dataclass
class _Plan:
    """An order's checked outcome, worked out before anything changes: the resting
    orders it fills (and how much of each), its owner's orders it cancels, and what
    is left over."""

    price: int
    qty: int
    matches: list[tuple[Order, int]]
    self_cancels: list[Order]
    remaining: int


class OrderBook:
    """The book for one YES contract, and the ledger of every account that trades in it.

    Orders rest in price levels, each a first-in-first-out queue, which gives price-time
    priority. Every order is planned and checked in full (`_plan`) before anything
    changes, so a rejected order leaves the book exactly as it was.
    """

    def __init__(self, taker_fee_bp: int = 0, maker_fee_bp: int = 0) -> None:
        self.taker_fee_bp = index(taker_fee_bp)
        self.maker_fee_bp = index(maker_fee_bp)
        if self.taker_fee_bp < 0 or self.maker_fee_bp < 0:
            raise ValueError("fee rates must be non-negative")
        self._accounts: dict[Hashable, Account] = {}
        # side -> price -> resting orders at that price, oldest first.
        self._levels: dict[str, dict[int, OrderedDict[int, Order]]] = {BUY: {}, SELL: {}}
        self._orders: dict[int, Order] = {}
        self._by_owner: dict[Hashable, dict[int, Order]] = {}
        self._last_id = 0
        self.fills: list[Fill] = []
        """Every fill so far, in order."""
        self.fees_collected = 0
        """Fees charged so far, in cents (they leave the accounts but stay counted)."""
        self.last_price: int | None = None
        """Price of the last trade, in cents."""
        self.settled = False

    # Accounts

    def open_account(self, owner: Hashable, cash: int) -> Account:
        """Open `owner`'s account with `cash` cents deposited and no position."""
        cash = index(cash)
        if owner in self._accounts:
            raise ValueError(f"account {owner!r} already exists")
        if cash < 0:
            raise ValueError("starting cash must be non-negative")
        account = Account(cash=cash, deposited=cash)
        self._accounts[owner] = account
        return account

    def account(self, owner: Hashable) -> Account:
        return self._accounts[owner]

    @property
    def accounts(self) -> dict[Hashable, Account]:
        return self._accounts

    # Orders

    def submit(
        self,
        owner: Hashable,
        side: str,
        price: int,
        qty: int,
        *,
        ioc: bool = False,
        post_only: bool = False,
    ) -> Submission:
        """Place a limit order; raise OrderRejected if it breaks a rule.

        `ioc` (immediate or cancel) drops whatever does not trade at once; `post_only`
        rejects the order if any of it would trade at once.
        """
        plan = self._plan(owner, side, price, qty, ioc=ioc, post_only=post_only)
        # The plan passed every check; now carry it out.
        result = Submission(order_id=None)
        for order in plan.self_cancels:
            self._remove(order)
            result.self_cancelled.append(order.order_id)

        self._last_id += 1
        taker = Order(self._last_id, owner, side, plan.price, plan.qty)
        for maker, fill_qty in plan.matches:
            result.fills.append(self._execute(taker, maker, fill_qty))
        if plan.remaining and not ioc:
            taker.remaining = plan.remaining
            self._rest(taker)
            result.order_id = taker.order_id
        return result

    def can_submit(
        self,
        owner: Hashable,
        side: str,
        price: int,
        qty: int,
        *,
        ioc: bool = False,
        post_only: bool = False,
    ) -> bool:
        """Whether `submit` with the same arguments would be accepted. Changes nothing."""
        try:
            self._plan(owner, side, price, qty, ioc=ioc, post_only=post_only)
        except OrderRejected:
            return False
        return True

    def cancel(self, order_id: int) -> bool:
        """Cancel a resting order for free. Returns False if it is no longer resting."""
        order = self._orders.get(order_id)
        if order is None:
            return False
        self._remove(order)
        return True

    def cancel_all(self, owner: Hashable) -> int:
        """Cancel every open order of `owner`; return how many were cancelled."""
        orders = self.open_orders(owner)
        for order in orders:
            self._remove(order)
        return len(orders)

    def cancel_everything(self) -> int:
        """Cancel every open order in the book."""
        orders = list(self._orders.values())
        for order in orders:
            self._remove(order)
        return len(orders)

    def open_orders(self, owner: Hashable) -> list[Order]:
        """`owner`'s resting orders, oldest first."""
        return list(self._by_owner.get(owner, {}).values())

    # Market data

    @property
    def best_bid(self) -> int | None:
        return max(self._levels[BUY], default=None)

    @property
    def best_ask(self) -> int | None:
        return min(self._levels[SELL], default=None)

    @property
    def spread(self) -> int | None:
        bid, ask = self.best_bid, self.best_ask
        return None if bid is None or ask is None else ask - bid

    @property
    def midpoint(self) -> float | None:
        bid, ask = self.best_bid, self.best_ask
        return None if bid is None or ask is None else (bid + ask) / 2

    def depth(self, side: str, window: int) -> int:
        """Contracts resting on `side` within `window` cents of that side's best price."""
        levels = self._levels[side]
        if not levels:
            return 0
        best = max(levels) if side == BUY else min(levels)
        return sum(
            sum(o.remaining for o in queue.values())
            for price, queue in levels.items()
            if abs(price - best) <= window
        )

    def imbalance(self, window: int) -> float:
        """(bid depth - ask depth) / total depth near the best prices; 0 if both are empty."""
        bids, asks = self.depth(BUY, window), self.depth(SELL, window)
        total = bids + asks
        return 0.0 if total == 0 else (bids - asks) / total

    def reference_price(self, spread_cap: int) -> float | None:
        """The price agents read and metrics score, as a probability.

        The midpoint when both sides are quoted within `spread_cap` cents, else the last
        trade, else None (nothing to read yet).
        """
        spread = self.spread
        if spread is not None and spread <= spread_cap:
            return self.midpoint / 100
        if self.last_price is not None:
            return self.last_price / 100
        return None

    # Settlement

    def settle(self, y: bool) -> None:
        """Cancel every open order, then pay 100c per YES held if YES wins."""
        self.cancel_everything()
        for account in self._accounts.values():
            if y:
                account.cash += 100 * account.position
            account.position = 0
        self.settled = True

    # Invariants

    def check_invariants(self) -> None:
        """Raise InvariantError unless contracts, money, and reserves all add up."""
        positions = sum(a.position for a in self._accounts.values())
        if positions != 0:
            raise InvariantError(f"positions sum to {positions}, not 0")
        money = sum(a.cash for a in self._accounts.values()) + self.fees_collected
        deposited = sum(a.deposited for a in self._accounts.values())
        if money != deposited:
            raise InvariantError(f"cash + fees = {money}, but {deposited} was deposited")
        expected = {owner: [0, 0] for owner in self._accounts}
        for order in self._orders.values():
            if order.remaining <= 0:
                raise InvariantError(f"order {order.order_id} rests with nothing left")
            if order.reserve != self._reserve_for(order, order.remaining):
                raise InvariantError(f"order {order.order_id} holds the wrong reserve")
            expected[order.owner][order.side == SELL] += order.reserve
        for owner, account in self._accounts.items():
            if [account.bid_reserve, account.ask_reserve] != expected[owner]:
                raise InvariantError(f"account {owner!r} reserves do not match its orders")
            if account.free_cash < 0 or account.free_yes_cover < 0:
                raise InvariantError(f"account {owner!r} is not fully collateralized")
        indexed = {oid for orders in self._by_owner.values() for oid in orders}
        if indexed != set(self._orders):
            raise InvariantError("the per-owner order index does not match the book")
        bid, ask = self.best_bid, self.best_ask
        if bid is not None and ask is not None and bid >= ask:
            raise InvariantError(f"the book is crossed: bid {bid} >= ask {ask}")
        for side in SIDES:
            if any(not queue for queue in self._levels[side].values()):
                raise InvariantError("an empty price level was left in the book")

    # Internals

    def _plan(
        self, owner: Hashable, side: str, price: int, qty: int, *, ioc: bool, post_only: bool
    ) -> _Plan:
        """Work out what an order would do and check it, without changing anything."""
        if self.settled:
            raise RuntimeError("the market has settled")
        if side not in SIDES:
            raise ValueError(f"side must be {BUY!r} or {SELL!r}, got {side!r}")
        price = _whole_number("price", price)
        qty = _whole_number("qty", qty)
        if not 1 <= price <= 99:
            raise ValueError(f"price must be between 1 and 99 cents, got {price}")
        if qty < 1:
            raise ValueError(f"qty must be at least 1, got {qty}")
        account = self._accounts[owner]

        # Walk the opposite side from its best price, in time order within each level.
        opposite = SELL if side == BUY else BUY
        levels = self._levels[opposite]
        crosses = (lambda p: p <= price) if side == BUY else (lambda p: p >= price)
        matches: list[tuple[Order, int]] = []
        self_cancels: list[Order] = []
        remaining = qty
        for level_price in sorted(levels, reverse=side == SELL):
            if remaining == 0 or not crosses(level_price):
                break
            for order in levels[level_price].values():
                if remaining == 0:
                    break
                if order.owner == owner:
                    self_cancels.append(order)
                    continue
                fill_qty = min(remaining, order.remaining)
                matches.append((order, fill_qty))
                remaining -= fill_qty
        if post_only and matches:
            raise OrderRejected("post-only order would trade on arrival")

        # Settlement wealth after the order: both covers must stay non-negative.
        fee_micro = sum(exact_fee(q, o.price, self.taker_fee_bp) for o, q in matches)
        traded_qty = qty - remaining
        traded_cost = sum(q * o.price for o, q in matches)
        sign = 1 if side == BUY else -1
        cash = account.cash - sign * traded_cost - ceil_cents(fee_micro)
        position = account.position + sign * traded_qty
        released = [0, 0]
        for order in self_cancels:
            released[order.side == SELL] += order.reserve
        bid_reserve = account.bid_reserve - released[0]
        ask_reserve = account.ask_reserve - released[1]
        if remaining and not ioc:
            resting = Order(0, owner, side, price, remaining, fee_micro, ceil_cents(fee_micro))
            reserve = self._reserve_for(resting, remaining)
            if side == BUY:
                bid_reserve += reserve
            else:
                ask_reserve += reserve
        if cash - bid_reserve < 0 or cash + 100 * position - ask_reserve < 0:
            raise OrderRejected("insufficient collateral")
        return _Plan(price, qty, matches, self_cancels, remaining)

    def _reserve_for(self, order: Order, qty: int) -> int:
        """Worst-case cost of `qty` more maker fills: price risk plus the fee left to pay."""
        risk = order.price if order.side == BUY else 100 - order.price
        future_fee = 0
        if qty and self.maker_fee_bp:
            total = order.fee_micro + exact_fee(qty, order.price, self.maker_fee_bp)
            future_fee = ceil_cents(total) - order.fee_paid
        return qty * risk + future_fee

    def _charge(self, order: Order, micro: int) -> int:
        """Add a fill's exact fee to the order's running total; return the cents now due
        (Kalshi's per-order fee accumulator)."""
        order.fee_micro += micro
        due = ceil_cents(order.fee_micro) - order.fee_paid
        order.fee_paid += due
        return due

    def _execute(self, taker: Order, maker: Order, qty: int) -> Fill:
        """Trade `qty` at the maker's price: charge fees, shrink the maker's reserve,
        move cash and contracts, and record the fill."""
        price = maker.price
        taker_fee = self._charge(taker, exact_fee(qty, price, self.taker_fee_bp))
        maker_fee = 0
        if self.maker_fee_bp:
            maker_fee = self._charge(maker, exact_fee(qty, price, self.maker_fee_bp))

        maker_account = self._accounts[maker.owner]
        old_reserve = maker.reserve
        maker.remaining -= qty
        maker.reserve = self._reserve_for(maker, maker.remaining)
        if maker.side == BUY:
            maker_account.bid_reserve -= old_reserve - maker.reserve
        else:
            maker_account.ask_reserve -= old_reserve - maker.reserve

        buyer, seller = (taker, maker) if taker.side == BUY else (maker, taker)
        buyer_account = self._accounts[buyer.owner]
        seller_account = self._accounts[seller.owner]
        buyer_account.cash -= qty * price
        buyer_account.position += qty
        seller_account.cash += qty * price
        seller_account.position -= qty
        self._accounts[taker.owner].cash -= taker_fee
        maker_account.cash -= maker_fee
        self.fees_collected += taker_fee + maker_fee

        if maker.remaining == 0:
            self._unlink(maker)
        self.last_price = price
        fill = Fill(
            price=price,
            qty=qty,
            taker=taker.owner,
            maker=maker.owner,
            taker_side=taker.side,
            taker_fee=taker_fee,
            maker_fee=maker_fee,
            taker_order=taker.order_id,
            maker_order=maker.order_id,
        )
        self.fills.append(fill)
        return fill

    def _rest(self, order: Order) -> None:
        """Put an order's unfilled part at the back of its price level and hold its reserve."""
        order.reserve = self._reserve_for(order, order.remaining)
        account = self._accounts[order.owner]
        if order.side == BUY:
            account.bid_reserve += order.reserve
        else:
            account.ask_reserve += order.reserve
        self._levels[order.side].setdefault(order.price, OrderedDict())[order.order_id] = order
        self._orders[order.order_id] = order
        self._by_owner.setdefault(order.owner, {})[order.order_id] = order

    def _remove(self, order: Order) -> None:
        """Cancel a resting order: release its reserve and take it out of the book."""
        account = self._accounts[order.owner]
        if order.side == BUY:
            account.bid_reserve -= order.reserve
        else:
            account.ask_reserve -= order.reserve
        order.reserve = 0
        self._unlink(order)

    def _unlink(self, order: Order) -> None:
        """Drop an order from its price level and the indexes (reserve already settled)."""
        queue = self._levels[order.side][order.price]
        del queue[order.order_id]
        if not queue:
            del self._levels[order.side][order.price]
        del self._orders[order.order_id]
        del self._by_owner[order.owner][order.order_id]


def _whole_number(name: str, value: object) -> int:
    """Accept any integer type (NumPy integers included) as a plain int; reject the rest.

    Floats such as 50.0 are rejected rather than truncated, and so are booleans.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    try:
        return index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {value!r}") from None
