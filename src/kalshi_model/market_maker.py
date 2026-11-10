"""The CLOB's market-maker component.

A model component, not an agent: it has no beliefs or clues. Each step, before agents
act, it cancels its quotes and requotes both sides:

- around a moving estimate of the price (an exponentially weighted average of trade
  prices, starting at 50c);
- skewed against its inventory, so a long position lowers both quotes to shed it
  (Avellaneda & Stoikov 2008);
- with a half-spread that widens after it loses to informed traders (Glosten &
  Milgrom 1985): each step it marks its new fills to the updated estimate, and every
  cent of markout loss per quoted contract adds `mm_widen` cents, decaying by
  `mm_decay` per step.

Quotes are post-only: a quote that would cross an agent's resting order is moved one
cent inside it, so the market maker never takes liquidity or pays taker fees. Past
`mm_max_inventory` contracts long (short) it never goes: each quote that adds to its
inventory is sized to the room left under the cap. It trades from its own order-book
account, so it is held to the same collateral rules as agents.
"""

from __future__ import annotations

import math

from kalshi_model.config import Config
from kalshi_model.orderbook import BUY, SELL, OrderBook

OWNER = "market_maker"


class MarketMaker:
    """The market maker: two-sided quotes from its own account in `book`."""

    def __init__(self, book: OrderBook, config: Config) -> None:
        self.book = book
        self.config = config
        self.account = book.open_account(OWNER, config.mm_cash)
        self.estimate = 50.0
        """Moving price estimate, in cents."""
        self.extra_half_spread = 0.0
        """Widening from recent markout losses, in cents, on top of `mm_half_spread`."""
        self.bid: int | None = None
        """Price of the current bid, or None if it is not bidding."""
        self.ask: int | None = None
        """Price of the current offer, or None if it is not offering."""
        self._fills_seen = len(book.fills)
        self.closing_position: int | None = None
        """Inventory when the market closed, just before settlement."""

    @property
    def half_spread(self) -> float:
        """The base half-spread plus loss widening, capped at `mm_max_half_spread`."""
        return min(
            self.config.mm_max_half_spread, self.config.mm_half_spread + self.extra_half_spread
        )

    def requote(self) -> None:
        """Cancel both quotes, learn from fills since the last requote, and quote again."""
        if self.config.mm_size == 0:  # turned off: never quotes, so never fills
            return
        self.book.cancel_all(OWNER)
        self._learn()
        cfg = self.config
        inventory = self.account.position
        # Skew: each contract of inventory shifts both quotes `mm_skew` cents against it.
        center = min(99.0, max(1.0, self.estimate - cfg.mm_skew * inventory))
        h = self.half_spread
        bid = math.floor(center - h)
        ask = max(math.ceil(center + h), bid + 1)

        # Post-only: never cross the agents' resting orders.
        best_ask, best_bid = self.book.best_ask, self.book.best_bid
        if best_ask is not None:
            bid = min(bid, best_ask - 1)
        if best_bid is not None:
            ask = max(ask, best_bid + 1)

        # A fully filled quote must not carry inventory past the cap.
        self.bid = self._quote(BUY, bid, cfg.mm_max_inventory - inventory)
        self.ask = self._quote(SELL, ask, cfg.mm_max_inventory + inventory)

    def _learn(self) -> None:
        """Update the price estimate and the loss widening from fills since last time.

        Every trade moves the estimate `mm_alpha` of the way to its price. Then each of
        the market maker's own fills is marked to the new estimate: buying above it or
        selling below it is a loss. The old widening decays by `mm_decay`, and the
        loss per quoted contract (`/ mm_size`) times `mm_widen` is added to it.
        """
        new_fills = self.book.fills[self._fills_seen :]
        self._fills_seen = len(self.book.fills)
        alpha = self.config.mm_alpha
        for fill in new_fills:
            self.estimate += alpha * (fill.price - self.estimate)
        loss = 0.0
        for fill in new_fills:
            if fill.maker != OWNER:
                continue
            # As maker, it bought when the taker sold.
            bought = fill.taker_side == SELL
            markout = (
                self.estimate - fill.price if bought else fill.price - self.estimate
            ) * fill.qty
            loss += max(0.0, -markout)
        self.extra_half_spread = (
            self.config.mm_decay * self.extra_half_spread
            + self.config.mm_widen * loss / self.config.mm_size
        )

    def _quote(self, side: str, price: int, room: int) -> int | None:
        """Post the largest affordable quote up to `mm_size` and `room` contracts.

        Returns the quote's price, or None if nothing was posted.
        """
        if not 1 <= price <= 99:
            return None
        lo, hi = 0, max(0, min(self.config.mm_size, room))
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.book.can_submit(OWNER, side, price, mid, post_only=True):
                lo = mid
            else:
                hi = mid - 1
        if lo == 0:
            return None
        self.book.submit(OWNER, side, price, lo, post_only=True)
        return price
