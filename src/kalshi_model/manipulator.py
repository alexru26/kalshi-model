"""The manipulator: an agent with the honest rules plus a
manipulation layer.

Its traits are set by the experimenter, not drawn: each is the population's single value
(`Config`), except that it ignores book imbalance (`w_depth = 0`, since it knows which
of the depth is its own), its informedness is `manip_n_clues` (the population median),
and its cash is `manip_cash_share` of all agents' starting cash on the seed. It holds
`manip_n_clues` private clues, sits off the social network (it learns news only through
the price), pays fees like any agent, and trades every step, in a random slot among the
agents' turns. All its draws come from the "manipulator" stream, so the world, the
population, and the agents' turns are the same with or without it, and its own draws are
the same in every E3 configuration.

Strategies (`Config.manipulator`):

- **twin**: the honest rules throughout (E3's baseline, the honest twin).
- **pump** (pump and dump): from `manip_start`, cancel its orders and buy up to
  `manip_chunk` contracts per turn, taking offers up to `pump_push` cents above the
  start price, until the price reaches that level or `pump_max_steps` pass; hold for
  `pump_hold` steps; then sell what the pump bought, `manip_chunk` per turn, into bids
  no lower than the start price, for at most `pump_max_steps`. Then honest again.
- **spoof**: for `spoof_steps` from `manip_start`, on every turn move its fake bid of
  `spoof_size` contracts to `spoof_ticks` cents behind the others' best bid, and keep a
  real offer at the best ask, posting `spoof_qty` more contracts whenever its offer has
  filled or the best ask has moved away from it. Then cancel both and go honest.
  Re-pegging the fake bid each turn keeps it behind the best bid, as in Wang & Wellman
  (2017), but it can still fill if the price falls to it between the spoofer's turns.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from kalshi_model.agents import Trader
from kalshi_model.config import Config
from kalshi_model.orderbook import BUY, SELL, OrderBook

if TYPE_CHECKING:
    from kalshi_model.model import MarketModel

OWNER = "manipulator"
"""The manipulator's order-book account (agents' accounts are their integer indices)."""

HONEST, PUMP, HOLD, DUMP, SPOOF = "honest", "pump", "hold", "dump", "spoof"
LAYER_PHASES = (PUMP, HOLD, DUMP, SPOOF)
"""Phases in which the manipulation layer, not the honest rules, sets the turn."""


def manipulator_traits(config: Config, cash: int) -> dict[str, float]:
    """The manipulator's traits: the population's single values
    (`traits.py`), `manip_n_clues`, `w_depth = 0`, and the given cash."""
    return {
        "n_clues": config.manip_n_clues,
        "w_hearsay": config.w_hearsay,
        "market_trust": config.market_trust,
        "w_depth": 0.0,
        "noise": config.noise,
        "min_edge": config.min_edge,
        "patience": config.patience,
        "kelly_frac": config.kelly_frac,
        "g": config.g,
        "cash": cash,
        "activity": math.inf,
    }


class Manipulator(Trader):
    """The manipulator, or its honest twin. Not one of `model.traders`."""

    def __init__(self, model: MarketModel, rng: np.random.Generator) -> None:
        config = model.config
        cash = round(config.manip_cash_share * int(model.traits.cash.sum()))
        super().__init__(model, -1, owner=OWNER, traits=manipulator_traits(config, cash))
        self.strategy = config.manipulator
        clue_rng, self._turn_rng = rng.spawn(2)
        n = min(config.manip_n_clues, config.n_private)
        for clue_id in clue_rng.choice(config.n_private, size=n, replace=False).tolist():
            self.learn_own(clue_id, bool(model.world.points_yes[clue_id]))

        self.phases: dict[int, str] = {}
        """Step -> what the manipulator did on its turn that step."""
        self.start_price: float | None = None
        """The price, in cents, when the manipulation began."""
        self.closing_position: int | None = None
        self.pump_reached = False
        """Whether the pump pushed the price to its target before `pump_max_steps`."""
        self._phase = HONEST
        self._phase_began = 0
        self._pump_base = 0
        """Position before the pump, so the dump sells only what the pump bought."""

    @property
    def layer_steps(self) -> list[int]:
        """Steps on which the manipulation layer acted, in order (none for the twin)."""
        return sorted(t for t, phase in self.phases.items() if phase in LAYER_PHASES)

    def draw_turn(self, n_turns: int) -> tuple[int, float, float]:
        """This step's slot among the `n_turns` agent turns (0 to n_turns: before agent
        turn `slot`), belief noise draw, and patience draw."""
        rng = self._turn_rng
        return int(rng.integers(n_turns + 1)), float(rng.standard_normal()), float(rng.random())

    def take_turn(self, book: OrderBook, z: float, u: float, step: int) -> None:
        phase = self._advance(book, step)
        self.phases[step] = phase
        config = self.model.config
        if phase == HONEST:
            self.take_turn_clob(book, z, u, step)
        elif phase == PUMP:
            limit = min(99, math.ceil(self.start_price + config.pump_push))
            self._submit_largest(book, BUY, limit, config.manip_chunk, ioc=True)
        elif phase == DUMP:
            excess = self.position - self._pump_base
            limit = max(1, math.ceil(self.start_price))
            self._submit_largest(book, SELL, limit, min(config.manip_chunk, excess), ioc=True)
        elif phase == SPOOF:
            self._spoof(book)

    def _advance(self, book: OrderBook, step: int) -> str:
        """Move through the strategy's phases; return this turn's phase."""
        config = self.model.config
        if self.strategy == "twin" or step < config.manip_start:
            return HONEST
        if self._phase == HONEST and self.start_price is None:
            book.cancel_all(OWNER)
            reference = book.reference_price(config.spread_cap)
            self.start_price = 50.0 if reference is None else 100 * reference
            self._pump_base = self.position
            self._enter(PUMP if self.strategy == "pump" else SPOOF, step)
        elapsed = step - self._phase_began
        if self._phase == PUMP:
            reference = book.reference_price(config.spread_cap)
            reached = reference is not None and 100 * reference >= (
                self.start_price + config.pump_push
            )
            if reached or elapsed >= config.pump_max_steps:
                self.pump_reached = reached
                self._enter(HOLD, step)
        elif self._phase == HOLD and elapsed >= config.pump_hold:
            self._enter(DUMP, step)
        if self._phase == DUMP and (
            self.position <= self._pump_base or step - self._phase_began >= config.pump_max_steps
        ):
            self._enter(HONEST, step)
        if self._phase == SPOOF and elapsed >= config.spoof_steps:
            book.cancel_all(OWNER)
            self._enter(HONEST, step)
        return self._phase

    def _enter(self, phase: str, step: int) -> None:
        self._phase = phase
        self._phase_began = step

    def _spoof(self, book: OrderBook) -> None:
        """Re-peg the fake bid behind the others' best bid; keep a real offer at the best
        ask (it keeps its place in the queue while it is still there)."""
        config = self.model.config
        orders = book.open_orders(OWNER)
        for order in orders:
            if order.side == BUY:
                book.cancel(order.order_id)
        offers = [o for o in orders if o.side == SELL]
        if not (offers and book.best_ask == offers[0].price):
            for order in offers:
                book.cancel(order.order_id)
            if book.best_ask is not None:
                self._submit_largest(book, SELL, book.best_ask, config.spoof_qty, ioc=False)
        best_bid = book.best_bid
        if best_bid is not None and best_bid - config.spoof_ticks >= 1:
            self._submit_largest(
                book, BUY, best_bid - config.spoof_ticks, config.spoof_size, ioc=False
            )
