"""The Mesa model: one market over one run (PLAN.md Section 2.5).

Each step releases scheduled news to seed agents, lets the CLOB's market maker requote,
then gives an activity-weighted sample of agents a turn, then records the step. After
the last step the market settles. Under the AMM agents trade with the LMSR; under the
CLOB they trade in the order book, whose accounts hold every trader's cash and position.
Every random part draws from its own stream (see `rng.py`), and the turn order, belief
noise, and news seeds use the same number of draws in every configuration, so a seed
gives the same population, world, and turn sequence whatever the market rules.
"""

from __future__ import annotations

import math

import mesa
import numpy as np

from kalshi_model.agents import Trader, solvent
from kalshi_model.config import Config
from kalshi_model.lmsr import LMSR
from kalshi_model.market_maker import MarketMaker
from kalshi_model.orderbook import BUY, SELL, OrderBook
from kalshi_model.rng import make_streams
from kalshi_model.traits import Traits, draw_traits
from kalshi_model.world import deal_private_clues, make_world


class InvariantError(AssertionError):
    """A conservation invariant failed: the accounting has a bug."""


class MarketModel(mesa.Model):
    """One prediction market run.

    `traits` replaces the drawn population (for limit-case tests). With `debug`, the
    conservation invariants are checked after every step.
    """

    def __init__(
        self,
        config: Config | None = None,
        seed: int = 0,
        *,
        traits: Traits | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(rng=seed)
        self.config = config = config or Config()
        self.seed = seed
        self.debug = debug
        streams = make_streams(seed)
        self._news_rng = streams["news"]
        self._turns_rng = streams["turns"]
        self._decisions_rng = streams["decisions"]
        self._patience_rng = streams["patience"]

        self.world = make_world(config, streams["world"])
        self.traits = traits if traits is not None else draw_traits(config, streams["traits"])
        if len(self.traits) != config.n_agents:
            raise ValueError("traits must have one entry per agent (config.n_agents)")

        self.traders = [Trader(self, i) for i in range(config.n_agents)]
        self.lmsr: LMSR | None = None
        self.book: OrderBook | None = None
        self.market_maker: MarketMaker | None = None
        if config.mechanism == "amm":
            self.lmsr = LMSR(config.lmsr_b)
        else:
            self.book = OrderBook(config.taker_fee_bp, config.maker_fee_bp)
            for trader in self.traders:
                trader.account = self.book.open_account(trader.index, trader.initial_cash)
            self.market_maker = MarketMaker(self.book, config)
        hands = deal_private_clues(self.traits.n_clues, config.n_private, streams["deal"])
        for trader, hand in zip(self.traders, hands, strict=True):
            for clue_id in hand.tolist():
                trader.learn_own(clue_id, bool(self.world.points_yes[clue_id]))
        self._known_real: set[int] = set().union(*(t.known for t in self.traders))

        weights = self.traits.activity
        self._turn_weights = weights / weights.sum()
        self._initial_total_cash = self._total_cash()
        self.settled = False
        self._closing_price: float | None = None
        self.volume = 0
        """Contracts traded in the current step."""
        self.empty_book_steps = 0
        """CLOB steps that ended with no bid or no offer in the book."""

        reporters = {
            "price": lambda m: m.price,
            "p_star": lambda m: m.p_star,
            "volume": lambda m: m.volume,
        }
        if self.book is not None:
            reporters |= {
                "best_bid": lambda m: m.book.best_bid,
                "best_ask": lambda m: m.book.best_ask,
                "spread": lambda m: m.book.spread,
                "depth": lambda m: (
                    m.book.depth(BUY, m.config.depth_window)
                    + m.book.depth(SELL, m.config.depth_window)
                ),
                "empty_book": lambda m: m.book.best_bid is None or m.book.best_ask is None,
                "mm_position": lambda m: m.market_maker.account.position,
                "mm_half_spread": lambda m: m.market_maker.half_spread,
                "mm_quoting": lambda m: (
                    m.market_maker.bid is not None and m.market_maker.ask is not None
                ),
            }
        self.datacollector = mesa.DataCollector(model_reporters=reporters)

    @property
    def price(self) -> float:
        """The scored price: what agents read as the market's probability.

        The LMSR price under the AMM; in the CLOB the midpoint within the spread cap, else
        the last trade, else NaN (nothing quoted or traded yet). After settlement it stays
        at the last price before the market closed.
        """
        if self._closing_price is not None:
            return self._closing_price
        if self.lmsr is not None:
            return self.lmsr.price
        reference = self.book.reference_price(self.config.spread_cap)
        return math.nan if reference is None else reference

    @property
    def p_star(self) -> float:
        """The benchmark: the posterior from every real clue some agent knows."""
        return self.world.p_star(self._known_real)

    def step(self) -> None:
        if self.settled:
            raise RuntimeError("the market has already settled")
        t = self.steps
        self._release_news(t)

        n_turns = self.config.turns_per_step
        order = self._turns_rng.choice(
            self.config.n_agents, size=n_turns, replace=False, p=self._turn_weights
        )
        noise = self._decisions_rng.standard_normal(n_turns).tolist()
        patience_draws = self._patience_rng.random(n_turns).tolist()
        self.volume = 0
        if self.lmsr is not None:
            for index, z in zip(order.tolist(), noise, strict=True):
                self.volume += abs(self.traders[index].take_turn_amm(self.lmsr, z))
        else:
            fills_before = len(self.book.fills)
            self.market_maker.requote()
            for index, z, u in zip(order.tolist(), noise, patience_draws, strict=True):
                self.traders[index].take_turn_clob(self.book, z, u, t)
            self.volume = sum(f.qty for f in self.book.fills[fills_before:])
            if self.book.best_bid is None or self.book.best_ask is None:
                self.empty_book_steps += 1

        self.datacollector.collect(self)
        if t >= self.config.n_steps:
            self._settle()
        if self.debug:
            self.check_invariants()

    def run(self) -> None:
        """Run every remaining step and settle."""
        while not self.settled:
            self.step()

    def _release_news(self, t: int) -> None:
        for clue_id in np.flatnonzero(self.world.news_steps == t) + self.world.n_private:
            seeds = self._news_rng.choice(
                self.config.n_agents, size=self.config.n_news_seeds, replace=False
            )
            points_yes = bool(self.world.points_yes[clue_id])
            for index in seeds.tolist():
                self.traders[index].learn_own(int(clue_id), points_yes)
            if self.config.n_news_seeds:
                self._known_real.add(int(clue_id))

    def _settle(self) -> None:
        self._closing_price = self.price
        y = self.world.y
        if self.lmsr is not None:
            for trader in self.traders:
                trader.settle(y)
            self.lmsr.settle(y)
        else:
            self.book.settle(y)
        self.settled = True
        self.running = False

    def _total_cash(self) -> int:
        if self.lmsr is None:
            return sum(a.cash for a in self.book.accounts.values()) + self.book.fees_collected
        return sum(t.cash for t in self.traders) + self.lmsr.cash

    def check_invariants(self) -> None:
        """Raise InvariantError if money or contracts were created or destroyed."""
        if self.book is not None:
            self.book.check_invariants()
            for trader in self.traders:
                if trader.account is not self.book.account(trader.index):
                    raise InvariantError(f"trader {trader.index} is off the order book's ledger")
            if self._total_cash() != self._initial_total_cash:
                raise InvariantError("cash + fees changed")
            return
        positions = sum(t.position for t in self.traders)
        if positions != self.lmsr.net:
            raise InvariantError(f"agents hold {positions} YES but the LMSR sold {self.lmsr.net}")
        if self._total_cash() != self._initial_total_cash:
            raise InvariantError(
                f"total cash {self._total_cash()} != starting {self._initial_total_cash}"
            )
        for trader in self.traders:
            if not solvent(trader.cash, trader.position):
                raise InvariantError(f"trader {trader.index} is not fully collateralized")
        if self.lmsr.cash < 0:
            raise InvariantError(f"the LMSR's cash is negative: {self.lmsr.cash}")
