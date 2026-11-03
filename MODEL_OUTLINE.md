# Model Outline: Kalshi as a Complex System

An agent-based model of one binary prediction market run under Kalshi's rules. We describe it in three levels:

- **Level 0:** two traders with private clues and a market maker. The smallest model in which a price can pool information.
- **Level 1:** three traders on a tiny social network that spreads scheduled news.
- **Level 2:** the full model used for the experiments: 300 traders with varied traits, a realistic (Holme-Kim) social network, disinformation, a fee-free market for comparison, and a manipulator.

Each level adds to the one before it. Settings not mentioned keep the values of the previous level. Level 0 and Level 1 can be run with `configs/level_0.toml` and `configs/level_1.toml`. `PLAN.md` has the full design and the reasons behind each choice; `SOURCES.md` has the references.

---

## 1. Purpose

We want to **understand**, not predict, how three things change how fast and how accurately a prediction market's price finds the truth:

1. **Market rules:** an order book with no fees vs. the same order book with Kalshi's fees.
2. **Information spread:** how much disinformation the social network that carries news also carries.
3. **Manipulation:** a well-funded trader who tries to move the price for profit (pump-and-dump or spoofing).

**Stylized facts** the model should be able to speak to:

- Prediction market prices are roughly calibrated: a 70c contract wins about 70% of the time (Page & Clemen 2013).
- They show a **favorite-longshot bias**: cheap contracts win less often than their price suggests and expensive ones more often (Burgi et al. 2025 on Kalshi).
- On Kalshi, traders who post resting orders (makers) earn more than traders who take them (takers).
- Kalshi spreads in liquid markets are small (median 2c).
- Manipulation attempts in prediction markets usually move prices only briefly and tend to lose money (Hanson & Oprea 2009; Rasooly & Rozzi 2025).

**Benchmark.** In every run we know the best possible forecast, `P*`: the exact Bayesian probability of YES given every real clue held by any trader. We score the market price against `P*`.

---

## 2. Background needed

- **Contract.** Pays 100c if the event happens (YES), 0 otherwise. Prices are whole cents from 1 to 99 and read as probabilities.
- **One YES book.** "Buy NO at x" is the same as "sell YES at 100 - x", so the model keeps one order book for YES. A position is a signed count of YES contracts (negative means holding NO).
- **Limit order book (CLOB).** Traders post orders to buy (bids) or sell (asks) at a price. An order that matches the other side trades at once (the *taker*); otherwise it *rests* in the book until filled or canceled (the *maker*). Matching uses price-time priority. Canceling is free.
- **Kalshi fee.** The taker pays `ceil(0.07 x C x P x (1 - P))` dollars for `C` contracts at price `P` (in dollars), rounded up to the cent per order. The maker pays nothing. The fee is largest at 50c (1.75c per contract).
- **Log-odds.** `logit(p) = ln(p / (1 - p))`, with inverse `p = 1 / (1 + e^(-logit))`. 50% is 0; YES-leaning beliefs are positive and NO-leaning ones negative. We work in log-odds for three reasons:
  - *Bayesian updating becomes addition.* Bayes' rule in odds form multiplies the odds by each clue's likelihood ratio; taking logs turns that product into a sum. A clue that is right with probability `q` adds `ln(q / (1 - q))` if it points YES and subtracts it if it points NO. So a posterior is just a running total of clues, which makes `P*` exact and cheap to compute. For example, with `q = 0.53` one clue moves 50% to 53%, and two move it to 56%.
  - *Combining sources stays valid.* Traders add their own evidence to the market price's log-odds. Log-odds run from minus to plus infinity, so any sum maps back to a probability strictly between 0 and 1. Adding raw probabilities could go outside that range.
  - *It matches how evidence feels near the extremes.* An equal step in log-odds moves the price a lot near 50% and little near 1% or 99%, so moving a confident market takes more evidence. Probability weighting (`g`) and belief noise are also applied in log-odds, so they treat YES and NO symmetrically.
- **Kelly betting.** The Kelly rule (Kelly 1956) sizes a bet to maximize the expected log of wealth. For a contract bought at price `c` (in dollars) that the trader thinks wins with probability `p > c`, it stakes the fraction `f = (p - c) / (1 - c)` of wealth. Bets grow with the trader's edge and shrink as the edge disappears, and a trader never stakes everything, so no one goes broke on one market. We use it because it is the standard rule for sizing bets on a perceived edge. It also gives a known result: under Kelly sizing the price settles near the wealth-weighted average belief (Wolfers & Zitzewitz 2006). Full Kelly bets are large and swing wealth sharply, so each trader stakes only a fraction `kelly_frac` of the Kelly amount (fractional Kelly).
- **Full collateral.** Every order reserves its worst-case cost plus fee, so no account can go negative. All money is integer cents.

---

## 3. Level 0: two traders and a market maker

### Main parts
- **2 traders** (the only agents).
- **A market maker**, a quoting component that is not an agent.
- **The world**: a hidden outcome and 10 clues about it.
- **The order book** with Kalshi fees.

### The world
- The true outcome `Y` is YES or NO, a fair coin flip.
- There are `M = 10` clues. Each points to the truth with probability `q = 0.6`, independently given `Y`. A clue that points the wrong way is honest noise, not a lie.
- Each clue moves the log-odds by `ln(q / (1 - q))`. So the posterior from a set of clues is
  `logit(P) = (#YES clues - #NO clues) x ln(q / (1 - q))`.
- All 10 clues are private. Each trader draws 5 distinct clues at random at setup (they may overlap). `P*` uses every distinct clue either trader holds.

### Traders
**Homogeneous**: both have the same fixed traits.

| Trait | Value | Meaning |
|---|---|---|
| `n_clues` | 5 | private clues held |
| `market_trust` | 5 | how many clues the trader thinks the price is worth |
| `noise` | 0 | random error added to beliefs each turn |
| `min_edge` | 2c | smallest expected profit per contract worth acting on |
| `patience` | 0.75 | chance of posting a resting order when taking is not worth it |
| `kelly_frac` | 0.5 | fraction of the Kelly bet placed |
| `g` | 0.8 | probability weighting (1 = none; below 1 overweights longshots) |
| `cash` | $1,000 | starting cash |

**Goal:** maximize expected log wealth (fractional Kelly) given their beliefs.

**Information:** their own 5 clues (from the start) and the market price (each turn). They do not see the other trader's clues.

**Actions on each turn**, in order:

1. **Update belief** `b` (probability of YES):
   ```
   logit(b) = L_own + w_market x logit(price) + noise x z
   w_market = market_trust / (market_trust + n_own)
   ```
   `L_own` is the log-odds from the trader's own clues, `n_own` the number of clues, `z` standard normal. Before any trade happens the trader ignores the price (`w_market = 0`). The trader *adds* its evidence to the price rather than averaging, so the price can come to reflect evidence that no single trader holds. Because `w_market < 1` and traders do not subtract the part of their evidence already in the price, they keep disagreeing and keep trading.
   - **Price** = the midpoint of best bid and ask when both exist and the spread is at most 10c, else the last trade price.
2. **Perceive probabilities** with weighting: `logit(p_yes) = g x logit(b)`, `p_no = 1 - p_yes`.
3. **Set a target position** by fractional Kelly. If `p_yes` is above the YES price, the full-Kelly stake is `f = (p_yes - price) / (1 - price)` of cash; the trader holds `kelly_frac x f x cash / price` YES contracts (at least 1). Symmetrically for NO. The order size is the gap between target and current position.
4. **Clean up resting orders.** Cancel an order whose expected edge has fallen below `min_edge`, or that has sat 3c or more behind the best price for 20 steps.
5. **Act:**
   - If trading now against resting orders beats its perceived value by at least `min_edge` per contract after fees, **take** (as deep into the book as that stays true; any unfilled remainder is canceled).
   - Otherwise, with probability `patience`, **post** a resting order `min_edge` inside its value, never crossing the book.
   - Otherwise **wait**.

### Market maker
No beliefs or clues. Each step, before traders act, it cancels its quotes and posts 36 contracts on each side around a moving price estimate (an exponential average of trade prices, starting at 50c). It:
- shifts both quotes against its inventory to shed it,
- widens its spread after losing money to better-informed traders, then narrows again over time,
- never trades on arrival (post-only, so it is always the maker and pays no fees),
- holds at most 200 contracts either way, under the same collateral rules as traders.

It exists so the book is rarely empty and traders always have someone to trade with.

### Space and time
- **No space.** Location does not matter; there is no network.
- **Time** is discrete steps. One run (500 steps here) stands for the life of one market, roughly days to weeks. Within a step the market maker requotes first, then traders act **one at a time** in random order (asynchronous), so each sees the book left by the one before.

### Interactions (one step)
1. Market maker cancels and requotes.
2. Both traders take a turn (in random order) as described above. Orders match immediately in the book; fees are charged to the taker.
3. Record price, `P*`, volume, spread, and depth.
4. (Debug mode) Check conservation: positions sum to zero; total cash plus fees is constant; every account can cover its worst case.

### Initialization
Draw `Y` and the clues; deal each trader 5 private clues; give each $1,000 and no position; give the market maker $10,000; empty book.

### Stopping
After a fixed number of steps (500). Then the market **settles**: each YES contract pays 100c if `Y` is YES. Profit and loss is computed for everyone.

### Parameters
Clue accuracy `q`, clues per trader, and the trait values above. Level 0 is a sanity check, not an experiment.

### Outputs
Price over time vs. `P*`; whether the price ends closer to `P*` than either trader's own posterior; volume, spread; each trader's profit.

---

## 4. Level 1: adding news and a network

Changes from Level 0:

- **3 traders**, still identical (`market_trust = 10`, `noise = 0.05`).
- **News.** There are now `M = 20` clues. The first 10 are private, dealt as before. The other 10 are **news**, released one at a time at evenly spaced steps (news `j` at step `(j + 1) x T / 11`).
- **Network.** Traders are the nodes of a triangle (everyone linked to everyone). Space now means *who talks to whom*, not physical location.
- **Spreading.** At release a news clue is given to 1 random trader. Each step, each trader spreading a clue tells each neighbor with probability `beta = 0.5`. A trader starts passing on a new clue only after its own next trading turn (traders act on news before sharing it). A trader stops retelling a clue once it tells someone who already knew it.
- **Private clues never spread.** They reach the price only through trading.
- **Beliefs.** A news clue a trader has learned counts as its own evidence (`L_own` and `n_own` include it). `P*` now includes every private clue held plus all news released so far.
- **Turns.** Each step, 2 of the 3 traders are drawn at random to act.

**Step order:** release news to its seed trader; spread clues one step; market maker requotes; the drawn traders act; record.

**New outputs:** how fast the price moves toward `P*` after each news release, and how long a clue takes to reach every trader.

---

## 5. Level 2: the full model

This is the model used for the experiments. Changes from Level 1:

### Agents (300 traders)
**Heterogeneous in informedness and cash only.** Traders have no type labels. Each is a vector of traits set at setup, and behavior follows from the traits ("expert" and "casual bettor" are regions of trait space). Two traits vary across traders:

| Trait | Controls | Distribution |
|---|---|---|
| `n_clues` | private clues held | negative binomial, mean 20, cap 60: about 40% hold none, the top 10% hold most |
| `cash` | starting budget | lognormal, rank-correlated with `n_clues` |

Every other trait is one value shared by all traders:

| Trait | Controls | Value |
|---|---|---|
| `w_hearsay` | weight on rumors heard from neighbors | 0.39 |
| `market_trust` | how many clues the price is worth | 35.3 |
| `w_depth` | belief nudge from order book imbalance | 0.1 |
| `noise` | belief error per turn | 0.0506 |
| `min_edge` | smallest edge worth acting on | 2.22c |
| `patience` | chance of posting instead of waiting | 0.902 |
| `kelly_frac` | fraction of Kelly bet | 0.274 |
| `g` | probability weighting | 0.8 (lab estimates) |

Better-informed traders are richer. Uninformed traders are needed: without them the informed have nobody to trade with. The cash distribution, most single trait values, and the market maker settings are **calibrated** so the default model reproduces six real Kalshi patterns at once (trade sizes, spreads, rarely empty book, informed traders earn more, price calibration, price beats the average trader). Exact values are in `src/kalshi_model/config.py`.

**Belief rule** gains two terms:
```
logit(b) = L_own + w_hearsay x L_heard + w_market x logit(price) + w_depth x imbalance + noise x z
w_market = market_trust / (market_trust + n_own + w_hearsay x n_heard)
```
`L_heard` is the log-odds of fake clues heard from neighbors (see below). `imbalance` = (bid depth - ask depth) / total depth within 2c of the best prices.

**Turns.** Each step, 20 traders are drawn without replacement, each equally likely.

### World and information
- `M = 80` clues, `q = 0.53`. The first 60 are private; each trader draws `n_clues` of them, so each private clue is held by about 65 traders. The other 20 are news, each seeded to 5 random traders.
- One clue moves `P*` from 50% by about 3 points, more than the fee at 50c.
- **Disinformation source** (a component, not an agent): releases `ratio x 20` fake clues, all pointing against the truth, at evenly spaced steps, seeded and spread like news. Traders cannot tell fakes from news, but true news can be checked at its source, so news counts in full while a fake counts only as hearsay, at weight `w_hearsay`. Fakes are excluded from `P*`.

### Network
300 nodes, mean degree 8, one trader per node, in one Holme-Kim network: preferential attachment gives it hubs (influencers), and closing triangles gives it clustering (friend circles), the two features real social networks share. The network type is fixed, not varied.

### Market rules
- **CLOB:** order book plus quoting market maker, no fees.
- **Kalshi (default):** the same, plus Kalshi taker fees.

### Manipulator (Experiment 3 only)
One extra agent with the population's trait values, 2 private clues, and 5% of all traders' cash. It is not on the network, trades every step, and from step 1000 adds a manipulation layer on top of the honest rules:
- **Pump and dump:** buy until the price is 5c above where it started (at most 100 steps), hold 50 steps, then sell what it bought.
- **Spoofing:** for 500 steps, keep a huge fake bid (20,000 contracts) 2c behind the best bid to inflate the imbalance traders read, while selling real contracts at the best ask.

Its effect is measured against an **honest twin**: the same agent, same seed, trading honestly throughout.

### One run
```
setup: draw Y and clues; build network; set traits; place traders on nodes; deal private clues
for t in 1..2000:
    release scheduled news and fakes to seed traders
    spread clues one step
    market maker requotes (order book rules only)
    20 randomly drawn traders (plus the manipulator, if present) take turns
    record price, P*, volume, spread, depth
settle at 100c per winning contract; compute profit for everyone
```

**Initialization:** traders start with their drawn cash and no positions; book empty; price estimate 50c.
**Stopping:** fixed length, 2,000 steps, then settlement.
**Randomness:** separate random streams for the world, traits, turns, spreading, and manipulator, so two configurations on the same seed differ only in the setting changed. Every configuration runs on the same 100 seeds.

### Parameters explored
Each experiment changes **one variable** around the default (Kalshi rules, Holme-Kim, no disinformation, no manipulator):

| Experiment | Variable | Values |
|---|---|---|
| E1 Market rules | fees | CLOB (no fees), Kalshi |
| E2 Information | disinformation ratio | 0, 0.05, 0.1, 0.2, 0.5 (0, 1, 2, 4, 10 fakes per 20 news clues) |
| E3 Manipulation | manipulator | honest twin, pump-and-dump, spoofing |

A validation run also sets `g = 1` for everyone to see whether the favorite-longshot bias comes from the rules or from traders' probability weighting.

### Outputs
| Output | Question |
|---|---|
| Aggregation error `|price - P*|` over time | Did the market use the information it had? |
| Brier score of the final price | Was the forecast right? |
| Price move toward each news clue and its half-life | How fast is news priced in? |
| Volume, spread, depth, empty-book steps | How liquid was the market? |
| Fees paid and profit, by informedness, network position, maker vs. taker | Who wins and who pays? |
| Steps for a clue to reach half the traders | How fast does the network spread news? |
| Distortion from fakes (gap from `P*` in the fakes' direction vs. the same seed without fakes) | How far do lies move the price? |
| Manipulator profit minus twin profit; price distortion and its duration | Does manipulation pay, and does it harm the market? |
| Calibration curve vs. real Kalshi data | Does the model show the favorite-longshot bias? |

Results are reported as means with 95% confidence intervals, and comparisons as paired differences on shared seeds.
