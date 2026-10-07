# Project Plan: Kalshi as a Complex System

**An agent-based model of a binary prediction market with Kalshi's fees, order book, rumors, and manipulation**

Audience: an entry-level college student in a complex systems / modeling course.
Language: Python 3.12 with the Mesa agent-based modeling framework.
Goal: the **Core** scope in Section 8. Stretch ideas are listed there as out of scope.

---

## 1. The pitch in one paragraph

A prediction market turns many people's scattered guesses into one number: the price of a contract that pays $1 if an event happens. Classical theory (Hayek, Plott & Sunder, Hanson's market scoring rules) says this price should become a good probability estimate. Kalshi is a real, US-regulated version of this idea, but it adds features the classical models ignore: an order book with "makers" and "takers", a fee that is largest at 50 cents and shrinks toward 0 and 100 cents, fees rounded up to the next cent, and free order cancellation. This project builds a simple agent-based model (ABM) of one binary market, then switches these Kalshi features on and off to see how they change **how fast and how accurately information gets into the price**, and how easily that price can be pushed around by false rumors and a deliberate manipulator.

### Headline research question

> How do Kalshi-specific market rules (the P(1-P) fee and the maker/taker order book) change the speed and accuracy with which a prediction market absorbs true information, false rumors, and manipulation, compared with a classical fee-free market?

### Sub-questions (each becomes one experiment, see Section 6)

1. **Fees:** Does Kalshi's fee create a "dead zone" where traders with small edges stop trading, and is this zone worst near 50 cents?
2. **Information spread:** How does the shape of the social network carrying news and rumors change how quickly the price reacts, and how long a false rumor survives?
3. **Manipulation:** How much money does it take to hold the price away from the truth, how long does the distortion last, and does Kalshi's fee make manipulation easier or harder?

---

## 2. Background: what makes Kalshi different

### 2.1 How a Kalshi binary contract works

- Each contract pays **$1 if YES** and $0 if NO. Prices are whole cents from 1 to 99.
- Buying NO at 40 cents is the same economic position as selling YES at 60 cents. The model can therefore use a **single YES order book** where "buy NO at x" is stored as "sell YES at 100 - x".
- Positions are fully collateralized: you pay the full price up front, so no one can lose more than they put in.
- Trading happens on a **central limit order book (CLOB)**. A *maker* posts a resting limit order; a *taker* accepts a resting order immediately.

### 2.2 Kalshi's fee formula (the key ingredient)

From Kalshi's official fee schedule:

```
taker fee = round_up_to_cent( 0.07   x C x P x (1 - P) )
maker fee = round_up_to_cent( 0.0175 x C x P x (1 - P) )   # only on some markets; many charge 0
```

where `C` is the number of contracts and `P` is the price in dollars. Maker fees are charged only when the resting order actually executes; **cancelling an order is free**. Some product families (for example certain index markets) use different multipliers, so the plan treats the multiplier as a parameter.

Why this matters for modeling:

| Price | Fee per contract (exact) | Charged on 1 contract | Fee as % of cost (1 contract) | Charged on 100 contracts |
|---|---|---|---|---|
| 5c  | 0.33c | 1c | 20% | $0.34 |
| 10c | 0.63c | 1c | 10% | $0.63 |
| 25c | 1.31c | 2c | 8%  | $1.32 |
| 50c | 1.75c | 2c | 4%  | $1.75 |
| 90c | 0.63c | 1c | 1.1% | $0.63 |

Three model-relevant consequences:

1. **The fee is highest in absolute terms at 50c**, exactly where the outcome is most uncertain and where new information should move the price the most.
2. **Rounding up makes small orders on cheap contracts very expensive in percentage terms.** Order size becomes a meaningful agent choice.
3. **YES at P and NO at 1 - P pay the same fee** (because P(1-P) is symmetric), which keeps the model tidy.

### 2.3 Classical models vs. Kalshi

| Feature | Classical / textbook prediction market | Kalshi |
|---|---|---|
| Who sets prices | Often an **automated market maker** such as Hanson's LMSR (always willing to trade) | Order book: other traders, including professional market makers |
| Fees | Usually ignored in theory; Iowa Electronic Markets charged no trading commissions | Taker fee proportional to P(1-P), maker fee on some markets, rounded up |
| Liquidity | Guaranteed by the market maker (bounded loss) | Depends on who is posting orders; can be thin |
| Cancellation | Not modeled | Free |
| Traders | Rational, Bayesian | Mostly retail, with documented behavioral biases |
| Price interpretation | Price = probability | Price is informative but biased (favorite-longshot bias, see Bürgi et al. 2025) |

Other platforms make useful comparison points: PredictIt charged a 10% fee on profits plus a 5% withdrawal fee, and Polymarket historically charged no trading fee on most markets. These give natural alternative "fee regimes" to simulate.

---

## 3. Literature review

The goal is not to read everything, but to know the handful of ideas each part of the model is built on. Each entry says *why it matters for this project*.

### 3.1 Foundations: why prices aggregate information

- **Hayek (1945), "The Use of Knowledge in Society."** Prices summarize knowledge that no single person has. This is the philosophical starting point for every prediction market.
- **Plott & Sunder (1988).** Lab experiments showing markets can aggregate private information, but not always. Gives you permission to find failures.
- **Wolfers & Zitzewitz (2004), "Prediction Markets"** and **Arrow et al. (2008), "The Promise of Prediction Markets" (Science).** Readable overviews; good for the introduction.
- **Berg, Forsythe, Nelson & Rietz (2008).** Iowa Electronic Markets results: markets often beat polls. The main "classical success story."
- **Manski (2006)** and **Wolfers & Zitzewitz (2006), "Interpreting prediction market prices as probabilities."** Shows that price equals average belief only under assumptions. Your model can test exactly when those assumptions break (fees, budgets).

### 3.2 Market mechanisms and microstructure

- **Hanson (2003, 2007), Logarithmic Market Scoring Rule (LMSR).** The classical automated market maker. About 20 lines of code; this is your **baseline "classical" market**.
- **Kyle (1985)** and **Glosten & Milgrom (1985).** The classic models of informed traders vs. market makers. Key idea: market makers widen the bid-ask spread when they fear trading against informed traders ("adverse selection"). Use it to design the market-maker agent.
- **Chen & Pennock (2007)** and **Othman, Pennock, Reeves & Sandholm (2013).** Variations on automated market makers and liquidity; optional reading.

### 3.3 Agent-based models of markets

- **Gode & Sunder (1993), "Zero-Intelligence" traders.** Random traders with only a budget constraint already produce sensible market prices. Your noise traders come from here. Shows that a lot of "market behavior" is about the rules, not the brains.
- **Arthur, Holland, LeBaron, Palmer & Tayler (1997), Santa Fe Artificial Stock Market.** The classic complex-systems market model: heterogeneous adaptive agents produce bubbles and crashes.
- **Othman (2008), "Zero-Intelligence Agents in Prediction Markets" (AAMAS).** ZI agents with diffuse beliefs reproduce empirical prediction market patterns. Very close in spirit to this project, and a good model to partially replicate as the first sanity check of your LMSR market (Step 3 in Section 9).
- **Chakraborty, Das & Peabody (2015).** A continuous double auction prediction market with ZI traders plus an LMSR-style market maker. Shows how to combine the two mechanisms in one simulation.
- **Grimm et al. (2020), the ODD protocol.** The standard way to document an ABM (Overview, Design concepts, Details). Use it for your write-up; it makes the model interpretable to graders.

### 3.4 Behavioral biases and the favorite-longshot bias

The model does not include biased agents (see Section 8), but these papers explain the bias the calibration check in Section 7 looks for.

- **Tversky & Kahneman (1992), probability weighting.** People overweight small probabilities.
- **Snowberg & Wolfers (2010).** The favorite-longshot bias in betting is best explained by misperceived probabilities, not love of risk. If the core model (no biased agents) does not reproduce Kalshi's bias, this is the explanation to cite.
- **Page & Clemen (2013).** Prediction markets are well calibrated near resolution but less so for long-horizon events.

### 3.5 Information spread on networks

- **Daley & Kendall (1964).** The classic rumor-spreading model (an SIR epidemic variant with "spreaders" and "stiflers").
- **DeGroot (1974)** and **DeMarzo, Vayanos & Zwiebel (2003), "Persuasion bias."** Belief averaging on networks; persuasion bias means people double-count information they hear repeatedly. This is a clean way to model why rumors can overshoot.
- **Watts & Strogatz (1998)** small-world networks and **Barabasi & Albert (1999)** scale-free networks. The two network types you will compare.
- **Bikhchandani, Hirshleifer & Welch (1992), Banerjee (1992), Granovetter (1978).** Information cascades and threshold models. Background only: herding agents are out of scope, but cite these when discussing what the model leaves out.

### 3.6 Manipulation

- **Hanson, Oprea & Porter (2006).** Lab experiment: manipulators failed to distort prices because other traders traded against them.
- **Hanson & Oprea (2009), "A Manipulator Can Aid Prediction Market Accuracy" (Economica).** Counterintuitive theory: a manipulator is a "free lunch" for informed traders, which can draw more of them in and improve accuracy. A great hypothesis to test.
- **Rhode & Strumpf (2004)** on historical betting markets, and **Rothschild & Sethi (2016)** on Intrade's 2012 presidential market: evidence of a single large trader apparently trying to hold one candidate's price up.
- **Rasooly & Rozzi (2025), "How manipulable are prediction markets?"** A field experiment that randomly pushed prices in 817 markets. Effects persisted up to 60 days but faded, and faded faster in markets with more traders, more volume, and outside probability estimates. **This gives you real-world numbers to compare your simulated "distortion half-life" against.**

### 3.7 New research specifically on Kalshi and modern platforms

- **Burgi, Deng & Whelan (2025), "Makers and Takers: The Economics of the Kalshi Prediction Market"** (UCD / CESifo working paper). Uses over 300,000 Kalshi contracts. Findings: prices are informative and improve near closing, but show a strong **favorite-longshot bias** (contracts under 10c win far less often than their price implies), and **makers earn higher returns than takers**. Their explanatory model uses heterogeneous beliefs and traders who over-estimate small probabilities. **This is the most important paper for the project: it gives you the stylized facts your model is compared against.**
- **Columbia University wash-trading study (2025, reported by Fortune).** Estimated that about 14% of Polymarket wallets showed wash-trading behavior and that wash trading made up a large, fluctuating share of volume. Shows that "volume" can itself be manipulated; worth a sentence in the discussion.
- **Galanis (2026), "Information Aggregation with AI Agents"** (arXiv). Puts LLM agents into an LMSR prediction market with private signals. Relevant for the discussion section, since AI trading bots are increasingly real participants.
- **LLM herding (ICLR 2026 workshop paper, "Information Homogenization Induces Herding in Retrieval-Augmented LLM Agent Markets").** When agents read the same news, their actions become correlated even without communicating. Connects to the rumor-spread experiment in the discussion.

---

## 4. Why this is a complex system (for the course framing)

- **Emergence:** the price is a collective belief that no single agent holds.
- **Feedback loops:** the market maker quotes around recent prices and widens its spread after losing to informed traders, which changes who can trade profitably next; with persuasion bias on, repeated rumors reinforce themselves through the network.
- **Networks:** information travels along a social graph, not instantly to everyone.
- **Heterogeneity:** informed traders, noise traders, market makers, and a manipulator coexist.
- **Nonlinearity and thresholds:** the fee creates a no-trade band whose width depends on the price level.
- **Path dependence:** the same true information can produce different final prices depending on the order and route in which it arrives.

---

## 5. Model design

Keep the model small enough to explain on one slide. Everything below is a starting point; each rule is 1 to 5 lines of code.

### 5.1 The world: hidden truth and "clues"

This is the trick that makes the model interpretable: we always know the **correct answer** the market *should* reach.

1. At the start of a run, flip a coin to decide the true outcome `Y` (YES or NO).
2. Create `M` clues (default `M = 30`). Each clue points to the true outcome with probability `q` (default `q = 0.55`), otherwise it points the wrong way.
3. Each agent privately sees a random handful of clues (e.g. 3). Other clues are released later as "news" (Section 5.4).
4. The **benchmark price** `P*` is what a perfect Bayesian would believe after seeing every clue released so far:

```
logit(P*) = (number of YES clues released - number of NO clues released) x ln(q / (1 - q))
```

This gives three clean accuracy measures:
- **Aggregation error:** `|price - P*|` (did the market combine the information it had?)
- **Forecast error:** Brier score `(final price - Y)^2` (was the market right?)
- **Calibration:** over many runs, do contracts priced at 20c resolve YES about 20% of the time? Because `P*` is the exact posterior, `P*` itself is perfectly calibrated, so any miscalibration in the price comes from the market, not the world.

Why `q = 0.55` and not 0.6: the final `P*` must usually stay between 10% and 90%, or every market is trivially obvious. Computed exactly from the binomial distribution with all 30 clues released:

| q | M | Final `P*` in [10%, 90%] | Final `P*` on the wrong side of 50% |
|---|---|---|---|
| 0.60 | 30 | 40% of runs | 10% |
| **0.55** | **30** | **93% of runs** | **23%** |
| 0.52 | 30 | 100% of runs | 34% |

Step 2 in Section 9 re-checks this with simulation and reports the `P*` distribution.

### 5.2 Market mechanisms (the main "treatment")

Same agents, different rules:

| ID | Mechanism | Fees | Represents |
|---|---|---|---|
| M0 | LMSR automated market maker | none | Classical theory (Hanson) |
| M1 | Order book | none | Classical exchange (IEM-like) |
| M2 | Order book | Kalshi taker 0.07, maker 0 | Kalshi (most markets) |
| M3 | Order book | Kalshi taker 0.07, maker 0.0175 | Kalshi (markets with maker fees) |
| M4 | Order book | 10% of profits at settlement | PredictIt-style |
| M5 | Order book | flat 1c per contract | Simple comparison |

LMSR with liquidity parameter `b`, written so it cannot overflow when share counts get large:

```python
def lmsr_cost(q_yes, q_no, b):
    m = max(q_yes, q_no)  # log-sum-exp trick
    return m + b * math.log(math.exp((q_yes - m) / b) + math.exp((q_no - m) / b))

def lmsr_price_yes(q_yes, q_no, b):
    d = (q_no - q_yes) / b
    if d >= 0:
        e = math.exp(-d)
        return e / (1 + e)
    return 1 / (1 + math.exp(d))
# cost of buying k YES shares = lmsr_cost(q_yes + k, q_no, b) - lmsr_cost(q_yes, q_no, b)
# the LMSR market maker can never lose more than b * ln(2) in a binary market
```

Order book: integer prices 1 to 99, price-time priority, one YES book. Kalshi fee in integer cents (avoid floating-point rounding bugs):

```python
def kalshi_fee_cents(contracts, price_cents, rate_bp=700):  # 700 = 0.07, 175 = 0.0175
    numerator = rate_bp * contracts * price_cents * (100 - price_cents)
    return -(-numerator // 1_000_000)  # ceiling division
```

(Check: 1 contract at 50c gives 700 x 2500 / 1,000,000 = 1.75, rounded up to 2c.)

### 5.3 Agents

Each agent has cash, a signed YES position (negative means it holds NO), and a belief `b` (its probability that YES happens). On each step, `K` randomly chosen agents act (random activation). Order size: 1 to 10 contracts.

| Type | Belief comes from | Trading rule | Purpose |
|---|---|---|---|
| **Informed** | Bayesian update from the clues it knows | Buy YES if `b - ask > fee(ask) + min_edge`; sell YES (buy NO) if `bid - b > fee(bid) + min_edge`; otherwise, with probability `patience`, post a limit order a little inside its belief | Brings truth into the price |
| **Zero-intelligence (noise)** | None | Random buy/sell at a random price within budget (Gode & Sunder) | Liquidity and noise |
| **Market maker** | Slow-moving average of recent prices | Posts a bid and ask around its estimate; widens spread after being "picked off"; skews quotes to reduce inventory | Kalshi-style liquidity provider; earns maker returns |
| **Manipulator** | Ignores clues | Brute force: buys YES whenever the price is below its target, until its budget runs out (Section 5.5) | Manipulation experiment |

### 5.4 News and rumors on a social network

- Agents live on a network (start with Watts-Strogatz small-world; compare with Erdos-Renyi random and Barabasi-Albert scale-free) built with NetworkX.
- At scheduled times, a hidden clue is **released** to a few "seed" agents.
- Each step, every agent who knows a clue tells each neighbor with probability `beta` (independent cascade / Daley-Kendall style). Optionally, agents stop spreading after a while ("stiflers").
- **False rumors** are fake clues pointing the wrong way, injected exogenously at scheduled times. Agents believe a clue with credibility `c` (weight `c x ln(q/(1-q))` in their log-odds).
- **Persuasion bias switch:** if `on`, agents count a clue again each time they hear it from a different neighbor (DeMarzo et al.). This is an easy way to show how echo chambers inflate confidence.

### 5.5 Manipulation

The manipulator has the **influence motive** seen in Rothschild & Sethi's Intrade whale: it wants the price to *look* a certain way and is willing to lose money. It uses one strategy, **brute force**: keep buying YES whenever the price is below its target, until the budget runs out. It pays taker fees like everyone else.

### 5.6 One run, step by step

```
setup: draw Y, make clues, build network, create agents, give each agent initial clues
for t in 1..T:
    release scheduled news clues and false rumors to seed agents
    spread clues/rumors one step along the network
    manipulator acts (if present)
    pick K random agents; each updates belief and maybe submits an order
    match orders, charge fees, record price, volume, spread, P*
    (debug mode) assert the conservation invariants in Section 9, Step 4
settle: pay $1 per winning contract; compute P&L by agent type
```

Suggested defaults: 100 agents, T = 2000 steps, 100 seeds per configuration (never fewer than 50). This runs in seconds to minutes per configuration.

### 5.7 What to measure

| Metric | Question it answers |
|---|---|
| Aggregation error `|price - P*|` over time | Did the market use the information it had? |
| Brier score of the final price | Was the forecast right? |
| Calibration curve across runs | Do prices mean what they say? Is there a favorite-longshot bias? |
| Reaction half-life after news | How fast does new information get priced in? |
| Volume, bid-ask spread, book depth | Liquidity and market quality |
| No-trade steps and empty-book steps | Did the market freeze? (Section 11) |
| Total fees and who paid them | Fee incidence (does the fee fall on informed traders or noise traders?) |
| P&L by agent type, maker vs taker | Who wins? Compare with Burgi et al.'s maker/taker finding |
| Peak distortion and distortion half-life | Manipulation size and persistence (compare with Rasooly & Rozzi) |

---

## 6. Experiments

Change **one or two parameters at a time** and keep everything else at defaults. Each experiment has a prediction written *before* running it, so results are interpretable either way. Every configuration in an experiment runs on the **same list of seeds** (common random numbers), so differences between mechanisms are paired run by run instead of being drowned in seed-to-seed noise.

### E1. Fees: does Kalshi's fee change how agents trade?

- **Vary:** mechanism M0 to M5; fee multiplier (0, 0.035, 0.07, 0.14); order size; mix of informed vs. noise traders.
- **Predictions:**
  - H1a: An informed trader only trades if their edge beats the fee, so there is a no-trade band around the price that is widest at 50c. Prices near 50c should react more slowly to small pieces of news.
  - H1b: Higher fees reduce volume a lot but accuracy only a little, because traders with large edges still trade.
  - H1c: Free maker orders (M2) push patient agents toward posting limit orders, and makers out-earn takers.
  - H1d: Rounding hurts small orders on cheap contracts most, discouraging small longshot bets.
- **Key figure:** heatmap of reaction half-life by (price level x fee multiplier); bar chart of P&L by agent type and maker/taker.

### E2. Information and rumors: how does news travel into the price?

- **Vary:** network type; spread probability `beta`; fraction of false rumors; credibility `c`; persuasion bias on/off.
- **Predictions:**
  - H2a: Scale-free networks spread news fastest when hubs are seeded, but also spread false rumors fastest.
  - H2b: False rumors move the price, and the market corrects only as fast as true information spreads, so the time to correct is roughly the time for the true clue to reach enough informed traders.
  - H2c: Persuasion bias makes the price overshoot `P*`.
- **Key figure:** price vs. `P*` over time for one example run per network type, plus average correction time.

### E3. Manipulation: can someone push the price around?

- **Vary:** manipulator budget; number of informed traders; market mechanism (M0 LMSR vs. M2/M3 Kalshi order book); manipulator target near 50c vs. near the extremes.
- **Predictions:**
  - H3a (Hanson & Oprea): informed traders profit from trading against the manipulator; with enough informed traders, the price returns to `P*` and average accuracy may even improve.
  - H3b (Rasooly & Rozzi): distortion lasts longer in markets with fewer traders and less volume.
  - H3c (Kalshi-specific): the fee makes small corrections unprofitable, so a manipulator can hold the price a few cents off `P*` almost for free, especially near 50c.
- **Key figure:** distortion half-life vs. number of informed traders, one line per mechanism; manipulator cost per cent of distortion.

---

## 7. Checking the model against real Kalshi data

An ABM is more convincing when it is compared with real facts. Kalshi's market data API is public and needs no account or key (base URL `https://api.elections.kalshi.com/trade-api/v2`; endpoints include markets, trades, candlesticks, and separate historical endpoints for closed markets; check the current docs for exact field names).

**The one check: calibration and the favorite-longshot bias.**

1. Pull a few thousand resolved binary markets.
2. For each, take the price a fixed time before close (for example the last trade or candlestick midpoint 24 hours before close). Use one rule for every market and state it.
3. Bin by price (for example 10 bins of 10c) and plot the fraction that resolved YES in each bin, with binomial 95% confidence intervals. You should see the bias Burgi et al. report: cheap contracts win less often than their price implies.
4. Build the same calibration curve from the simulation (final prices across all seeds of the default M2 configuration).

How to read the result: the core model has no behaviorally biased agents, and `P*` is perfectly calibrated by construction. So the comparison tests whether **Kalshi's rules alone** (fees, rounding that discourages small longshot bets per H1d, maker/taker structure) produce any favorite-longshot bias. If the simulated curve is close to the diagonal while the real one is not, that is evidence the real bias comes from trader behavior (Snowberg & Wolfers), and it names a clear piece of future work.

What you **cannot** do with public data: see who traded, so you cannot directly verify agent types. Say this explicitly as a limitation.

---

## 8. Scope

**Target: Core.** Minimum is the fallback order if time runs short.

| Tier | Includes | Enough for |
|---|---|---|
| **Minimum (build first)** | LMSR vs. Kalshi order book; informed + noise + market-maker agents; clue benchmark `P*`; experiment E1 | A complete, passing project |
| **Core (the goal)** | Minimum + network news/rumors (E2) + brute-force manipulator (E3) + calibration check against Kalshi data | A strong project |

If you must cut below Core, **E1 + E3** alone directly answer "how is Kalshi different" and "can agents manipulate the price."

**Out of scope** (too much for this project; mention as future work in the report): herding and momentum agents and the cascade tipping-point experiment; longshot-lover agents with probability weighting; rumor-plus-trade and spoofing manipulation (a book-watching agent would be needed); Kelly sizing; LLM trading agents.

---

## 9. Implementation steps

Build in this order. Each step ends with a concrete check, and the risk mitigations from Section 11 are built into the step where they belong rather than left for the end.

1. **Project skeleton, defaults, and fees.**
   Set up the repo (Python 3.12, Mesa, numpy, pandas, matplotlib, networkx, requests, pytest). Put every parameter and its default in one `Config` dataclass in `src/config.py`; every result file saves the config it was run with, and the report's defaults table is generated from this dataclass so it cannot drift from the code. Write `fees.py` for all fee regimes.
   *Done when:* `test_fees.py` passes: the five rows of the table in Section 2.2 (for example 1 contract at 50c gives 2c, 100 contracts at 5c give 34c), `fee(P) == fee(100 - P)` for every price, maker fees charged only on fill, M4 charges only on positive profit at settlement. The fee-curve figure is drawn.

2. **World: outcome, clues, and `P*`.**
   Write `world.py`.
   *Done when:* a test checks `P*` against the formula in Section 5.1 on hand-made clue sets, and a script simulates 10,000 worlds with the defaults and reports the distribution of final `P*`. At least 90% of runs end with `P*` between 10% and 90% (about 93% expected for `q = 0.55`, `M = 30`). If not, retune `q` or `M` before going further.

3. **Mesa model with the LMSR market (M0), informed and noise agents.**
   Write `lmsr.py`, `agents.py` (informed, zero-intelligence), and `model.py` (the Mesa `Model`, its `step()`, and a `DataCollector` recording price, `P*`, volume and fees).
   *Done when:* tests show LMSR prices stay in (0, 1), the cost of a trade does not depend on how it is split into pieces, and the market maker's loss never exceeds `b * ln(2)`. A plot shows the M0 price tracking `P*` on one example run, and ZI-only runs behave like Othman (2008).

4. **Order book and matching engine (M1), tests first.**
   Write `test_orderbook.py` before `orderbook.py`: price-time priority, partial fills, cancels (free, and they release reserved cash), "buy NO at x" behaving exactly like "sell YES at 100 - x", and orders rejected when the agent cannot afford them (cash may never go negative, and a resting order reserves its full cost plus its worst-case fee).
   Add an invariant check that runs every step in debug mode and in every test. With positions measured as signed YES contracts:
   - the sum of all positions is 0 (every YES held is matched by a NO held);
   - total agent cash (free + reserved) + fees collected + 100c x open contracts = total starting cash, where open contracts = the sum of the positive positions;
   - after settlement there are no open contracts, so total cash + fees collected = total starting cash.
   *Done when:* all order book tests pass and 50 full M1 runs finish with zero invariant violations.

5. **Kalshi fees (M2, M3), other regimes (M4, M5), and the market maker. Run E1.**
   Add the market-maker agent. Record no-trade steps and empty-book steps every run (Section 5.7) so a frozen market is visible instead of silently producing a flat price. Check that noise traders and the market maker stay solvent over all T steps at the default budgets (track their minimum cash); fees and adverse selection drain both, and a broke liquidity provider freezes the market late in the run. Raise default budgets if they do not.
   Write down the E1 predictions, then run E1 with `mesa.batch_run`, passing the same seed list for every configuration (`rng=list_of_seeds`).
   *Done when:* the E1 figures exist and no default-configuration run spends more than a small, reported fraction of steps with an empty book.

6. **Network, news, and rumors. Run E2.**
   Write `network.py`: build the graph with NetworkX, release clues to seeds, spread them each step, inject false rumors, and add the persuasion-bias switch.
   *Done when:* a test confirms that with `beta = 1` a clue reaches everyone within the graph's diameter in steps, and the E2 figures exist.

7. **Brute-force manipulator. Run E3.**
   Add the manipulator agent and the distortion metrics (peak distortion, distortion half-life, manipulator cost per cent of distortion).
   *Done when:* the invariants still hold with the manipulator present, and the E3 figures exist.

8. **Kalshi calibration check.**
   Write `data/fetch_kalshi.py` and the calibration comparison in Section 7.
   *Done when:* real and simulated calibration curves are on one figure, each with confidence intervals.

9. **Robustness.**
   For every headline result, report the mean over runs with a 95% confidence interval, treating **each run as one data point** (time steps within a run are not independent samples). Report mechanism comparisons as paired differences on the shared seeds. If a confidence interval is too wide to tell treatments apart, add seeds. Re-run the main E1 and E3 comparisons with each key default (agent count, `K`, `q`, budgets) moved up and down once.
   *Done when:* a robustness appendix shows which conclusions survive the changed defaults.

10. **Write-up.**
    ODD model description, the three experiments with their pre-written predictions and results, the calibration check, limitations, and future work (the out-of-scope list in Section 8). State real-world claims as "mechanisms that could explain" aggregate patterns.

---

## 10. Tools and code layout

- **Python 3.12** (required by current Mesa), **Mesa 3.x**, `numpy`, `pandas`, `matplotlib`, `networkx`, `requests`, `pytest`.
- **Mesa** provides the parts every ABM needs: `mesa.Agent` and `mesa.Model` classes, the agent set (`model.agents`) for random activation, `DataCollector` for per-step records, and `mesa.batch_run` for parameter sweeps over many seeds.
- Keep the market mechanics (fees, LMSR, order book) as **plain Python modules with no Mesa dependency**. They are the code most likely to have bugs, and plain functions and classes are the easiest to unit-test.
- Reproducibility: create every model with an explicit seed, `MarketModel(rng=seed)`. Draw all randomness from the model's `self.random` or `self.rng`, never from the global `random` module. Mesa's older `seed=` argument is deprecated; use `rng=`.

A sketch of how the pieces fit:

```python
import mesa

class Trader(mesa.Agent):
    def __init__(self, model, cash):
        super().__init__(model)
        self.cash = cash
        self.position = 0  # signed YES contracts

class MarketModel(mesa.Model):
    def __init__(self, mechanism="M2", n_agents=100, k_active=10, rng=None):
        super().__init__(rng=rng)
        # build world, market, network, agents ...
        self.datacollector = mesa.DataCollector(
            model_reporters={"price": "price", "p_star": "p_star", "volume": "volume"}
        )

    def step(self):
        # release news, spread rumors, manipulator acts ...
        for agent in self.random.sample(list(self.agents), self.k_active):
            agent.step()
        # match orders, charge fees, check invariants ...
        self.datacollector.collect(self)

results = mesa.batch_run(
    MarketModel,
    parameters={"mechanism": ["M0", "M1", "M2", "M3", "M4", "M5"]},
    rng=list(range(100)),  # same 100 seeds for every mechanism
    max_steps=2000,
    data_collection_period=1,
)
```

```
kalshi-abm/
  README.md               # how to run, ODD model description
  src/
    config.py             # Config dataclass: every parameter and default
    world.py              # outcome Y, clues, P* benchmark
    fees.py               # kalshi_fee_cents and other fee regimes
    lmsr.py               # classical automated market maker
    orderbook.py          # limit order book, matching, invariant checks
    agents.py             # informed, zero-intelligence, market maker, manipulator (Mesa agents)
    network.py            # social graph + news and rumor spreading
    model.py              # Mesa MarketModel: one run, DataCollector
  experiments/
    e1_fees.py  e2_information.py  e3_manipulation.py   # batch_run sweeps -> results/*.csv
    robustness.py
  data/
    fetch_kalshi.py       # pulls resolved markets from the public API
  notebooks/
    figures.ipynb         # all plots from results/*.csv
  tests/
    test_fees.py          # Section 2.2 table, symmetry, maker-on-fill, M4 settlement
    test_world.py         # P* formula
    test_lmsr.py          # price bounds, path independence, loss <= b ln 2
    test_orderbook.py     # priority, partial fills, cancels, YES/NO symmetry, invariants
    test_network.py       # spread reaches everyone with beta = 1
```

---

## 11. Risks and how to handle them

Each mitigation is built into the implementation step listed.

| Risk | Mitigation | Step |
|---|---|---|
| Order book bugs silently ruin results | Tests written before the order book; three conservation invariants (positions sum to 0; cash + fees + 100c x open contracts is constant; cash + fees is constant after settlement) asserted every step in debug mode and in tests; cash can never go negative | 4, 7 |
| LMSR numerical or logic bugs | Overflow-safe formulas (Section 5.2); tests for price bounds, path independence, and the `b * ln(2)` loss bound | 3 |
| Results depend on one lucky seed | 100 seeds per configuration (never fewer than 50); the same seeds across configurations so comparisons are paired; each run is one data point for confidence intervals; add seeds when intervals are too wide | 5, 9 |
| Too many parameters to explain | One `Config` dataclass holds every default; results save their config; the report's defaults table is generated from it; vary at most two parameters per experiment | 1 |
| Market freezes (no one trades) | Market maker and noise traders in every configuration; budgets checked so they stay solvent for all T steps; no-trade and empty-book steps recorded and reported | 5 |
| `P*` too close to 0 or 1 | Default `q = 0.55`, `M = 30` (93% of runs end in [10%, 90%]); distribution of `P*` simulated and reported before building markets | 2 |
| Overclaiming about real Kalshi | Validate only against an aggregate fact (calibration); frame results as "mechanisms that could explain"; state that agent types cannot be observed in public data | 8, 10 |
| Ethical concern about manipulation research | The model studies market resilience; strategies are never run on real markets | 10 |

---

## 12. Final deliverables

1. Code repository with tests and a README.
2. Written report (8 to 12 pages): question, background, ODD model description, three experiments with hypotheses and results, Kalshi calibration check, limitations, future work.
3. Five to seven key figures (fee curve, price vs. `P*` example, E1 heatmap, E1 P&L by agent type, E2 network comparison, E3 manipulation half-life, real vs. simulated calibration).
4. A short presentation.

---

## 13. References

**Kalshi and recent work**
- Kalshi, Fee Schedule (official PDF). https://kalshi.com/docs/kalshi-fee-schedule.pdf
- Kalshi Trade API v2 (public market data). https://api.elections.kalshi.com/trade-api/v2
- Burgi, C., Deng, W., & Whelan, K. (2025). Makers and Takers: The Economics of the Kalshi Prediction Market. UCD Economics WP2025_19 / CESifo WP 12122. https://www.ucd.ie/economics/t4media/WP2025_19.pdf
- Rasooly, I., & Rozzi, R. (2025). How manipulable are prediction markets? arXiv:2503.03312. https://arxiv.org/abs/2503.03312
- Galanis, S. (2026). Information Aggregation with AI Agents. arXiv:2604.20050. https://arxiv.org/abs/2604.20050
- Fortune (Nov 7, 2025). Columbia researchers on wash trading in Polymarket. https://fortune.com/2025/11/07/polymarket-wash-trading-inflated-prediction-markets-columbia-research
- Information Homogenization Induces Herding in Retrieval-Augmented LLM Agent Markets (ICLR 2026). https://www.iclr.cc/virtual/2026/10016534

**Prediction markets: classic**
- Hayek, F. A. (1945). The Use of Knowledge in Society. *American Economic Review*, 35(4).
- Plott, C., & Sunder, S. (1988). Rational expectations and the aggregation of diverse information in laboratory security markets. *Econometrica*, 56(5).
- Hanson, R. (2003). Combinatorial information market design. *Information Systems Frontiers*, 5(1).
- Hanson, R. (2007). Logarithmic market scoring rules for modular combinatorial information aggregation. *Journal of Prediction Markets*, 1(1).
- Wolfers, J., & Zitzewitz, E. (2004). Prediction Markets. *Journal of Economic Perspectives*, 18(2).
- Wolfers, J., & Zitzewitz, E. (2006). Interpreting prediction market prices as probabilities. NBER WP 12200.
- Manski, C. (2006). Interpreting the predictions of prediction markets. *Economics Letters*, 91(3).
- Berg, J., Forsythe, R., Nelson, F., & Rietz, T. (2008). Results from a dozen years of election futures markets research. *Handbook of Experimental Economics Results*.
- Arrow, K. J., et al. (2008). The promise of prediction markets. *Science*, 320(5878).
- Page, L., & Clemen, R. (2013). Do prediction markets produce well-calibrated probability forecasts? *Economic Journal*, 123(568).
- Chen, Y., & Pennock, D. (2007). A utility framework for bounded-loss market makers. *UAI*.
- Othman, A., Pennock, D., Reeves, D., & Sandholm, T. (2013). A practical liquidity-sensitive automated market maker. *ACM TEAC*.

**Microstructure and agent-based markets**
- Kyle, A. S. (1985). Continuous auctions and insider trading. *Econometrica*, 53(6).
- Glosten, L., & Milgrom, P. (1985). Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. *Journal of Financial Economics*, 14(1).
- Gode, D., & Sunder, S. (1993). Allocative efficiency of markets with zero-intelligence traders. *Journal of Political Economy*, 101(1).
- Arthur, W. B., Holland, J., LeBaron, B., Palmer, R., & Tayler, P. (1997). Asset pricing under endogenous expectations in an artificial stock market. In *The Economy as an Evolving Complex System II*.
- Othman, A. (2008). Zero-intelligence agents in prediction markets. *AAMAS*. https://www.site.uottawa.ca/~nelkadri/CSI5389/Papers/126-Othman_AAMAS2008.pdf
- Chakraborty, M., Das, S., & Peabody, J. (2015). Price evolution in a continuous double auction prediction market with a scoring-rule based market maker. *AAAI*. https://cs.gmu.edu/~sanmay/papers/CDAwithSRMM.pdf
- Grimm, V., et al. (2020). The ODD protocol for describing agent-based and other simulation models: a second update. *JASSS*, 23(2).
- Kelly, J. L. (1956). A new interpretation of information rate. *Bell System Technical Journal*, 35(4).

**Behavior, networks, and cascades**
- Tversky, A., & Kahneman, D. (1992). Advances in prospect theory. *Journal of Risk and Uncertainty*, 5(4).
- Snowberg, E., & Wolfers, J. (2010). Explaining the favorite-long shot bias: Is it risk-love or misperceptions? *Journal of Political Economy*, 118(4).
- Bikhchandani, S., Hirshleifer, D., & Welch, I. (1992). A theory of fads, fashion, custom, and cultural change as informational cascades. *Journal of Political Economy*, 100(5).
- Banerjee, A. (1992). A simple model of herd behavior. *Quarterly Journal of Economics*, 107(3).
- Granovetter, M. (1978). Threshold models of collective behavior. *American Journal of Sociology*, 83(6).
- Daley, D. J., & Kendall, D. G. (1964). Epidemics and rumours. *Nature*, 204.
- DeGroot, M. (1974). Reaching a consensus. *Journal of the American Statistical Association*, 69(345).
- DeMarzo, P., Vayanos, D., & Zwiebel, J. (2003). Persuasion bias, social influence, and unidimensional opinions. *Quarterly Journal of Economics*, 118(3).
- Watts, D., & Strogatz, S. (1998). Collective dynamics of small-world networks. *Nature*, 393.
- Barabasi, A.-L., & Albert, R. (1999). Emergence of scaling in random networks. *Science*, 286.

**Manipulation**
- Hanson, R., Oprea, R., & Porter, D. (2006). Information aggregation and manipulation in an experimental market. *Journal of Economic Behavior & Organization*, 60(4).
- Hanson, R., & Oprea, R. (2009). A manipulator can aid prediction market accuracy. *Economica*, 76(302).
- Rhode, P., & Strumpf, K. (2004). Historical presidential betting markets. *Journal of Economic Perspectives*, 18(2).
- Rothschild, D., & Sethi, R. (2016). Trading strategies and market microstructure: Evidence from a prediction market. *Journal of Prediction Markets*, 10(1). https://papers.ssrn.com/abstract=2322420
