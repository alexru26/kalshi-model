# Project Plan: Kalshi as a Complex System

**An agent-based model of one binary prediction market: how Kalshi's rules, the network carrying news and disinformation, and manipulators change how fast and how accurately the price finds the truth.**

Python 3.12 + Mesa 3.x. Sources are listed with one-line summaries in `SOURCES.md`.

---

## 1. Overview and research questions

A prediction market turns scattered private guesses into one price: what a contract paying $1 on YES costs. Classical theory (Hayek; Plott & Sunder; Hanson's LMSR) says that price becomes a good probability. Kalshi is a real, regulated version, but its rules differ from the textbook: a central limit order book (CLOB) instead of an automated market maker, a taker fee of `0.07 x C x P x (1 - P)` rounded up to the cent (largest at 50c), no fee on most maker orders, and free cancellation.

We simulate one binary market with a known benchmark: `P*`, the exact Bayesian posterior given every real clue that any agent holds. It is the best forecast the market's real information allows, and every price is scored against it.

**Headline question.** How do Kalshi's market rules, the structure of the network that spreads information, and manipulators change the speed and accuracy with which a prediction market absorbs information?

### Experiments

Every configuration runs on the same 100 seeds, so comparisons are paired run by run. All three experiments share one **default configuration**: Kalshi rules, Holme-Kim network, no disinformation, no manipulator. Disinformation is off by default so that E1 and E3 test clean markets. Predictions are written before running.

**E1. Kalshi rules (3 configurations).** AMM (LMSR, no fees) vs. CLOB (no fees, free cancellation) vs. Kalshi (CLOB + Kalshi fees). Free cancellation is on in both CLOBs because every real exchange allows it, so each step changes one thing:
- AMM to CLOB: what an open order book changes.
- CLOB to Kalshi: what Kalshi's fee rules change.
- *Predictions:* the fee creates a no-trade band widest at 50c, so prices near 50c react more slowly to small news; fees cut volume much more than accuracy; Kalshi's free maker orders push patient agents to post resting orders, and makers out-earn takers (as Burgi et al. find on real Kalshi data).

**E2. Information spread (8 configurations).** One variable at a time around the default:
- *Network sweep:* 5 networks at disinformation ratio 0.1 (Section 2.4).
- *Disinformation sweep:* ratios 0, 0.05, 0.1, 0.2 on Holme-Kim, i.e. exactly 0, 1, 2, 4 fakes added to the 20 real news clues. A real clue pushes toward the truth by `2q - 1 = 0.06` on average and a fake pushes 1 against it, so at ratio 0.06 the news stream carries no net signal: 0.05 sits just below that break-even point, and 0.1 and 0.2 go past it as stress tests.
- *Predictions:* well-connected agents trade earlier and earn more (Ozsoylev et al. 2014); echo chambers sustain the largest disinformation distortion; distortion grows faster than the ratio, because agents who follow the price amplify a rumor once it moves the price.

**E3. Manipulation (3 configurations).** Kalshi default with an honest twin vs. pump-and-dump manipulator vs. spoofing manipulator (Section 2.3).
- *Predictions:* pump-and-dump moves the price but loses money, because agents with clues trade against it (Hanson & Oprea 2009); spoofing can profit because its fake orders never fill, to the extent agents read book depth (Wang & Wellman 2017). Both distort less, and for less time, in more liquid runs (Rasooly & Rozzi 2025).

**Validation.** Compare the simulated calibration curve with real Kalshi data (favorite-longshot bias, Section 3).

**Out of scope** (future work in the report): factorial designs crossing experiment variables; a separate agent-behavior experiment; liar agents who trade on their own disinformation; herding agent rules; disinformation targeted at hubs; PredictIt-style or flat fee regimes; LLM agents.

---

## 2. Model design and parameters

**Terminology.** *Agents* are the traders. The *manipulator* is an agent with its own extra rules. The *market maker* (the LMSR, or a quoting component in the CLOBs) and the *information sources* (scheduled news, the disinformation source) are parts of the model, not agents.

### 2.1 How the markets work

- **Contract:** pays 100c on YES, 0 on NO; integer prices 1 to 99. Money is integer cents.
- **One YES book:** "buy NO at x" is "sell YES at 100 - x". Positions are signed YES counts. Everything is fully collateralized: a resting order reserves its full cost plus worst-case fee.
- **AMM:** Hanson's LMSR with liquidity `b`; it always quotes a price and its loss is bounded by `b ln 2`.
- **CLOB:** price-time priority, partial fills, free cancels; an order that would trade with its owner's own resting order cancels that order instead. A market-maker component quotes both sides around a moving price estimate, cancels and requotes each step, skews quotes to shed inventory, and widens its spread after losing to informed trades (Glosten & Milgrom 1985; Avellaneda & Stoikov 2008). Its quotes are post-only, its inventory is capped, and it trades from an order-book account under the same collateral rules as agents.
- **Kalshi fees:** taker `0.07 x C x P x (1 - P)` rounded up to the cent, maker 0, cancellation free. Rounding is per order: like Kalshi's fee accumulator, an order filled in pieces pays what one equivalent fill would. The fee is largest at 50c and hits small orders on cheap contracts hardest (one contract at 5c pays 1c, 20% of cost).

### 2.2 Agents

Agents have no type labels. Each agent is a vector of traits drawn at setup, and its behavior follows from them. Uninformed recreational bettors and experts are regions of trait space.

**Each turn an agent:**
1. **Believes.** In log-odds:
   ```
   logit(b) = (1 - w_market) * (L_own + w_hearsay * L_heard) + w_market * logit(mid) + w_depth * imbalance + noise * z
   w_market = market_trust / (market_trust + n_eff + eps),   n_eff = n_own + w_hearsay * n_heard
   ```
   `L` is net clue evidence ((# YES - # NO) x ln(q / (1 - q))) from clues seen directly (`own`, including news delivered to the agent as a seed) or relayed by neighbors (`heard`); each clue counts once however many neighbors relay it. The own estimate and the market price are averaged with weights that sum to 1 and follow how much information each carries (inverse-variance weighting). So an agent with no clues follows the market, and experts lean on their own clues. Averaging rather than adding the price's information is deliberate: agents underweight what the price reveals ("cursed" traders, Eyster, Rabin & Vayanos 2019), which keeps them trading; fully rational traders would not trade (Milgrom & Stokey 1982). The cost is that a belief always lies between the agent's own evidence and the price, so no agent pushes the price past what it knows: the price settles near a weighted average of agents' posteriors (a linear opinion pool, as with log-utility traders, Wolfers & Zitzewitz 2006) and falls short of `P*`, which sums every clue. That shortfall is the aggregation loss the metrics measure; how much of the pool the informed hold sets its size. `eps` (for example 1e-9) only guards against 0/0. `mid` is the LMSR price under the AMM; in a CLOB it is the midpoint when both sides are quoted and the spread is at most a set cap (starting guess 10c), else the last trade price, and if nothing has traded yet the agent ignores the market (`w_market = 0`). `imbalance` is (bid depth - ask depth) / total depth near the best prices; it is 0 under the AMM.
2. **Sizes.** Fractional Kelly (Kelly 1956) sets a *target position*; the agent trades the difference from its current position. Kelly betting is log utility, under which the price is the wealth-weighted average belief (Wolfers & Zitzewitz 2006), which gives a theoretical benchmark.
   Steps 2 and 3 use *perceived* probabilities from the symmetric linear-in-log-odds weighting function (Goldstein & Einhorn 1987; Gonzalez & Wu 1999): `logit(p_yes) = g * logit(b)` and `p_no = 1 - p_yes`. With `g < 1` an agent overweights small probabilities, so it overpays for longshots and underpays for favorites; at the same `g` the longshot overweighting is close to Tversky & Kahneman's (1992) form. Unlike that form, perceived YES and NO always sum to 1 and a 50% belief stays at 50%, so weighting creates no no-trade band of its own near 50c, where the fee is largest, and E1 sees the fee's effect clearly. This is the misperception explanation of the favorite-longshot bias (Snowberg & Wolfers 2010) and the mechanism Burgi et al. (2025) propose for Kalshi.
3. **Acts.** Trade immediately (immediate-or-cancel, as deep in the book as the edge after fees still exceeds `min_edge`); otherwise, with probability `patience`, post a resting order `min_edge` inside its value, never crossing the book and sized to the gap between its Kelly target and its position plus resting orders; otherwise wait. Under the AMM, buy from the LMSR until its price reaches `p_yes - min_edge`.
4. **Maintains orders** (done right after the belief update, before acting, so an agent never rests an order and takes the other side in one turn). Cancel a resting order whose edge has fallen below `min_edge` (the picking-off risk), and cancel an order stuck several cents behind the best quote for many steps so it can be reposted (the non-execution risk). These are the two motives for cancellation in the microstructure literature (Copeland & Galai 1983; Liu 2009; Fong & Liu 2010). Agents notice stale orders only on their own turns, which models costly monitoring (Foucault, Roell & Sandas 2003).

**Agent traits:**

| Trait | Controls | Distribution |
|---|---|---|
| `n_clues` | private clues at setup (informedness) | negative binomial, heavy-tailed: about 40% know nothing, the top 10% know most of the pool |
| `w_hearsay` | weight on relayed clues | logit-normal, mean decreasing in `n_clues` (experts discount advice, Yaniv 2004) |
| `market_trust` | how many clues the agent thinks the price is worth | Gamma |
| `w_depth` | log-odds nudge from book imbalance | Beta, small by default |
| `noise` | belief error per turn | half-normal |
| `min_edge` | smallest edge worth acting on | half-normal |
| `patience` | chance of posting a resting order instead of waiting | logit-normal, mean increasing in `n_clues` |
| `kelly_frac` | risk tolerance: fraction of the Kelly bet placed | Beta, well below 1 |
| `g` | probability weighting (1 = none, below 1 = overweights longshots) | bounded between about 0.6 and 1, mean set from lab estimates (Tversky & Kahneman 1992; Gonzalez & Wu 1999) |
| `cash` | starting budget | lognormal |
| `activity` | how often the agent gets a turn | lognormal weights |

Bounded traits use bounded distributions, so no clipping is needed. Traits are independent except two links to informedness: `w_hearsay` falls and `patience` rises with `n_clues`. The patience link means that when news arrives, inexperienced agents take liquidity at once, while informed agents post resting orders at their value and let the price come to them. That matches retail traders' attention-driven buying (Barber & Odean 2008) and the informed trader who trades gradually (Kyle 1985). Uninformed agents are essential, not noise to remove: without them, informed agents have no one to trade with (Milgrom & Stokey 1982; Grossman & Stiglitz 1980).

**How values are set.** Each agent's traits are drawn from the model's random number generator at setup. Each seed yields the same population in every configuration. The distribution parameters are calibrated once with pattern-oriented modeling (Grimm et al. 2005): tune until the Kalshi default reproduces several real patterns at once (Kalshi trade sizes, spreads within the middle half of liquid Kalshi markets, a book that is rarely empty, informed agents earning more than uninformed ones, who lose), then freeze them for all experiments. The spread pattern only rules out wide spreads: the model's median spread is 1c, Kalshi's tightest quartile (Kalshi median 2c, IQR 1-6c), so it does not pin the spread down. Whether the price beats the average agent's own judgment (its posterior from its own clues alone) is reported, not required: because beliefs are averages, the price is only slightly more accurate (Brier better by 0.002 to 0.005 over 144-seed sets, price better in about half the seeds). `w_hearsay` and `w_depth` have no data anchor; the report names them as assumptions.

### 2.3 Manipulation strategies

The manipulator is an agent: it has the trait vector, a fixed set of private clues, and a large budget, and it pays fees. Its traits are set by the experimenter, not drawn. It follows the honest agent rules **plus a manipulation layer**, and it aims to maximize profit:

- **Pump and dump** (trade-based, Allen & Gale 1992): buy until the price is a set amount above its start, hold, then sell into the inflated price. It profits only if agents who follow the price buy from it.
- **Spoofing:** post a large fake bid a few ticks behind the best bid; sell real contracts while the imbalance lifts beliefs; cancel the fake bid; repeat. It profits only if agents read book depth, and it requires free cancellation.

**How effectiveness is measured.** In the E3 default configuration, the same agent with the same clues, budget, and traits trades honestly (the *honest twin*). **Manipulation premium = manipulator P&L - twin P&L on the same seed.** This separates strategy from information. Comparing with the average agent would not, because many agents have no clues and lose by design. Market harm (distortion from `P*` and money taken from other agents) is reported separately, since a manipulator can move the price and still lose money.

**Liquidity and responses.** Success depends on the market's liquidity and on how other agents respond. No extra configurations are needed: populations differ by seed, so book depth just before the manipulation starts varies across runs. Plot distortion against that depth. To show the responses, split the manipulator's trades by counterparty: informed agents trading against it (resistance) versus price-following or depth-reading agents trading with it (amplification).

### 2.4 Information layer

- **World and clues.** Draw the true outcome `Y` by coin flip. Make `M = 80` clues, each pointing to the truth with probability `q = 0.53`. Clues pointing the wrong way are honest misinformation and are already counted in `P*`. The first `K = 60` clues are the private pool: each agent draws `n_clues` (capped at `K`) distinct clues from it at random, so each private clue is held by about 65 of the 300 agents, mostly the informed few. The other 20 are released as scheduled news over the run. With these values about 94% of final `P*` fall between 10% and 90%, and one clue moves `P*` from 50% by about 3 points, above the 1.75c fee; the price moves less (Section 2.2).
- **Benchmark.** `logit(P*) = (# YES - # NO real clues known to any agent) x ln(q / (1 - q))`, from a fair-coin prior and clues independent given `Y`. Without disinformation, `P*` is the true posterior given all information in the market: it is calibrated by construction, and the price's Brier score splits exactly into `P*`'s unavoidable error plus the aggregation loss `E[(price - P*)^2]`. With disinformation, `P*` is the price the market would show if it filtered out every lie, not the true posterior: fakes always point against the truth, so they carry information about `Y` (a Bayesian who knew the source's rules would read news as correct with probability `q / (1 + r)` and could reverse it), and the Brier split no longer holds exactly.
- **Spreading.** Private clues never spread: an informed trader profits by trading on what others do not know (Kyle 1985), so private information reaches the price only through trading, and the network carries only news and disinformation. News is released to a few seed agents (5). Each step, an agent spreading a clue tells each neighbor with fixed probability `beta` (0.5 for now). An agent relays a new clue only after its next turn to trade (traders keep news until they have acted on it, van Bommel 2003). An agent stops retelling a clue once it tells someone who already knew it (stiflers, Daley & Kendall 1964). `beta` and mean degree are the same in every network; how fast each network spreads news is recorded as an outcome rather than equalized.
- **Disinformation.** A non-trading disinformation source deliberately fabricates clues pointing against the truth, added at a set ratio to true news (ratio x 20 fakes, released at evenly spaced steps) and seeded like true news. Each clue spreads with its own random numbers, so on the same seed the real news spreads identically with and without fakes. Agents cannot tell fakes apart; how much they believe any heard clue depends only on their own traits. Fakes are excluded from `P*`. Unlike random misinformation, disinformation is one-sided, so it does not cancel out (Wardle & Derakhshan 2017).
- **Networks** (each adds one property of real social networks):

| Network | Hubs | Clustering | Communities | Represents |
|---|---|---|---|---|
| Erdos-Renyi | - | - | - | null baseline |
| Watts-Strogatz | - | yes | - | friend circles plus a few long links |
| Barabasi-Albert | yes | - | - | influencers |
| Holme-Kim (default) | yes | yes | - | most realistic simple social network |
| Stochastic block model | - | within groups | yes | echo chambers |

### 2.5 One run

```
setup: draw Y and clues; build network; draw agent traits; place agents on nodes; hand out private clues
for t in 1..T:
    release scheduled news and disinformation to seed agents
    spread clues one step along the network
    market maker requotes
    activity-weighted sample of agents (and the manipulator, if present) take turns
    match orders, charge fees, record price, P*, volume, spread, depth
    (debug) assert conservation invariants
settle at 100c per winning contract; compute P&L for every agent
```

Starting defaults: 300 agents, T = 2000 steps, 100 seeds per configuration (never fewer than 50).

---

## 3. Metrics

| Metric | Question |
|---|---|
| Aggregation error `|price - P*|` over time | Did the market use the information it had? |
| Brier score of the final price | Was the forecast right? |
| Reaction half-life: steps until the gap to the new `P*` halves after a news release | How fast is news priced in? |
| Volume, spread, depth; no-trade and empty-book steps | Liquidity; did the market freeze? |
| Fees paid, by informedness and maker/taker | Who bears the fee? |
| P&L by `n_clues`, network position, and maker/taker | Who wins? |
| Steps for a clue to reach half the agents (E2) | How fast does each network spread news? |
| Disinformation distortion: signed gap to `P*` in the fakes' direction, minus the same seed's gap without disinformation (E2; the price already falls short of `P*` toward 50c, the fakes' side, about 7c) | How far do lies move the price? |
| Manipulation premium; peak and duration of distortion; money taken from others (E3) | Does manipulation pay, and does it harm the market? |

**Scored price.** "Price" in every metric is the same reference agents use (Section 2.2): the midpoint when both sides are quoted and the spread is within the cap, otherwise the last trade price.

**Statistics.** Each run is one data point. Report means with 95% confidence intervals, and report configuration comparisons as paired differences on shared seeds. Add seeds if intervals are too wide.

**Kalshi calibration check.** Pull resolved binary markets from Kalshi's public API, take each price a fixed time before close, bin by price, and plot the fraction resolving YES with confidence intervals. Do the same for the simulated default. The `g` distribution is set from lab estimates of probability weighting, not tuned to Kalshi, so a match is an out-of-sample test rather than a fit. To separate rules from behavior, rerun the default with `g = 1` for everyone. Any remaining bias then comes from Kalshi's rules alone (Snowberg & Wolfers 2010; Burgi et al. 2025). Public data does not show who traded; state this as a limitation.

---

## 4. Implementation steps

Market mechanics (`fees.py`, `lmsr.py`, `orderbook.py`) are plain Python with no Mesa dependency. Mesa covers agents, the model step, data collection, and `batch_run`. Every parameter lives in one `Config` dataclass in `src/kalshi_model/config.py`; results save the config they ran with. Figures are drawn from `results/*.csv`.

1. **Skeleton and fees.** Repo, `Config`, `fees.py` in integer cents. *Done when* fee tests pass (for example 1 contract at 50c pays 2c, 100 contracts at 5c pay 34c; `fee(P) == fee(100 - P)`).
2. **World and `P*`.** *Done when* `P*` matches hand-made cases and 10,000 simulated worlds show at least 90% of final `P*` between 10% and 90%.
3. **Agents and the AMM.** Trait population, belief rule, Kelly targets, LMSR, Mesa model. *Done when* LMSR tests pass (price in (0,1), path independence, loss at most `b ln 2`), a fully informed population drives the price to `P*`, and on default runs the price moves toward `P*` (it beats always guessing 50c) and informed agents earn more than uninformed ones.
4. **Order book, tests first.** Price-time priority, partial fills, free cancels that release reserved cash, YES/NO symmetry, no negative cash. Conservation invariants every step in debug mode: positions sum to 0; cash + fees is constant, before and after settlement (trades move cash between accounts, so no escrow is needed); every account can pay its worst case under both outcomes, with reserves matching its open orders. *Done when* tests pass, 50 runs show zero violations, and a zero-intelligence population reproduces Othman (2008) as a sanity check (his agents trade in a continuous double auction, so the check needs the order book).
5. **CLOB markets.** Market-maker component, agent cancel rules, Kalshi fees. *Done when* the market maker and agents stay solvent over T steps and empty-book steps are rare and reported (`experiments/clob_check.py`).
6. **Information layer.** Networks, spreading with trade-before-telling and stiflers, disinformation source. *Done when* a clue reaches everyone within the network diameter at `beta = 1` (with every agent trading each step, an agent at distance d learns it exactly d steps after release). `experiments/information_check.py` reports spread speed and distortion.
7. **Calibrate and freeze defaults.** Pattern-oriented tuning of the trait distributions (except `g`, and leaving informedness, `w_hearsay`, and `w_depth` as set) and the market maker against the Section 2.2 patterns, with real targets fetched from Kalshi's public API (`data/fetch_kalshi_patterns.py`), a rejection search (`experiments/calibrate.py`, results in `results/calibration.csv`): a Latin hypercube of candidates, keeping every one that meets the patterns, since pattern-oriented modeling finds a region of settings rather than one best point; the most central accepted setting that also passes on fresh seeds is frozen, and a spread-out set of accepted settings is kept for Step 11, and a check of the frozen values (`experiments/validate_defaults.py`). *Done when* the defaults are frozen in `Config`.
8. **Run E1 and E2.** Predictions written first.
9. **Manipulator; run E3.** Pump-and-dump and spoofing layers, honest twin. *Done when* invariants hold with the manipulator present.
10. **Kalshi calibration check** (Section 3), including the `g = 1` rerun.
11. **Robustness.** Rerun the key paired comparisons of E1 to E3 on the spread-out ensemble of accepted calibrations from Step 7 and report, for each effect, how often its sign holds (appendix figure).
12. **Write-up.** ODD protocol description (Grimm et al. 2020), three experiments with predictions and results, calibration check, limitations, future work. Real-world claims are framed as mechanisms that could explain aggregate patterns. Deliverables: tested code repo with README, an 8 to 12 page report, 5 to 7 key figures, and a short presentation.
