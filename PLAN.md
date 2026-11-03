# Project Plan: Kalshi as a Complex System

**An agent-based model of one binary prediction market. It asks how Kalshi's trading rules, the social network that carries news and disinformation, and manipulators change how fast and how accurately the market price finds the truth.**

Python 3.12 + Mesa 3.x (an agent-based modeling library). Code is in `src/kalshi_model/`, experiment scripts in `experiments/`, per-run results in `results/`. Every cited source has a one-line summary in `SOURCES.md`. Predictions are in `PREDICTIONS.md` and results in `RESULTS.md`.

**Status:** Steps 1 to 9 (Section 4) are done; Steps 10 to 12 remain.

---

## 1. Overview and research questions

### Background

**Prediction markets.** A prediction market sells contracts that pay $1 if an event happens (YES) and nothing otherwise (NO). The price of a YES contract, in cents, reads as the market's probability of the event: 70c means 70%. Classical theory says the price pools traders' scattered private information into a good probability (Hayek; Plott & Sunder; Hanson).

**Kalshi** is a real, US-regulated prediction market. Its rules differ from the textbook model:
- It uses a **central limit order book (CLOB)**, not an automated market maker. Traders post *limit orders* (buy or sell up to some quantity at a stated price); orders that do not trade at once *rest* in the book until filled or canceled.
- It charges the **taker** (the trader whose order trades against a resting order) a fee of `0.07 x C x P x (1 - P)` dollars, rounded up to the cent, for `C` contracts at price `P` (in dollars). The fee is largest at 50c. The **maker** (owner of the resting order) pays nothing in most markets.
- Canceling an order is free.

**Terms used throughout.**
- *Bid / ask:* the best (highest) resting buy price and best (lowest) resting sell price. *Spread* = ask - bid. *Midpoint* = their average. *Depth* = contracts resting near the best prices. *Book imbalance* = (bid depth - ask depth) / total depth.
- *Log-odds:* `logit(p) = ln(p / (1 - p))`. Independent evidence adds in log-odds.
- *Kelly betting:* bet sizing that maximizes expected log wealth (Kelly 1956). *Fractional Kelly* bets a fixed fraction of that.
- *Brier score:* `(forecast - outcome)^2`, lower is better; always guessing 50% scores 0.25.
- *P&L:* profit and loss.
- *Seed:* the number that fixes a run's random draws. Two configurations run on the same seed differ only in the setting changed, so comparisons are *paired* run by run.

### Research question and experiments

**Headline question.** How do Kalshi's market rules, the structure of the network that spreads information, and manipulators change the speed and accuracy with which a prediction market absorbs information?

**Benchmark.** Every run has a known best possible forecast, `P*`: the exact Bayesian probability of YES given every real clue held by any agent (Section 2.4). Prices are scored against it.

**Design.** Each experiment changes one variable around a shared **default configuration**: Kalshi rules, Holme-Kim network, no disinformation, no manipulator (disinformation is off so E1 and E3 test clean markets). Every configuration runs on the same 100 seeds (20000-20099). Predictions are written in `PREDICTIONS.md` before running and are not edited afterwards.

**E1. Market rules (2 configurations):** CLOB (order book, no fees, free cancels) vs. Kalshi (CLOB + Kalshi fees). The only difference is the fee, so E1 isolates its effect.
- *Predictions:* the fee creates a band around the price where no trade is worth its fee, widest at 50c, so prices near 50c react more slowly to small news; fees cut volume much more than accuracy; free maker orders push patient agents to post resting orders, and makers out-earn takers (as Burgi et al. 2025 find on real Kalshi data).

**E2. Information spread (12 configurations):**
- *Network sweep:* the 5 networks of Section 2.4, each at disinformation ratio 0.1 and at ratio 0 (so distortion is measured against the same network without fakes).
- *Disinformation sweep:* ratios 0, 0.05, 0.1, 0.2 on Holme-Kim, i.e. 0, 1, 2, 4 fake clues added to the 20 real news clues. A real clue points the right way on net by `2q - 1 = 0.06` per clue and a fake points the wrong way by 1, so at ratio 0.06 the news stream carries no net signal: 0.05 sits just below that break-even, and 0.1 and 0.2 are stress tests.
- *Predictions:* well-connected agents trade earlier and earn more (Ozsoylev et al. 2014); echo chambers (the block model) sustain the largest distortion; distortion grows faster than the ratio, because agents who follow the price amplify a rumor once it moves the price.

**E3. Manipulation (3 configurations):** Kalshi default with an honest manipulator twin vs. a pump-and-dump manipulator vs. a spoofing manipulator (Section 2.3).
- *Predictions:* pump-and-dump moves the price but loses money, because informed agents trade against it (Hanson & Oprea 2009); spoofing can profit because its fake orders never fill, to the extent agents read book depth (Wang & Wellman 2017). Both distort less, and for less time, in more liquid runs (Rasooly & Rozzi 2025).

**Validation.** Compare the simulated calibration curve with real Kalshi data, in particular the favorite-longshot bias (Section 3).

**Out of scope** (future work in the report): factorial designs crossing experiment variables; a separate agent-behavior experiment; liar agents who trade on their own disinformation; herding rules; disinformation targeted at hubs; PredictIt-style or flat fee regimes; LLM agents; an automated market maker (AMM) baseline. The LMSR AMM (Hanson's logarithmic market scoring rule) was dropped from E1; its code remains as a legacy feature (`market = "amm"`), unused by the experiments.

---

## 2. Model design and parameters

**Who is an agent.** *Agents* are the 300 traders. The *manipulator* is an agent with extra rules. The *market maker* (a quoting component in the order book) and the *information sources* (scheduled news and the disinformation source) are model components, not agents.

### 2.1 Markets

- **Contract:** pays 100c on YES, 0 on NO; integer prices 1 to 99c. All money is integer cents.
- **One YES book:** "buy NO at x" is treated as "sell YES at 100 - x". Positions are signed YES counts (negative = holding NO). Everything is fully collateralized: a resting order reserves its full cost plus worst-case fee, so no account can go negative.
- **CLOB:** price-time priority (best price first, then earliest), partial fills, free cancels. An order that would trade with its owner's own resting order cancels that resting order instead.
- **Market maker (CLOB only):** quotes 10 contracts on each side around a moving price estimate, cancels and requotes every step, shifts quotes to shed inventory, and widens its spread after losing to informed traders (Glosten & Milgrom 1985; Avellaneda & Stoikov 2008). Its quotes are post-only (never trade on arrival, so it is always the maker), its inventory is capped at 200 contracts, and it obeys the same collateral rules as agents.
- **Kalshi fees:** taker fee as in Section 1, maker fee 0, cancels free. Rounding is per order: an order filled in pieces pays what one fill of the same total would (as Kalshi's fee accumulator does). The fee hits small orders on cheap contracts hardest: one contract at 5c pays 1c, 20% of its cost.

### 2.2 Agents

Agents have no type labels. Each agent is a vector of traits set at setup, and behavior follows from the traits; "recreational bettor" and "expert" are regions of trait space. Uninformed agents are essential: without them informed agents have no one to trade with (Milgrom & Stokey 1982; Grossman & Stiglitz 1980).

**Each turn an agent:**

1. **Updates its belief `b`** (probability of YES), in log-odds:
   ```
   logit(b) = L_own + w_hearsay * L_heard + w_market * logit(mid) + w_depth * imbalance + noise * z
   w_market = market_trust / (market_trust + n_eff + eps),   n_eff = n_own + w_hearsay * n_heard
   ```
   - `L_own` is the net evidence `(# YES - # NO) x ln(q / (1 - q))` of the agent's own clues (its private clues and all true news it has learned); `L_heard` the same for fakes heard from neighbors; `n_own`, `n_heard` count those clues.
   - `mid` is the market price: the midpoint if both sides are quoted and the spread is at most 10c, else the last trade price. Before any trade the agent ignores the market (`w_market = 0`).
   - `imbalance` is book imbalance within 2c of the best prices. `z` is standard normal noise. `eps` (1e-9) only guards against 0/0.
   - **Why this form.** The agent treats the price as evidence from other traders and *adds* its own evidence to it. `w_market` is the share of the market's information the agent thinks it lacks: an agent with no clues follows the market, while an expert leans on its clues (inverse-variance weighting). Adding rather than averaging lets the price sum information no single agent holds, as `P*` does; an earlier averaging rule kept the price stuck within about 5c of 50c (Step 8b). Two departures from full rationality keep agents trading, since fully rational traders would not (Milgrom & Stokey 1982): they underweight what the price reveals (`w_market < 1`, "cursed" traders, Eyster, Rabin & Vayanos 2019), and they do not remove the part of their evidence the price already contains (correlation neglect, Enke & Zimmermann 2019). If `market_trust` is too high, agents amplify the price and it runs to 1c or 99c; the calibration patterns (end of Section 2.2) rule this out.
2. **Sizes a target position** by fractional Kelly on *perceived* probabilities: `logit(p_yes) = g * logit(b)`, `p_no = 1 - p_yes` (linear-in-log-odds weighting, Goldstein & Einhorn 1987; Gonzalez & Wu 1999). With `g < 1` an agent overweights small probabilities, overpaying for longshots and underpaying for favorites: the misperception explanation of the favorite-longshot bias (Snowberg & Wolfers 2010), proposed for Kalshi by Burgi et al. (2025). Perceived YES and NO always sum to 1 and 50% stays 50%, so weighting adds no no-trade band near 50c, where the fee is largest, and E1 sees the fee's effect cleanly. The agent trades toward the gap between target and current position (at least 1 contract). Under Kelly sizing the price settles near the wealth-weighted average belief (Wolfers & Zitzewitz 2006), a theoretical benchmark.
3. **Maintains its resting orders** (before acting, so it never rests an order and takes the other side in one turn). It cancels an order whose edge has fallen below `min_edge` (risk of being picked off by better-informed traders), and an order 3c or more behind the best quote for 20 steps, so it can repost (risk of never filling). These are the two motives for canceling in the literature (Copeland & Galai 1983; Liu 2009; Fong & Liu 2010). Agents notice stale orders only on their own turns, which models costly monitoring (Foucault, Roell & Sandas 2003).
4. **Acts.** If trading now beats its perceived value by at least `min_edge` after fees, it trades immediately (immediate-or-cancel, as deep in the book as that holds). Otherwise, with probability `patience`, it posts a resting order `min_edge` inside its value, never crossing the book, sized to the gap between target and position plus existing resting orders. Otherwise it waits.

**Turns.** Each step, 20 agents are drawn without replacement, each equally likely.

**Traits.** Heterogeneity is added only where a mechanism needs it. Two traits vary across agents:

| Trait | Controls | Distribution |
|---|---|---|
| `n_clues` | private clues held (informedness) | negative binomial (mean 20, capped at 60): about 40% hold none, the top 10% hold most of the pool |
| `cash` | starting budget | lognormal, rank-correlated with `n_clues` (Gaussian copula, `cash_rho`) |

Informedness must vary, since informed traders need uninformed ones to trade with. Cash varies because wealth is heavy-tailed, which sets the spread of trade sizes. It is linked to informedness because traders with better beliefs grow wealth over many markets, so the informed come to hold more of the money (Blume & Easley 2006; Kets et al. 2014). Only who holds the cash changes, not its overall distribution. Without this link the price follows the few richest agents (the top tenth hold about three quarters of the cash), who are mostly uninformed. The link is drawn after the other traits, so they are the same on each seed with or without it.

Every other trait is one value shared by all agents, since each works through its level, not its spread:

| Trait | Controls | Value |
|---|---|---|
| `w_hearsay` | weight on fakes heard from neighbors | 0.39 (no data anchor) |
| `market_trust` | how many clues the agent thinks the price is worth | 49 |
| `w_depth` | log-odds nudge from book imbalance | 0.1 (no data anchor) |
| `noise` | belief error per turn (log-odds) | 0.033 |
| `min_edge` | smallest edge (cents) worth acting on | 2.07c |
| `patience` | chance of posting a resting order instead of waiting | 0.75 |
| `kelly_frac` | fraction of the Kelly bet placed | 0.325 |
| `g` | probability weighting (1 = none, below 1 = overweights longshots) | 0.8, from lab estimates (Tversky & Kahneman 1992; Gonzalez & Wu 1999) |

These values are the population means of the Step 8b trait distributions, pending recalibration (Step 8c). Any trait can still be pinned per agent in a custom configuration.

**How values are set (calibration).** The cash distribution and its link, the single trait values (except `g`, `w_hearsay`, and `w_depth`), and the market maker's size and spread are calibrated once with **pattern-oriented modeling** (Grimm et al. 2005): search for settings under which the Kalshi default reproduces several real patterns at once, then freeze them for all experiments. Real targets come from Kalshi's public API. The patterns:

1. **Trade sizes** match Kalshi's.
2. **Spreads** fall within the middle half of liquid Kalshi markets (median 2c, interquartile range 1-6c). This only rules out wide spreads: the model's median is 1c.
3. **The book is rarely empty.**
4. **Informed agents earn more than uninformed ones, who lose.** P&L is valued at final `P*` rather than the outcome: without disinformation the two have the same expectation, but outcome luck makes realized P&L about six times as noisy across seeds.
5. **The price is calibrated:** regressing `logit(P*)` on the logit of the price over the second half of each run gives a slope between 0.8 and 1.4. Without disinformation `P*` stands in for the outcome with far less noise. A slope above 1 means an underconfident price (too close to 50c). Real prices are roughly calibrated near resolution (Page & Clemen 2013) and, where they miss, are underconfident (the favorite-longshot bias, Burgi et al. 2025 on Kalshi), so the band reaches further above 1. Only this overall slope is fitted; the bias at the extremes stays an out-of-sample test (Section 3).
6. **The price beats the average agent's own judgment** (Brier score versus each agent's posterior from its own clues alone).

The search (`experiments/calibrate.py`, `results/calibration.csv`) draws candidates by Latin hypercube and keeps every one that meets all patterns, since pattern-oriented modeling finds a region of settings rather than one best point. The most central accepted setting that also passes on fresh seeds is frozen; a spread-out set of accepted settings is kept for robustness checks (Step 11). `g` is set from lab estimates, informedness by design, and `w_hearsay` and `w_depth` have no data anchor; the report names these, and the network settings, as assumptions.

### 2.3 Manipulator (E3)

The manipulator is an agent with the trait vector, a fixed set of private clues, and a large budget; it pays fees. Its traits are set, not drawn: the population's single values, with median informedness (2 private clues), except that it ignores book imbalance (it knows which depth is its own). Its budget is 5% of all agents' cash on that seed. It trades every step, is not on the social network (it learns news only through the price), and uses its own random stream, so the world, the population, and agents' turns are identical with or without it. It follows the honest agent rules **plus a manipulation layer** from step 1000 (order book only), aiming to maximize profit:

- **Pump and dump** (Allen & Gale 1992): buy, taking up to 2,000 contracts a turn, until the price is 5c above its starting level (at most 100 steps); hold 50 steps; then sell what it bought into bids no lower than the starting price (at most 100 steps). It profits only if price-following agents buy from it.
- **Spoofing:** for 500 steps, keep a fake bid of 20,000 contracts 2c behind the best bid (moved every turn as the best bid moves, Wang & Wellman 2017), and a real offer of 2,000 contracts at the best ask, reposted when it fills or the ask moves. The fake bid inflates the imbalance agents read, pushing their beliefs up so they buy its offer. It profits only if agents read depth, and it needs free cancellation. The fake bid can still fill if a large seller sweeps the book between the spoofer's turns; at 1c behind it filled far more often than at 2c, the edge of the 2c imbalance window.

**Measuring effectiveness.** In the E3 baseline, the same agent (same clues, budget, traits) trades honestly throughout: the *honest twin*. **Manipulation premium = manipulator P&L - twin P&L on the same seed**, valued at final `P*`. This separates strategy from information; comparing with the average agent would not, since many agents have no clues and lose by design. The three E3 configurations are identical until step 1000 and share `P*`, so *distortion* is the price minus the twin run's price on the same seed, step by step. Market harm (distortion, money taken from other agents) is reported separately, since a manipulator can move the price and still lose money.

**Liquidity and responses.** Book depth just before step 1000 varies across seeds, so distortion is plotted against it (no extra configurations). The manipulator's trades are split by counterparty: informed agents trading against it (resistance) versus price-following or depth-reading agents trading with it (amplification).

### 2.4 Information layer

**World, clues, and the benchmark `P*`.**

- **World.** The true outcome `Y` (YES or NO) is a coin flip. There are `M = 80` clues, each pointing to the truth with probability `q = 0.53`, independently given `Y`. Clues pointing the wrong way are honest noise and are counted in `P*`.
- **Private clues.** The first `K = 60` clues form the private pool. Each agent draws `n_clues` distinct ones at random at setup, so each private clue is held by about 65 agents, mostly the well-informed few.
- **News.** The other 20 clues are released over the run as scheduled news (see Spreading below).
- **`P*`:** `logit(P*) = (# YES - # NO among real clues known to any agent) x ln(q / (1 - q))`. With these values one clue moves `P*` from 50% by about 3 points (more than the 1.75c fee at 50c), and about 94% of final `P*` values fall between 10% and 90%.
- **What `P*` means.** Without disinformation, `P*` is the true posterior given all information in the market, so it is calibrated by construction, and the price's Brier score splits exactly into `P*`'s unavoidable error plus the aggregation loss `E[(price - P*)^2]`. With disinformation, `P*` is what the market would show if it filtered out every lie. It is not then the true posterior (fakes always point against the truth, so a Bayesian who knew the source's rules could learn from them), and the Brier split no longer holds exactly.

**Spreading and disinformation.**

- **Private clues never spread.** An informed trader profits from what others do not know (Kyle 1985), so private information reaches the price only through trading. The network carries only news and fakes.
- **Seeding and spreading.** Each news clue is given to 5 random seed agents at release. Each step, an agent spreading a clue tells each neighbor with probability `beta = 0.5`. An agent starts relaying a new clue only after its next turn to trade (traders act on news before passing it on, van Bommel 2003). An agent stops retelling a clue once it tells someone who already knew it (a "stifler", Daley & Kendall 1964).
- **Disinformation.** A non-trading source fabricates clues that point against the truth, `ratio x 20` of them, released at evenly spaced steps and seeded like news. Each clue spreads with its own random numbers, so on the same seed the real news spreads identically with and without fakes. Fakes are excluded from `P*`. Unlike random noise, disinformation is one-sided, so it does not cancel out (Wardle & Derakhshan 2017).
- **How agents weigh what they hear.** Agents cannot tell a fake from news. But true news has a source that can be checked, so news counts in full however the agent learned it, while a fake heard from a neighbor counts only as hearsay, at the agent's weight `w_hearsay`. Each clue counts once however many neighbors relay it.
- **Networks.** Mean degree 8 and `beta` are the same in all networks; how fast each spreads news is measured, not equalized.

| Network | Hubs | Clustering | Communities | Represents |
|---|---|---|---|---|
| Erdos-Renyi | - | - | - | random baseline |
| Watts-Strogatz | - | yes | - | friend circles plus a few long links |
| Barabasi-Albert | yes | - | - | influencers |
| Holme-Kim (default) | yes | yes | - | most realistic simple social network |
| Stochastic block model | - | within groups | yes | echo chambers (2 groups, 5% of links between) |

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

Defaults: 300 agents, T = 2000 steps, 100 seeds per configuration (never fewer than 50). All parameters live in one `Config` dataclass (`src/kalshi_model/config.py`); results save the config they ran with.

---

## 3. Metrics

| Metric | Question |
|---|---|
| Aggregation error `|price - P*|` over time | Did the market use the information it had? |
| Brier score of the final price | Was the forecast right? |
| News response: mean price move toward a news clue over the 300 steps after release, and its half-life (steps until the mean move reaches half its 300-step size; the price may never close the full gap to `P*`) | How fast is news priced in? |
| Volume, spread, depth; no-trade and empty-book steps | Liquidity; did the market freeze? |
| Fees paid, by informedness and maker/taker | Who bears the fee? |
| P&L by `n_clues`, network position, and maker/taker | Who wins? |
| Steps for a clue to reach half the agents (E2) | How fast does each network spread news? |
| Disinformation distortion: signed gap from `P*` in the fakes' direction, minus the same seed's gap without fakes (E2; even without fakes the price falls short of `P*` toward 50c, which is the fakes' side) | How far do lies move the price? |
| Manipulation premium; distortion while the layer acts and over the 300 steps after; peak of the mean distortion curve and steps until it falls below half that peak; money taken from others (E3) | Does manipulation pay, and does it harm the market? |

**Scored price.** "Price" in every metric is what agents read: the midpoint when both sides are quoted and the spread is at most 10c, otherwise the last trade price.

**Statistics.** Each run is one data point. Report means with 95% confidence intervals, and configuration comparisons as paired differences on shared seeds. Add seeds if intervals are too wide.

**Kalshi calibration check.** Pull resolved binary markets from Kalshi's public API, take each price a fixed time before close, bin by price, and plot the fraction resolving YES with confidence intervals; do the same for the simulated default. Since `g` comes from lab estimates, not Kalshi, a match is an out-of-sample test. To separate rules from behavior, rerun the default with `g = 1` for everyone; any remaining bias comes from the market rules alone (Snowberg & Wolfers 2010; Burgi et al. 2025). Limitation: public data does not show who traded.

---

## 4. Implementation steps

Market mechanics (`fees.py`, `orderbook.py`, and `lmsr.py` for the legacy AMM) are plain Python with no Mesa dependency; Mesa covers agents, the model step, data collection, and batch runs. Figures are drawn from `results/*.csv`; `experiments/plot_run.py` charts one seed's price against `P*` with clue releases, volume, and spread, plus its network colored by each agent's P&L.

1. **Skeleton and fees** (done). Repo, `Config`, `fees.py` in integer cents. *Done when* fee tests pass (e.g. 1 contract at 50c pays 2c, 100 contracts at 5c pay 34c, `fee(P) == fee(100 - P)`).
2. **World and `P*`** (done). *Done when* `P*` matches hand-made cases and in 10,000 simulated worlds at least 90% of final `P*` fall between 10% and 90%.
3. **Agents and the AMM** (done; the AMM is now a legacy feature, dropped from E1). Traits, belief rule, Kelly targets, LMSR, Mesa model. *Done when* LMSR tests pass (price in (0,1), path independence, loss at most `b ln 2`); a fully informed population with `market_trust = 0` drives the price to `P*` (with trust above 0, agents who all know everything double count it through the price); on default runs the price beats always guessing 50c and informed agents out-earn uninformed ones.
4. **Order book, tests first** (done). Price-time priority, partial fills, free cancels that release reserved cash, YES/NO symmetry, no negative cash. **Conservation invariants**, asserted every step in debug mode and in tests: positions sum to 0; total cash + fees is constant, before and after settlement; every account can pay its worst case under both outcomes, with reserves matching its open orders. *Done when* tests pass, 50 runs show zero violations, and zero-intelligence traders in the order book reproduce Othman (2008) as a sanity check.
5. **CLOB markets** (done). Market maker, agent cancel rules, Kalshi fees. *Done when* the market maker and agents stay solvent over T steps and empty-book steps are rare and reported (`experiments/clob_check.py`).
6. **Information layer** (done). Networks, spreading with trade-before-telling and stiflers, disinformation source. *Done when* at `beta = 1` with every agent trading each step, an agent at network distance d learns a clue exactly d steps after release (`experiments/information_check.py` reports spread speed and distortion).
7. **Calibrate and freeze defaults** (done). The calibration at the end of Section 2.2, with targets fetched by `data/fetch_kalshi_patterns.py` and the frozen values checked on fresh seeds by `experiments/validate_defaults.py`. *Done when* the defaults are frozen in `Config`.
8. **Run E1 and E2** (done). Predictions first, then `experiments/e1_rules.py` and `experiments/e2_information.py` write one row per run to `results/`. *Done when* every run passes the invariants and `RESULTS.md` reports each prediction against the data.
8b. **Fix price aggregation, then redo Steps 7 and 8** (done). Step 8 found the price compressed toward 50c: the final price moved about 0.2 logit units per unit of `P*`, so a 60c price meant about 88% YES, and E1 and E2 could not speak to accuracy. Causes and fixes, found on unused seeds:
    - Agents *averaged* their evidence with the price, so the price settled near an average of agents' evidence rather than its sum. Fix: the additive belief rule (Section 2.2).
    - Cash was unrelated to information, so the price followed rich uninformed agents. Fix: the cash link (`cash_rho`).
    - With these, the informed held most of the cash and traded with each other at a loss. Fix: noise falling with informedness (`noise_delta`). Trust rising with informedness did not help, and letting patient agents post before taking only got their orders picked off.
    - Realized P&L was too noisy to judge the informed-earn pattern on 48 seeds (95% intervals about $400 per agent). Fix: value it at final `P*` (about $60).
    - Every candidate meeting the other patterns had a slope of 1.2 to 1.3, since the informed earn only when the price leaves them an edge (Grossman & Stiglitz 1980). Fix: widen the slope band from 0.8-1.25 to 0.8-1.4, in the underconfident direction real markets show.
    - The price ignored news (weighting a news clue at 0.21 of its value versus 0.84 for a private clue), because news reached most agents secondhand as hearsay, which the cash-rich informed discount. Fix: true news counts as own evidence; only fakes are hearsay.

    Then recalibrate with patterns 5 and 6 required and `cash_rho`, `noise_delta`, and wider `market_trust` and `noise_scale` ranges in the search, and rerun E1 and E2 on the same seeds. The final search accepted 101 of 1,500 candidates; on 480 fresh seeds the frozen setting's slope is 1.10 to 1.18, and the informed earn on three of four seed sets (+$66 per agent pooled). `PREDICTIONS.md` stays unchanged (its predictions concern mechanisms); `RESULTS.md` keeps a short record of the first model's results and why the model changed. *Done when* the frozen default passes every pattern on fresh seeds and `RESULTS.md` reports each prediction for the revised model.
8c. **Simplify the traits** (in progress). Only informedness and cash vary across agents; every other trait is one value (Section 2.2), so the report justifies two distributions instead of eleven, and the links of `w_hearsay`, `noise`, and `patience` to informedness are gone. The single values start at the means of the Step 8b distributions. The `noise` link was added in Step 8b so the informed would have noise traders to profit from. Before recalibration, on 48 fresh seeds, every pattern passes except informed-earn: the best-informed tenth lose $91 per agent, and agents with no clues about break even. Then recalibrate over the new search space (`calibration.SPACE`) and rerun E1 to E3. *Done when* the frozen default passes every pattern on fresh seeds and `RESULTS.md` reports each prediction for the simplified model.
9. **Manipulator; run E3** (done). Pump-and-dump and spoofing layers and the honest twin (`src/kalshi_model/manipulator.py`); predictions first, then `experiments/e3_manipulation.py` writes `results/e3_manipulation.csv`. *Done when* invariants hold with the manipulator present and `RESULTS.md` reports each E3 prediction against the data.
10. **Kalshi calibration check** (Section 3), including the `g = 1` rerun.
11. **Robustness.** Rerun the key paired comparisons of E1 to E3 on the spread-out set of accepted calibrations from Step 7 and report how often each effect's sign holds (appendix figure).
12. **Write-up.** ODD protocol model description (the standard format for agent-based models, Grimm et al. 2020), the three experiments with predictions and results, the calibration check, limitations, and future work. Real-world claims are framed as mechanisms that could explain aggregate patterns. Deliverables: tested code repo with README, an 8 to 12 page report, 5 to 7 key figures, and a short presentation.
