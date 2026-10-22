# Predictions

Written before the experiments ran, and committed before any results, so the outcome cannot shape them. Each prediction names the metric it is judged on (PLAN.md Section 3; definitions in `src/kalshi_model/metrics.py`). Every configuration runs on seeds 20000-20099, which no earlier step used, and comparisons are paired seed by seed with 95% confidence intervals. Results are in `RESULTS.md`.

**What was known beforehand.** Earlier step checks showed the Kalshi book has a narrower median spread than the fee-free CLOB (1c vs 2.7c, Step 5), news reaches half the agents in about 22-26 steps in every network except Watts-Strogatz (about 42, Step 6), and on 10 seeds 1, 2, and 4 fakes moved the price about 0.4, 0.9, and 1.9c toward them (Step 6). A 6-seed probe (seeds 10000-10005) checked that the news-response metric is measurable in all three markets. For E3, a probe on 24 unused seeds (40000-40023) chose the spoofer's distance behind the best bid (2c rather than 1c, where large sellers filled the fake bid far more often) and checked the script: there the pump's premium at `P*` was about -$3,000 and the spoofer's about $0, and the price sat about 1.8c (pump) and 0.5c (spoof) above the twin's while the layer acted.

## E1. Kalshi rules (AMM vs CLOB vs Kalshi)

- **E1.1 The fee slows the pricing of news, most near 50c.** Kalshi's mean price move toward a news clue over the 90 steps after its release is smaller than the CLOB's, and its news half-life is longer. The shortfall is larger for news that arrives while the price is between 40c and 60c than for news that arrives outside that range, because the fee per contract is largest at 50c.
- **E1.2 Fees cut volume much more than accuracy.** The percentage fall in total volume from the CLOB to Kalshi is more than twice the percentage change in mean aggregation error `|price - P*|`.
- **E1.3 Free maker orders draw agents into making.** The share of traded contracts whose maker (the resting order) is an agent rather than the market maker is higher in Kalshi than in the CLOB.
- **E1.4 Makers out-earn takers on Kalshi** (as Burgi et al. 2025 find on real Kalshi data). Agents' maker fills earn more per contract than their taker fills after fees, and taker fills lose money on average.
- **E1.5 The fee falls hardest on the uninformed.** On Kalshi, agents with no private clues pay more fee per contract traded than the best-informed tenth, since patience rises with informedness and patient agents post free maker orders.
- **E1.6 The AMM prices news fastest.** The LMSR always quotes and every trade moves its price, so the AMM's news half-life is shorter than the CLOB's. The LMSR loses money on average (its subsidy, at most `b ln 2`, $34.66 per run).

Reported without a prediction: aggregation error and final Brier score across the three markets, spread, depth, and empty-book and no-trade steps.

## E2. Information spread

Network sweep: 5 networks at disinformation ratio 0.1. Disinformation sweep: ratios 0, 0.05, 0.1, 0.2 on Holme-Kim. Distortion is measured against the same seed and network with no disinformation, so each network also runs at ratio 0.

- **E2.1 Watts-Strogatz spreads news slowest** (a replication of Step 6): its mean steps for a news clue to reach half the agents is the largest of the five networks.
- **E2.2 Slower spread means slower pricing.** The networks' news half-lives follow their spread speeds, with Watts-Strogatz the slowest to price news.
- **E2.3 Well-connected agents hear news earlier and earn more** (Ozsoylev et al. 2014). The top quarter of agents by degree learn news sooner and earn more than the bottom quarter, most clearly in the hub networks (Barabasi-Albert and Holme-Kim). Agents sit on nodes at random, so degree is independent of their traits.
- **E2.4 Echo chambers sustain the largest distortion.** At ratio 0.1 the stochastic block model has the largest disinformation distortion of the five networks, both over the whole run and in its last quarter.
- **E2.5 Distortion grows faster than the ratio.** Distortion per fake clue rises with the ratio: distortion at 0.2 divided by 4 fakes exceeds distortion at 0.1 divided by 2, which exceeds distortion at 0.05 (1 fake), because agents who follow the price amplify a rumor once it moves the price.

## E3. Manipulation

Honest twin vs pump and dump vs spoofing, all on the Kalshi default with the same manipulator agent (PLAN.md Section 2.3). Each seed runs the three together; distortion is the price minus the twin's price on the same seed, and the manipulation premium is the manipulator's P&L minus the twin's, both valued at final `P*`.

- **E3.1 Pump and dump moves the price but loses money** (Hanson & Oprea 2009). While the layer acts, the price is above the twin's (mean distortion above 0), and the premium is negative, because agents with clues trade against it: the best-informed tenth's share of the contracts sold to the pump exceeds their share of all contracts traded over the same steps in the twin's run.
- **E3.2 Spoofing profits, to the extent agents read book depth** (Wang & Wellman 2017). While the layer acts, the price is above the twin's, the premium is positive, and the agents who buy the spoofer's real offers read depth more than agent buyers in general: their contract-weighted `w_depth` exceeds that of every agent buyer over the same steps in the twin's run.
- **E3.3 More liquid runs distort less, and for less time** (Rasooly & Rozzi 2025). Across seeds, book depth within 2c just before the layer starts is negatively correlated (Spearman) with the distortion while the layer acts and with the distortion over the 300 steps after it stops, for both strategies.

Reported without a prediction: the peak of the mean distortion curve and the steps until it falls below half that peak, money taken from agents and from the market maker, the change in fees, how often the pump reaches its target and the fake bid fills, and how the twin's presence alone changes the Kalshi default (against E1 on the same seeds).

## Amendments

Changes made after the predictions were committed, each before the full runs unless noted. The scripts were smoke-tested on 3 seeds (20000-20002, part of the experiment set) before the full runs.

- **E1.3 metric** (after a 3-seed smoke test of the scripts). The first version compared agents' own maker share: of the contracts agents traded, the share they filled as makers. That share is about 0.5 in both books by construction, since most trades are between two agents, one the maker and one the taker. It now compares the share of all contracts whose maker is an agent rather than the market maker, which measures how much liquidity agents supply.
- **Degree quarters** (after the full runs). The top and bottom quarters by degree (E2.3) were chosen by quantile, so tied degrees made the groups larger than a quarter and, in Watts-Strogatz, overlap. They are now exactly a quarter each, ranked by degree with ties broken by agent index, and both experiments were rerun.
- **Step 8b model revision** (after the first full runs). The first model's price stayed too close to 50c to test price accuracy, so the belief rule now adds the price to an agent's own evidence instead of averaging the two, and starting cash rises with informedness (PLAN.md Step 8b). The predictions above were not changed. The news window grew from 90 to 300 steps because the price was still moving at 90; E1.1's "90 steps" now means the 300-step window. The first model's results are kept in `RESULTS.md`.
- **Relayed news counts as own evidence** (Step 8b, after the second full runs). Single-run charts showed the price ignoring news: true news reached nearly every agent as hearsay, which the cash-rich informed discount. A relayed true news clue now counts as the listener's own evidence, since it can be checked at its source; fakes stay hearsay. The predictions above were not changed, and the calibration and both experiments were rerun.
- **E3 reporting** (after the first full E3 run; the E3 script was smoke-tested on unused seeds 40000-40023, not on the experiment set). Two reported measures were added, with the predictions and their metrics unchanged: a per-seed interval on the informed tenth's share of the manipulator's counterparties (it was pooled only), and book depth averaged over the 100 steps before the layer starts, beside the single-step depth E3.3 is judged on.
