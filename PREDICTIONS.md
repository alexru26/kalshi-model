# Predictions

Written before the experiments ran, and committed before any results, so the outcome cannot shape them. Each prediction names the metric it is judged on (PLAN.md Section 3; definitions in `src/kalshi_model/metrics.py`). Every configuration runs on seeds 20000-20099, which no earlier step used, and comparisons are paired seed by seed with 95% confidence intervals. Results are in `RESULTS.md`.

**What was known beforehand.** Earlier step checks showed the Kalshi book has a narrower median spread than the fee-free CLOB (1c vs 2.7c, Step 5), news reaches half the agents in about 22-26 steps in every network except Watts-Strogatz (about 42, Step 6), and on 10 seeds 1, 2, and 4 fakes moved the price about 0.4, 0.9, and 1.9c toward them (Step 6). A 6-seed probe (seeds 10000-10005) checked that the news-response metric is measurable in all three markets.

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

## Amendments

Changes made after the predictions were committed, each before the full runs. The scripts were smoke-tested on 3 seeds (20000-20002, part of the experiment set) before the full runs.

- **E1.3 metric** (after a 3-seed smoke test of the scripts). The first version compared agents' own maker share: of the contracts agents traded, the share they filled as makers. That share is about 0.5 in both books by construction, since most trades are between two agents, one the maker and one the taker. It now compares the share of all contracts whose maker is an agent rather than the market maker, which measures how much liquidity agents supply.
