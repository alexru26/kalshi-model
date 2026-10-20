# Results

Each prediction in `PREDICTIONS.md` against the data. Every configuration ran on seeds 20000-20099 with the conservation invariants checked every step (no violations). Values are means over seeds with 95% intervals: `+-` is a normal interval on the mean or on a paired difference, `[low, high]` a bootstrap interval over seeds (paired across configurations). The per-run data are in `results/e1_rules.csv` and `results/e2_information.csv`. The JSON file beside each records the commit, seeds, and full configurations. `uv run python experiments/e1_rules.py summary` (or `e2_information.py summary`) reprints every number here from those files.

Context for every result: the price stays close to 50c (final prices average 5c from it) and well short of `P*`, as the averaging belief rule implies (PLAN.md Section 2.2). The mean gap `|price - P*|` is about 14c in every market, and a news clue that moves `P*` by 2.4c moves the price by under 1c within 90 steps. Final Brier scores are 0.223-0.227, against 0.185 for `P*` and 0.25 for a constant 50c.

## E1. Kalshi rules

| Metric | AMM | CLOB | Kalshi |
|---|---|---|---|
| Mean `|price - P*|` (c) | 14.13 +- 1.56 | 13.94 +- 1.59 | 14.21 +- 1.64 |
| Final Brier | 0.227 +- 0.010 | 0.224 +- 0.010 | 0.223 +- 0.010 |
| News move by lag 90 (c) | 0.41 +- 0.22 | 0.85 +- 0.13 | 0.67 +- 0.08 |
| News half-life (steps) | 19.8 [14.7, 26.5] | 26.5 [22.5, 31.2] | 39.7 [36.1, 43.3] |
| Volume (contracts) | 84,818 | 495,984 | 217,969 |
| No-trade steps | 2% | 20% | 65% |
| Median spread (c) | - | 2.07 | 1.00 |
| Depth within 2c (contracts) | - | 8,975 | 27,840 |
| Empty-book steps | - | 3.1% | 0.4% |
| LMSR or market-maker P&L ($/run) | +51.2 +- 1.4 | +234 +- 30 | +35 +- 9 |
| P&L, no clues / top tenth ($/agent) | -20 / +77 | -71 / +217 | -80 / +205 |

| Prediction | Verdict | Evidence |
|---|---|---|
| E1.1 The fee slows the pricing of news, most near 50c | **Partly held** | Kalshi's 90-step news move is 0.19 +- 0.14c smaller than the CLOB's and its half-life is 13.2 [7.2, 18.7] steps longer. The near-50c part could not be tested: only 2.2 of each run's 20 news clues arrive outside 40-60c, and the shortfall is, if anything, larger there (1.18 +- 1.33c) than near 50c (0.17 +- 0.16c). |
| E1.2 Fees cut volume much more than accuracy | **Held** | From the CLOB to Kalshi, volume falls 56.1% [53.9, 58.2] and mean aggregation error rises 1.9% [0.7, 3.0], about 30 times less. Final Brier scores do not differ (0.000 +- 0.004). |
| E1.3 Free maker orders draw agents into making | **Held, small** | Agents are the maker on 99.1% of Kalshi contracts vs 96.9% in the CLOB (+2.2 +- 0.3 points). Agents already supply nearly all liquidity without fees; the larger change is in the book itself: half the spread and 3.1 times the depth. |
| E1.4 Makers out-earn takers on Kalshi | **Failed** | Per contract, agents' maker fills earn -0.86 +- 1.27c and their taker fills -0.88 +- 1.25c after fees, a gap of 0.02 +- 2.52c. Before fees, takers earn about +0.8c per contract (not significantly above 0) and the 1.7c taker fee removes it. Unlike on real Kalshi, informed agents take when news arrives, likely picking off resting orders whose owners only update them on their own turns. |
| E1.5 The fee falls hardest on the uninformed | **Failed (reversed)** | Agents with no clues pay 0.78 +- 0.04c in fees per contract traded and the top tenth 0.91 +- 0.08c (difference -0.13 +- 0.10c). The informed take liquidity more often, because their edges are large enough to clear the fee. |
| E1.6 The AMM prices news fastest | **Failed** | The AMM's half-life is 6.6 [-13.6, 1.3] steps shorter than the CLOB's (not significant), and its 90-step move is half the CLOB's (0.41 vs 0.85c). The LMSR gains $51.2 +- 1.4 per run instead of losing its subsidy: uninformed agents' losses to it exceed what informed agents win. |

## E2. Information spread

| Network (ratio 0.1) | Steps to half reach | News half-life (steps) | Distortion (c) | Last quarter (c) | P&L, top - bottom degree quarter ($) |
|---|---|---|---|---|---|
| Erdos-Renyi | 23.2 +- 0.4 | 39.4 [36.5, 46.4] | 0.80 +- 0.07 | 1.77 +- 0.16 | +22.8 +- 46.7 |
| Watts-Strogatz | 41.8 +- 0.5 | 49.4 [45.7, 53.6] | 0.83 +- 0.06 | 1.79 +- 0.16 | +2.0 +- 33.7 |
| Barabasi-Albert | 20.9 +- 0.5 | 41.7 [38.6, 44.9] | 0.80 +- 0.07 | 1.71 +- 0.15 | +20.9 +- 43.5 |
| Holme-Kim | 21.5 +- 0.5 | 41.3 [37.9, 44.4] | 0.81 +- 0.07 | 1.69 +- 0.17 | +12.4 +- 42.4 |
| Stochastic block model | 25.3 +- 0.4 | 44.4 [39.9, 48.6] | 0.80 +- 0.08 | 1.75 +- 0.17 | +32.6 +- 48.0 |

| Holme-Kim ratio | Fakes | Distortion (c) | Per fake (c) |
|---|---|---|---|
| 0.05 | 1 | 0.43 +- 0.05 | 0.43 +- 0.05 |
| 0.1 | 2 | 0.81 +- 0.07 | 0.40 +- 0.04 |
| 0.2 | 4 | 1.65 +- 0.10 | 0.41 +- 0.03 |

| Prediction | Verdict | Evidence |
|---|---|---|
| E2.1 Watts-Strogatz spreads news slowest | **Held** | 41.8 steps to half reach vs 20.9-25.3 elsewhere; paired gaps of 16.5 to 20.9 steps, each +- 0.7 or less. |
| E2.2 Slower spread means slower pricing | **Held** | Watts-Strogatz has the longest news half-life, 5.0 [0.6, 9.8] to 10.0 [3.0, 14.5] steps longer than each other network. The stochastic block model is second slowest on both. Among the three fastest spreaders the half-lives do not differ significantly. |
| E2.3 Well-connected agents hear news earlier and earn more | **Partly held** | The top degree quarter learns each news clue 12-14 steps before the bottom quarter (3 steps in Watts-Strogatz, whose degrees barely vary), but earns only $18 +- 19 more across all five networks, with no network significant. News is a small part of what agents know, and one clue moves the price by under 1c, so hearing it early is worth little next to the P&L spread between runs. |
| E2.4 Echo chambers sustain the largest distortion | **Failed** | Distortion is 0.80-0.83c in all five networks; the stochastic block model differs from each other network by at most 0.03 +- 0.09c (0.06 +- 0.21c in the last quarter). By the end of a run 95-99% of agents know each fake in every network (checked on 6 seeds per network), even with only 5% of links between the two groups, so how fast a rumor travels does not change its total effect. |
| E2.5 Distortion grows faster than the ratio | **Failed** | Distortion is proportional to the number of fakes: 0.43, 0.40, and 0.41c per fake (changes -0.03 +- 0.05 and +0.01 +- 0.03c). The price is a weighted average of beliefs, and each heard fake shifts a belief by a fixed amount, so fakes add up rather than feed on each other. |

## What this means for the model

- The averaging belief rule dominates both experiments. The price stays near 50c, where the fee barely varies (1.68c per contract at 40c or 60c, 1.75c at 50c, against 1.12c at 20c), so the fee's peak at 50c has nothing to show itself against (E1.1). Rumors add linearly (E2.5), since price-following agents average toward a price that moves less than they do. This is the price-accuracy limit accepted in Step 3 (PLAN.md Section 2.2), now visible in the results; the report should give it as the reason E1.1 and E2.5 were not supported.
- Fees act on liquidity, not accuracy. Kalshi's fee cuts volume by more than half and makes 65% of steps trade-free, yet it narrows the spread and triples depth, because agents post free resting orders instead of paying to take. It slows news pricing by about 13 steps but leaves final accuracy unchanged.
- Network structure changes how fast news moves, not where the price ends up. Spreading and pricing are slowest on Watts-Strogatz, but distortion and final accuracy are the same in every network.
