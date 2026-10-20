# Results

Each prediction in `PREDICTIONS.md` against the data, for the Step 8b model (additive belief rule, cash and belief noise linked to informedness, recalibrated defaults; `PLAN.md` Step 8b). Every configuration ran on seeds 20000-20099 with the conservation invariants checked every step (no violations). Values are means over seeds with 95% intervals: `+-` is a normal interval on the mean or on a paired difference, `[low, high]` a bootstrap interval over seeds (paired across configurations). The per-run data are in `results/e1_rules.csv` and `results/e2_information.csv`. The JSON file beside each records the commit, seeds, and full configurations. `uv run python experiments/e1_rules.py summary` (or `e2_information.py summary`) reprints every number here from those files. A news clue's move is measured over the 300 steps after its release.

P&L "at P*" values each agent's closing position at the final `P*` instead of the outcome. Without disinformation this has the same expectation as realized P&L with far less outcome luck (`metrics.expected_pnl`), so it is reported for E1 and for E2 without fakes only.

## The calibrated default

On 96 fresh seeds (3000-3095) the frozen default meets every Step 7 pattern: median spread 1c, fill sizes within a factor of 2 of Kalshi's, 1.6% of steps with an empty book side, a calibration slope of 1.29 (the band is 0.8-1.4, `PLAN.md` Section 2.2), and a price Brier score 0.025 +- 0.023 better than the average agent's own judgment (better in 68% of seeds). Agents with no clues lose $14.9 per agent at `P*`; the best-informed tenth earn $17.6. That last margin is thin: the informed tenth's P&L varies by about $40 per agent between 100-seed sets, and on the E1 seeds it is negative on Kalshi (below).

Context for every result: the price now tracks `P*`. In the Kalshi market the final price sits 16.4c from 50c on average against 18.7c for `P*`, with a logit slope on `P*` of 0.68 (0.17-0.19 in the first model), and the mean gap `|price - P*|` is about 6c in every market (14c before). Final Brier scores are 0.187-0.195, against 0.185 for `P*` and 0.25 for a constant 50c.

## E1. Kalshi rules

| Metric | AMM | CLOB | Kalshi |
|---|---|---|---|
| Mean `|price - P*|` (c) | 6.64 +- 0.40 | 6.08 +- 0.62 | 6.03 +- 0.64 |
| Final Brier | 0.187 +- 0.025 | 0.195 +- 0.029 | 0.195 +- 0.030 |
| News move by lag 300 (c) | 1.43 +- 0.36 | 0.84 +- 0.17 | 0.73 +- 0.14 |
| News half-life (steps) | 14.3 [12.0, 21.4] | 23.4 [14.9, 27.9] | 26.4 [22.5, 44.9] |
| Volume (contracts) | 94,508 | 311,237 | 211,737 |
| No-trade steps | 2% | 24% | 41% |
| Median spread (c) | - | 2.97 | 1.21 |
| Depth within 2c (contracts) | - | 11,882 | 21,545 |
| Empty-book steps | - | 5.0% | 1.3% |
| LMSR or market-maker P&L ($/run) | +40.9 +- 3.3 | +217 +- 29 | +26 +- 9 |
| P&L at `P*`, no clues / top tenth ($/agent) | -1.9 / +18.6 | -5.4 / +30.2 | -9.3 / -14.3 |

The top tenth's P&L at `P*` has wide intervals (+-12, +-45, and +-42 $/agent); agents with no clues lose significantly in all three markets.

| Prediction | Verdict | Evidence |
|---|---|---|
| E1.1 The fee slows the pricing of news, most near 50c | **Not supported** | Kalshi's 300-step news move is 0.12 +- 0.17c smaller than the CLOB's and its half-life 3.0 [-2.5, 21.1] steps longer, neither significant. Near 50c the shortfall is larger (0.27 +- 0.40c) than outside 40-60c (0.07 +- 0.30c), the predicted direction, but the difference, 0.20 [-0.70, 0.29]c, is not significant. With prices now spread out, 5.3 of each run's news clues arrive near 50c and 11.7 outside, so the test has data on both sides. |
| E1.2 Fees cut volume much more than accuracy | **Held** | From the CLOB to Kalshi, volume falls 32.0% [29.5, 34.5] while mean aggregation error changes by -0.8% [-3.5, 2.0]. Final Brier scores do not differ (0.001 +- 0.005). |
| E1.3 Free maker orders draw agents into making | **Held** | Agents are the maker on 90.9% of Kalshi contracts vs 84.0% in the CLOB (+6.9 +- 0.6 points). The Kalshi book also has 40% of the CLOB's spread and 1.8 times its depth. |
| E1.4 Makers out-earn takers on Kalshi | **Held** | Valued at `P*`, agents' maker fills earn +0.29 +- 0.57c per contract and their taker fills -1.71 +- 0.51c after fees, a gap of 1.99 +- 1.08c. Realized at the outcome the gap is 5.93 +- 4.20c. As in Burgi et al. (2025), takers lose on average. |
| E1.5 The fee falls hardest on the uninformed | **Held** | Agents with no clues pay 1.27 +- 0.05c in fees per contract traded, the top tenth 0.26 +- 0.03c (difference 1.01 +- 0.06c). Patience and cash rise with informedness, which likely leaves the informed posting free resting orders while the noisier, less patient uninformed take and pay the fee. |
| E1.6 The AMM prices news fastest | **Held** | The AMM's half-life is 9.1 [0.6, 14.7] steps shorter than the CLOB's and its 300-step move 0.59 +- 0.37c larger (1.43 vs 0.84c). The LMSR still gains $40.9 +- 3.3 per run instead of losing its subsidy: uninformed agents' losses to it exceed what informed agents win. |

## E2. Information spread

| Network (ratio 0.1) | Steps to half reach | News half-life (steps) | Distortion (c) | Last quarter (c) | Realized P&L, top - bottom degree quarter ($) |
|---|---|---|---|---|---|
| Erdos-Renyi | 21.7 +- 0.3 | 26.9 [23.2, 37.5] | 0.72 +- 0.11 | 1.60 +- 0.28 | +30.5 +- 59.9 |
| Watts-Strogatz | 37.6 +- 0.5 | 47.3 [40.4, 55.6] | 0.73 +- 0.10 | 1.44 +- 0.23 | +50.8 +- 51.8 |
| Barabasi-Albert | 19.5 +- 0.4 | 23.4 [20.2, 28.5] | 0.69 +- 0.11 | 1.53 +- 0.28 | +60.8 +- 62.0 |
| Holme-Kim | 20.1 +- 0.4 | 30.7 [22.9, 42.4] | 0.74 +- 0.12 | 1.65 +- 0.26 | +60.1 +- 70.9 |
| Stochastic block model | 23.5 +- 0.3 | 31.2 [24.4, 40.6] | 0.79 +- 0.11 | 1.74 +- 0.24 | +24.7 +- 50.5 |

| Holme-Kim ratio | Fakes | Distortion (c) | Per fake (c) |
|---|---|---|---|
| 0.05 | 1 | 0.36 +- 0.07 | 0.36 +- 0.07 |
| 0.1 | 2 | 0.74 +- 0.12 | 0.37 +- 0.06 |
| 0.2 | 4 | 1.43 +- 0.16 | 0.36 +- 0.04 |

| Prediction | Verdict | Evidence |
|---|---|---|
| E2.1 Watts-Strogatz spreads news slowest | **Held** | 37.6 steps to half reach vs 19.5-23.5 elsewhere; paired gaps of 14.1 to 18.2 steps, each +- 0.6 or less. |
| E2.2 Slower spread means slower pricing | **Held** | Watts-Strogatz has the longest news half-life, 16.2 [6.6, 25.5] to 23.9 [16.2, 32.1] steps longer than each other network. The same holds without disinformation (47.7 steps vs 24.6-35.5). |
| E2.3 Well-connected agents hear news earlier and earn more | **Held, small** | The top degree quarter learns each news clue 11.2-11.5 steps before the bottom quarter (4.7 steps in Watts-Strogatz, whose degrees vary least). With fakes (ratio 0.1) it earns $45.4 +- 26.5 more in realized P&L across all five networks, no single network significant. Without fakes, valued at `P*`, the gap is $7.0 +- 6.6 across all five, and no single network is significant. News is a small part of what agents know, so hearing it early is worth little. |
| E2.4 Echo chambers sustain the largest distortion | **Failed** | The stochastic block model's distortion is 0.05 to 0.10c larger than each other network's (each +- 0.10c, none significant). In the last quarter it is 0.10 to 0.30c larger, significant only against Watts-Strogatz (0.30 +- 0.19c). |
| E2.5 Distortion grows faster than the ratio | **Failed** | Distortion is proportional to the number of fakes: 0.36, 0.37, and 0.36c per fake (changes 0.01 +- 0.05 and -0.01 +- 0.03c). Each heard fake shifts a belief by a fixed amount, so fakes add up rather than feed on each other. |

## The first model (Step 8)

The first run of E1 and E2 used the averaging belief rule, `(1 - w_market) * (L_own + w_hearsay * L_heard) + w_market * logit(mid)`, with cash unrelated to informedness and a 90-step news window. Its price stayed within about 5c of 50c (against 19c for `P*`), moved 0.17-0.19 logit units per unit of `logit(P*)`, and missed `P*` by 14c on average, so a 60c price meant about 88% YES. Under it E1.2, E1.3 (small), E2.1, and E2.2 held, E1.1 and E2.3 partly held, and E1.4, E1.5 (reversed), E1.6, E2.4, and E2.5 failed. Because those results could not speak to price accuracy, Step 8b changed the belief rule and linked cash and belief noise to informedness (`PLAN.md` Step 8b). Predictions were not changed. Against the first model, E1.4, E1.5, and E1.6 now hold, E1.3 and E2.3 hold more clearly, and E1.1 is no longer supported.

## What this means for the model

- Prices now aggregate information: they track `P*` about as closely in every market (6c mean gap, final Brier within 0.01 of `P*`), and the default is mildly underconfident (calibration slope 1.25-1.3), the direction of the favorite-longshot bias.
- Fees act on liquidity, not accuracy. Kalshi's fee cuts volume by a third and makes 41% of steps trade-free, yet it narrows the spread and nearly doubles depth, because agents post free resting orders instead of paying to take. Its effect on news pricing is not significant at 100 seeds.
- On Kalshi the fee and the spread fall on the uninformed, who take, while the informed make (E1.4, E1.5), as Burgi et al. (2025) find. Whether the informed then profit is fragile: the best-informed tenth come out about even, with seed-set means from -$14 to +$18 per agent, while the uninformed lose in every set.
- Network structure changes how fast news moves, not where the price ends up. Spreading and pricing are slowest on Watts-Strogatz, but distortion and final accuracy are about the same in every network, and fakes add linearly.
