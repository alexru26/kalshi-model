# Results

Each prediction in `PREDICTIONS.md` against the data, for the Step 8b model: the additive belief rule, cash and belief noise linked to informedness, relayed true news counted as own evidence (fakes stay hearsay), and recalibrated defaults (`PLAN.md` Step 8b). Every configuration ran on seeds 20000-20099 with the conservation invariants checked every step (no violations). Values are means over seeds with 95% intervals: `+-` is a normal interval on the mean or on a paired difference, `[low, high]` a bootstrap interval over seeds (paired across configurations). The per-run data are in `results/e1_rules.csv` and `results/e2_information.csv`. The JSON file beside each records the commit, seeds, and full configurations. `uv run python experiments/e1_rules.py summary` (or `e2_information.py summary`) reprints every number here from those files. A news clue's move is measured over the 300 steps after its release. `uv run python experiments/plot_run.py <seed>` charts a single run.

P&L "at P*" values each agent's closing position at the final `P*` instead of the outcome. Without disinformation this has the same expectation as realized P&L with far less outcome luck (`metrics.expected_pnl`), so it is reported for E1 and for E2 without fakes only.

## The calibrated default

`experiments/validate_defaults.py` checked the frozen default on four fresh seed sets (3000-3095, 5000-5143, 6000-6143, 7000-7095; 480 seeds). On every set the median spread is 1c, fill sizes are within a factor of 2 of Kalshi's, 0.3% of steps have an empty book side, the calibration slope is 1.10 to 1.18 (band 0.8-1.4, `PLAN.md` Section 2.2), and the price's Brier score beats the average agent's own judgment by 0.014 to 0.027. Agents with no clues lose $4.9 to $12.1 per agent at `P*` on every set. The best-informed tenth earn +$132, +$122, and +$51 per agent on three sets but lose $22.8 on 5000-5143, so the informed-earn pattern fails there; pooled over all 480 seeds they earn about +$66. Their P&L is concentrated in large positions and varies widely between seed sets.

Context for every result: in the Kalshi market the final price sits 17.0c from 50c on average against 18.7c for `P*`, with a logit slope on `P*` of 0.77, and the mean gap `|price - P*|` is 3.5c in both order books and 7.1c under the AMM. Final Brier scores are 0.192-0.196, against 0.185 for `P*` and 0.25 for a constant 50c. Regressing the final logit price on the two kinds of evidence over the 100 Kalshi runs gives 1.01 +- 0.07 per unit of news evidence and 0.69 +- 0.04 per unit of private evidence (`P*` weights both at 1): news is now priced in full, and the private clues, held by few agents, are the part the price misses.

## E1. Kalshi rules

| Metric | AMM | CLOB | Kalshi |
|---|---|---|---|
| Mean `|price - P*|` (c) | 7.14 +- 0.63 | 3.48 +- 0.44 | 3.52 +- 0.45 |
| Final Brier | 0.196 +- 0.028 | 0.192 +- 0.030 | 0.192 +- 0.030 |
| News move by lag 300 (c) | 3.18 +- 0.39 | 3.00 +- 0.27 | 2.94 +- 0.27 |
| News half-life (steps) | 19.6 [17.3, 22.4] | 32.7 [29.9, 35.3] | 52.9 [48.2, 58.5] |
| Volume (contracts) | 110,441 | 449,031 | 317,416 |
| No-trade steps | 2% | 45% | 64% |
| Median spread (c) | - | 1.17 | 1.00 |
| Depth within 2c (contracts) | - | 29,269 | 57,954 |
| Empty-book steps | - | 1.1% | 0.3% |
| LMSR or market-maker P&L ($/run) | +43.3 +- 3.2 | +34 +- 6 | +7 +- 2 |
| P&L at `P*`, no clues / top tenth ($/agent) | -6.2 / +64.5 | -3.5 / +122.6 | -5.7 / +55.1 |

A news clue moves `P*` by 2.43c on average, so every market now prices news in full or slightly beyond it. The top tenth's P&L at `P*` has wide intervals (+-16, +-78, and +-80 $/agent); on Kalshi it is not significantly above 0.

| Prediction | Verdict | Evidence |
|---|---|---|
| E1.1 The fee slows the pricing of news, most near 50c | **Partly held** | Kalshi's news half-life is 20.3 [16.0, 25.1] steps longer than the CLOB's (52.9 vs 32.7), but by lag 300 the move is the same (difference -0.06 +- 0.14c). The near-50c part is not supported: near 50c Kalshi's move is 0.36 +- 0.42c larger than the CLOB's, outside 40-60c 0.11 +- 0.41c smaller, a difference of 0.47 [-0.13, 1.08]c in the opposite direction to the prediction and not significant (5.3 news clues per run arrive near 50c, 11.7 outside). |
| E1.2 Fees cut volume much more than accuracy | **Held** | From the CLOB to Kalshi, volume falls 29.3% [26.3, 32.2] while mean aggregation error changes by +1.2% [-1.3, 3.6]. Final Brier scores do not differ (-0.001 +- 0.004). |
| E1.3 Free maker orders draw agents into making | **Held, tiny** | Agents are the maker on 99.8% of Kalshi contracts vs 99.6% in the CLOB (+0.2 +- 0.1 points). With a 10-contract market maker, agents supply nearly all liquidity in both books; the larger change is in the book itself: twice the depth and a spread at the 1c tick. |
| E1.4 Makers out-earn takers on Kalshi | **Held** | Valued at `P*`, agents' maker fills earn -0.17 +- 0.50c per contract and their taker fills -1.31 +- 0.50c after fees, a gap of 1.14 +- 0.99c. Realized at the outcome the gap is 2.95 +- 4.00c, not significant. As in Burgi et al. (2025), takers lose on average. |
| E1.5 The fee falls hardest on the uninformed | **Held** | Agents with no clues pay 0.79 +- 0.04c in fees per contract traded, the top tenth 0.60 +- 0.04c (difference 0.19 +- 0.06c). |
| E1.6 The AMM prices news fastest | **Held** | The AMM's half-life is 13.1 [10.2, 15.8] steps shorter than the CLOB's; by lag 300 the moves do not differ (0.17 +- 0.34c). Its prices are less accurate, though: the mean gap to `P*` is 3.66 +- 0.48c larger than the CLOB's, and the LMSR gains $43.3 +- 3.2 per run instead of losing its subsidy. |

## E2. Information spread

| Network (ratio 0.1) | Steps to half reach | News half-life (steps) | Distortion (c) | Last quarter (c) | Realized P&L, top - bottom degree quarter ($) |
|---|---|---|---|---|---|
| Erdos-Renyi | 24.5 +- 0.4 | 51.8 [47.4, 57.1] | 0.59 +- 0.07 | 1.19 +- 0.17 | +5.9 +- 99.1 |
| Watts-Strogatz | 45.3 +- 0.7 | 74.1 [70.2, 79.0] | 0.51 +- 0.08 | 1.06 +- 0.19 | +44.8 +- 76.9 |
| Barabasi-Albert | 21.6 +- 0.5 | 51.6 [46.8, 56.2] | 0.52 +- 0.06 | 0.98 +- 0.16 | +64.8 +- 121.6 |
| Holme-Kim | 22.7 +- 0.6 | 51.3 [46.9, 56.4] | 0.48 +- 0.07 | 0.95 +- 0.17 | +12.5 +- 119.9 |
| Stochastic block model | 26.7 +- 0.4 | 56.5 [52.3, 61.6] | 0.50 +- 0.07 | 1.06 +- 0.14 | +45.7 +- 101.9 |

| Holme-Kim ratio | Fakes | Distortion (c) | Per fake (c) |
|---|---|---|---|
| 0.05 | 1 | 0.24 +- 0.04 | 0.24 +- 0.04 |
| 0.1 | 2 | 0.48 +- 0.07 | 0.24 +- 0.03 |
| 0.2 | 4 | 1.02 +- 0.11 | 0.25 +- 0.03 |

| Prediction | Verdict | Evidence |
|---|---|---|
| E2.1 Watts-Strogatz spreads news slowest | **Held** | 45.3 steps to half reach vs 21.6-26.7 elsewhere; paired gaps of 18.7 to 23.7 steps, each +- 0.8 or less. |
| E2.2 Slower spread means slower pricing | **Held** | Watts-Strogatz has the longest news half-life, 17.5 [14.2, 21.1] to 22.8 [19.2, 26.8] steps longer than each other network. The same holds without disinformation (75.4 steps vs 50.3-57.5). |
| E2.3 Well-connected agents hear news earlier and earn more | **Held, small** | The top degree quarter learns each news clue 16.0-17.0 steps before the bottom quarter (6.4 steps in Watts-Strogatz, whose degrees vary least). Without fakes, valued at `P*`, it earns $13.5 +- 11.3 more across all five networks, no single network significant. With fakes (ratio 0.1) the realized gap is $34.7 +- 46.9, not significant. |
| E2.4 Echo chambers sustain the largest distortion | **Failed** | The stochastic block model's distortion differs from each other network's by -0.09 to +0.02c (each +- 0.08c, none significant), and by -0.13 to +0.11c in the last quarter (none significant). |
| E2.5 Distortion grows faster than the ratio | **Failed** | Distortion is proportional to the number of fakes: 0.24, 0.24, and 0.25c per fake (changes -0.00 +- 0.04 and 0.02 +- 0.03c). Each heard fake shifts a belief by a fixed amount, so fakes add up rather than feed on each other. |

## Earlier versions of the model

**First model (Step 8).** The first run of E1 and E2 used the averaging belief rule, `(1 - w_market) * (L_own + w_hearsay * L_heard) + w_market * logit(mid)`, with cash unrelated to informedness and a 90-step news window. Its price stayed within about 5c of 50c (against 19c for `P*`), moved 0.17-0.19 logit units per unit of `logit(P*)`, and missed `P*` by 14c on average, so a 60c price meant about 88% YES. Under it E1.2, E1.3 (small), E2.1, and E2.2 held, E1.1 and E2.3 partly held, and E1.4, E1.5 (reversed), E1.6, E2.4, and E2.5 failed. Because those results could not speak to price accuracy, Step 8b changed the belief rule and linked cash and belief noise to informedness.

**News as hearsay (Step 8b, first calibration).** With those changes but relayed news still counted as hearsay, the Kalshi price missed `P*` by 6.0c on average and moved 0.73c per news clue against a 2.43c `P*` jump: over the 100 runs it weighted news at 0.21 of its value and private clues at 0.84, because the cash-rich informed discount hearsay. Its verdicts matched the current ones except that E1.1 was not supported (no significant fee effect on news pricing) and E1.3 was larger (+6.9 points). Its informed tenth earned -$51 to +$18 per agent across fresh seed sets. `experiments/plot_run.py` on seed 20001 showed the price flat at 28c while `P*` rose to 38c; it now ends at 38.5c against 38.2c.

Predictions were never changed. Against the first model, E1.4, E1.5, and E1.6 now hold, E1.3 and E2.3 hold more clearly, and E1.1 is still only partly held.

## What this means for the model

- Prices now aggregate information: they price news in full, track `P*` within 3.5c in the order books, and are mildly underconfident (calibration slope 1.10-1.18), the direction of the favorite-longshot bias. What they miss is mostly private information held by few agents.
- Fees act on liquidity and speed, not accuracy. Kalshi's fee cuts volume by 29% and makes 64% of steps trade-free, slows news pricing (half-life about 20 steps longer), and leaves final accuracy unchanged, while the book doubles its depth because agents post free resting orders instead of paying to take.
- On Kalshi takers lose to makers and the uninformed pay more fees per contract (E1.4, E1.5), as Burgi et al. (2025) find. The informed tenth earn on average, but by amounts that vary widely between seed sets, while the uninformed lose in every set.
- Network structure changes how fast news moves, not where the price ends up. Spreading and pricing are slowest on Watts-Strogatz, distortion and final accuracy are about the same in every network, and fakes add linearly.
