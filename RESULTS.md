# Results

Each prediction in `PREDICTIONS.md` against the data, for the Step 8b model: the additive belief rule, cash and belief noise linked to informedness, relayed true news counted as own evidence (fakes stay hearsay), and recalibrated defaults (`PLAN.md` Step 8b). Every configuration ran on seeds 20000-20099 with the conservation invariants checked every step (no violations). Values are means over seeds with 95% intervals: `+-` is a normal interval on the mean or on a paired difference, `[low, high]` a bootstrap interval over seeds (paired across configurations). The per-run data are in `results/e1_rules.csv`, `results/e2_information.csv`, and `results/e3_manipulation.csv`. The JSON file beside each records the commit, seeds, and full configurations. `uv run python experiments/e1_rules.py summary` (or `e2_information.py summary`, `e3_manipulation.py summary`) reprints every number here from those files. A news clue's move is measured over the 300 steps after its release. `uv run python experiments/plot_run.py <seed>` charts a single run.

P&L "at P*" values each agent's closing position at the final `P*` instead of the outcome. Without disinformation this has the same expectation as realized P&L with far less outcome luck (`metrics.expected_pnl`), so it is reported for E1 and for E2 without fakes only.

## The calibrated default

`experiments/validate_defaults.py` checked the frozen default on four fresh seed sets (3000-3095, 5000-5143, 6000-6143, 7000-7095; 480 seeds). On every set the median spread is 1c, fill sizes are within a factor of 2 of Kalshi's, 0.3% of steps have an empty book side, the calibration slope is 1.10 to 1.18 (band 0.8-1.4, `PLAN.md` Section 2.2), and the price's Brier score beats the average agent's own judgment by 0.014 to 0.027. Agents with no clues lose $4.9 to $12.1 per agent at `P*` on every set. The best-informed tenth earn +$132, +$122, and +$51 per agent on three sets but lose $22.8 on 5000-5143, so the informed-earn pattern fails there; pooled over all 480 seeds they earn about +$66. Their P&L is concentrated in large positions and varies widely between seed sets.

Context for every result: in the Kalshi market the final price sits 17.0c from 50c on average against 18.7c for `P*`, with a logit slope on `P*` of 0.77, and the mean gap `|price - P*|` is 3.5c in both order books. Final Brier scores are 0.192, against 0.185 for `P*` and 0.25 for a constant 50c. Regressing the final logit price on the two kinds of evidence over the 100 Kalshi runs gives 1.01 +- 0.07 per unit of news evidence and 0.69 +- 0.04 per unit of private evidence (`P*` weights both at 1): news is now priced in full, and the private clues, held by few agents, are the part the price misses.

## E1. Kalshi rules

E1 compares the fee-free CLOB with Kalshi. An AMM (LMSR) configuration was run at first and later dropped from the experiment; `results/e1_rules.csv` still holds its rows, which the summary ignores.

| Metric | CLOB | Kalshi |
|---|---|---|
| Mean `|price - P*|` (c) | 3.48 +- 0.44 | 3.52 +- 0.45 |
| Final Brier | 0.192 +- 0.030 | 0.192 +- 0.030 |
| News move by lag 300 (c) | 3.00 +- 0.27 | 2.94 +- 0.27 |
| News half-life (steps) | 32.7 [29.9, 35.3] | 52.9 [48.2, 58.5] |
| Volume (contracts) | 449,031 | 317,416 |
| No-trade steps | 45% | 64% |
| Median spread (c) | 1.17 | 1.00 |
| Depth within 2c (contracts) | 29,269 | 57,954 |
| Empty-book steps | 1.1% | 0.3% |
| Market-maker P&L ($/run) | +34 +- 6 | +7 +- 2 |
| P&L at `P*`, no clues / top tenth ($/agent) | -3.5 / +122.6 | -5.7 / +55.1 |

A news clue moves `P*` by 2.43c on average, so both markets price news in full or slightly beyond it. The top tenth's P&L at `P*` has wide intervals (+-78 and +-80 $/agent); on Kalshi it is not significantly above 0.

| Prediction | Verdict | Evidence |
|---|---|---|
| E1.1 The fee slows the pricing of news, most near 50c | **Partly held** | Kalshi's news half-life is 20.3 [16.0, 25.1] steps longer than the CLOB's (52.9 vs 32.7), but by lag 300 the move is the same (difference -0.06 +- 0.14c). The near-50c part is not supported: near 50c Kalshi's move is 0.36 +- 0.42c larger than the CLOB's, outside 40-60c 0.11 +- 0.41c smaller, a difference of 0.47 [-0.13, 1.08]c in the opposite direction to the prediction and not significant (5.3 news clues per run arrive near 50c, 11.7 outside). |
| E1.2 Fees cut volume much more than accuracy | **Held** | From the CLOB to Kalshi, volume falls 29.3% [26.3, 32.2] while mean aggregation error changes by +1.2% [-1.3, 3.6]. Final Brier scores do not differ (-0.001 +- 0.004). |
| E1.3 Free maker orders draw agents into making | **Held, tiny** | Agents are the maker on 99.8% of Kalshi contracts vs 99.6% in the CLOB (+0.2 +- 0.1 points). With a 10-contract market maker, agents supply nearly all liquidity in both books; the larger change is in the book itself: twice the depth and a spread at the 1c tick. |
| E1.4 Makers out-earn takers on Kalshi | **Held** | Valued at `P*`, agents' maker fills earn -0.17 +- 0.50c per contract and their taker fills -1.31 +- 0.50c after fees, a gap of 1.14 +- 0.99c. Realized at the outcome the gap is 2.95 +- 4.00c, not significant. As in Burgi et al. (2025), takers lose on average. |
| E1.5 The fee falls hardest on the uninformed | **Held** | Agents with no clues pay 0.79 +- 0.04c in fees per contract traded, the top tenth 0.60 +- 0.04c (difference 0.19 +- 0.06c). |

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

## E3. Manipulation

All three configurations add the same manipulator to the Kalshi default (`PLAN.md` Section 2.3): 5% of the agents' cash, median traits at 2 private clues, a turn every step, and a layer from step 1000. Each seed ran the three together. Distortion is the price minus the twin's price on the same seed, in cents; P&L is valued at final `P*`.

| Metric | Twin | Pump and dump | Spoofing |
|---|---|---|---|
| Manipulator P&L at `P*` ($/run) | -380 +- 525 | -3,862 +- 1,312 | -387 +- 542 |
| Premium over the twin at `P*` ($) | - | -3,482 +- 1,129 | -8 +- 164 |
| Premium, realized at the outcome ($) | - | -5,705 +- 5,043 | -513 +- 801 |
| Distortion while the layer acts (c) | - | 1.76 +- 0.18 | 0.56 +- 0.15 |
| Distortion, 300 steps after (c) | - | 0.32 +- 0.23 | -0.07 +- 0.23 |
| Peak of the mean distortion curve (c) | - | 2.42 [2.19, 2.77], 20 steps in | 0.87 [0.69, 1.19], 75 steps in |
| Steps until it falls below half the peak | - | 126 [99, 148] | 450 [120, 554] |
| Steps the layer acted | - | 171 | 500 |
| Contracts bought / sold by the layer | - | 55,635 / 14,022 | 3,195 / 9,453 |
| Taken from agents ($/run) | - | -2,448 +- 1,057 | 43 +- 210 |
| Change in fees ($/run) | - | +1,033 +- 197 | +51 +- 89 |
| Volume (contracts) | 328,677 | 400,511 | 331,945 |

The market maker's P&L changes by under $3 in either case. The pump reached its 5c target in 78% of runs, but sold back only a quarter of what it bought: the price soon fell to its start, below which the dump does not sell, and it held the rest. The twin's presence alone barely changes the Kalshi default (paired against E1 on the same seeds): mean `|price - P*|` rises by 0.11 +- 0.10c, volume by 11,261 +- 7,983 contracts (3%), and the best-informed tenth earn $24.6 +- 18.8 more per agent at `P*`.

| Prediction | Verdict | Evidence |
|---|---|---|
| E3.1 Pump and dump moves the price but loses money | **Held** | While it acts the price sits 1.76 +- 0.18c above the twin's (peak 2.42c), and the pump loses $3,482 +- 1,129 more than its twin. Agents gain $2,448 +- 1,057 from it and the fee pot $1,033 +- 197. The best-informed tenth sold it 55% of its contracts against 37% of all trading by chance (per-seed gap +0.14 +- 0.05). |
| E3.2 Spoofing profits, to the extent agents read book depth | **Partly held** | The spoofed bid lifts the price 0.56 +- 0.15c while it rests (peak 0.87c), and the agents who buy the spoofer's offers read depth more than agent buyers in general (contract-weighted `w_depth` 0.175 vs 0.139, gap 0.036 +- 0.011). But the spoofer earns no premium (-$8 +- 164). The lift, about half a cent, is below the 1c tick and the minimum edge of most agents; the spoofer sold only 9,453 contracts per run, and its fake bid filled in 44% of runs (3,195 +- 2,561 contracts). |
| E3.3 More liquid runs distort less, and for less time | **Failed** | Across seeds, depth just before the start is not correlated with distortion while acting (Spearman 0.09 [-0.12, 0.26] pump, -0.07 [-0.25, 0.13] spoof) or after (0.07 [-0.11, 0.27], -0.08 [-0.27, 0.11]). With depth averaged over the 100 steps before, the pump's correlations lean the other way (0.19 [-0.01, 0.37] while acting) and its premium falls with depth (-0.20 [-0.40, -0.00]): in deeper books the pump buys more and loses more rather than moving the price less. |

## Earlier versions of the model

**First model (Step 8).** The first run of E1 and E2 used the averaging belief rule, `(1 - w_market) * (L_own + w_hearsay * L_heard) + w_market * logit(mid)`, with cash unrelated to informedness and a 90-step news window. Its price stayed within about 5c of 50c (against 19c for `P*`), moved 0.17-0.19 logit units per unit of `logit(P*)`, and missed `P*` by 14c on average, so a 60c price meant about 88% YES. Under it E1.2, E1.3 (small), E2.1, and E2.2 held, E1.1 and E2.3 partly held, and E1.4, E1.5 (reversed), E1.6, E2.4, and E2.5 failed. Because those results could not speak to price accuracy, Step 8b changed the belief rule and linked cash and belief noise to informedness.

**News as hearsay (Step 8b, first calibration).** With those changes but relayed news still counted as hearsay, the Kalshi price missed `P*` by 6.0c on average and moved 0.73c per news clue against a 2.43c `P*` jump: over the 100 runs it weighted news at 0.21 of its value and private clues at 0.84, because the cash-rich informed discount hearsay. Its verdicts matched the current ones except that E1.1 was not supported (no significant fee effect on news pricing) and E1.3 was larger (+6.9 points). Its informed tenth earned -$51 to +$18 per agent across fresh seed sets. `experiments/plot_run.py` on seed 20001 showed the price flat at 28c while `P*` rose to 38c; it now ends at 38.5c against 38.2c.

Predictions were never changed. Against the first model, E1.4, E1.5, and E1.6 now hold, E1.3 and E2.3 hold more clearly, and E1.1 is still only partly held.

## What this means for the model

- Prices now aggregate information: they price news in full, track `P*` within 3.5c in the order books, and are mildly underconfident (calibration slope 1.10-1.18), the direction of the favorite-longshot bias. What they miss is mostly private information held by few agents.
- Fees act on liquidity and speed, not accuracy. Kalshi's fee cuts volume by 29% and makes 64% of steps trade-free, slows news pricing (half-life about 20 steps longer), and leaves final accuracy unchanged, while the book doubles its depth because agents post free resting orders instead of paying to take.
- On Kalshi takers lose to makers and the uninformed pay more fees per contract (E1.4, E1.5), as Burgi et al. (2025) find. The informed tenth earn on average, but by amounts that vary widely between seed sets, while the uninformed lose in every set.
- Manipulation does not pay here. A pump moves the price about 2c for about a hundred steps and loses money to the informed, who sell to it; a spoof lifts the price about half a cent and earns nothing, since agents read depth too little to move past the tick. Neither effect depends on how deep the book was, so the liquidity result of Rasooly & Rozzi (2025) is not reproduced. The spoofing result is conditional on `w_depth`, which has no data anchor.
- Network structure changes how fast news moves, not where the price ends up. Spreading and pricing are slowest on Watts-Strogatz, distortion and final accuracy are about the same in every network, and fakes add linearly.
