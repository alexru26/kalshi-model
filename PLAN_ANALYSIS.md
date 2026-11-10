# Analysis Plan

How the model is analyzed, what each analysis looks for, and the figures, tables, equations, and movie the final report and presentation are built on. Numbers marked "now" come from the current 500-seed runs (`results/*.csv`); everything else is a best guess, to be replaced by data.

**Core question.** How do Kalshi's fee, disinformation on the social network, and a manipulator change how fast and how accurately the price finds the truth?

**Yardstick.** Every run has an exact benchmark, `P*`: the Bayesian probability of YES given every real clue some agent holds. A perfect market would trade at `P*`, so every accuracy metric is a gap to it.

## 1. Data

| Source | What it holds | Used for |
|---|---|---|
| Step series (Mesa `DataCollector`, one row per step) | price, `P*`, volume, best bid and ask, spread, depth, market maker inventory and spread; in E3 the manipulator's orders and P&L | single-run charts, the movie, news-response and distortion curves |
| Run rows (`results/<experiment>.csv`, one row per run, `metrics.run_metrics`) | about 75 metrics per run (125 in E3): error, Brier, news response by lag and half-life, volume, liquidity, fees, P&L by informedness, network position, and maker/taker role, distortion, manipulation premium | every experiment comparison |
| Run metadata (`results/<experiment>.json`) | git commit, seeds, full configurations | reproducibility |
| Kalshi public API (`data/fetch_kalshi_patterns.py`) | resolved binary markets: spreads, trade sizes, price a fixed time before close, outcome | calibration targets and the calibration-curve check |

**Design.** Each experiment changes one variable around one default (Kalshi rules, Holme-Kim network, no fakes, no manipulator). Every configuration runs on the same **500 seeds** (20000-20499), with the conservation invariants checked every step. A seed fixes the world, the population, and the turn order, so configurations are compared **seed by seed** (paired), which removes most between-run noise. 500 seeds (up from 100) cut every interval to under half its width, so the effects that were not significant at 100 seeds got a fair test, and the violins show smooth distributions. The spoofer's premium is now clearly negative (-$180 +- 60, -$112 +- 122 before); makers vs takers and the P&L of well-connected agents stay null, so any real effect there is small. A full rerun of all three experiments takes about 40 minutes on 16 cores.

**Statistics.** Each run is one data point. Means get a normal 95% interval; differences between configurations are paired differences on shared seeds with 95% intervals. Statistics of a mean curve (news half-life of the mean response, peak distortion) get a bootstrap interval that resamples seeds, the same seeds for every configuration. Cross-seed associations use Spearman correlation.

## 2. What we look for

### 2.1 Key behaviors (stylized facts)

The default market must reproduce these before the experiments mean anything.

| Behavior | Real-world fact | Model metric | Expectation |
|---|---|---|---|
| Information aggregation | prices pool scattered information (Hayek; Plott & Sunder) | `|price - P*|`; price Brier vs the average agent's | mean gap 5.8c on Kalshi (now); beats the average agent's own judgment |
| Calibration and the favorite-longshot bias | a 70c contract wins about 70%; longshots win less than their price (Burgi et al. 2025 on Kalshi) | calibration curve; logit slope of `P*` on price | slope 1.3 on fresh seeds (`validate_defaults.py`): the price is too close to 50c, which is the favorite-longshot pattern |
| Tight, rarely empty book | median spread 2c in liquid Kalshi markets | median spread, empty-book steps | 1c spread, empty book under 1% of steps |
| Heavy-tailed trade sizes | Kalshi fill sizes | fill-size quantiles | match Kalshi's 25th-90th percentiles |
| Informed earn, uninformed lose | Grossman & Stiglitz 1980 | P&L at `P*` by informedness tier | best-informed tenth +$90 per agent, no-clue agents -$15 (now, Kalshi) |
| Makers out-earn takers | Burgi et al. 2025 | return per contract, maker vs taker fills | makers ahead by 0.19 +- 0.33c (now): right sign, not significant |
| News is priced in quickly | | news response curve and half-life | half-life 26 (CLOB) and 43 (Kalshi) steps (now); the move overshoots the 2.4c `P*` jump, since the underconfident price catches up |
| Manipulation is brief and does not pay | Hanson & Oprea 2009; Rasooly & Rozzi 2025 | distortion curve, premium | pump about 3c for about 100 steps, spoof under 1c; both premiums negative (now) |

### 2.2 Experiments

Each experiment is reduced to the few metrics that answer its question: **speed** (news half-life), **accuracy** (gap to `P*`), **liquidity** (volume), and **who pays or profits**. Metrics that describe side issues stay in the printed tables but not in the figures.

- **E1 Kalshi rules (CLOB with no fees vs Kalshi).** Does the fee change the speed and accuracy of the price, and at what cost in trading? Figure metrics: volume, news half-life, `|price - P*|`, and the best-informed tenth's P&L (the reward for gathering information, which the fee taxes). Dropped from the figures: fee per contract by informedness (who pays the fee, not what the fee does to the market) and maker vs taker returns (a stylized-fact check, reported in Table 2). Now: volume -30%, half-life +17 steps (26 to 43), error +0.35c (+6.5%), informed tenth's P&L -$9 per agent; the fee slows the price and thins trading far more than it hurts accuracy.
- **E2 Information spread (disinformation ratio 0 to 0.5, 0 to 10 fakes).** How far do fakes pull the price from `P*`, and does it compound? Figure metrics: distortion (signed gap toward the fakes minus the same seed's gap without them) over the whole run and the last quarter, against the number of fakes; when well- and poorly-connected agents learn news, and what they earn. Now: distortion is linear in fakes, 0.65c per fake over the run and 1.14c in the last quarter (it builds up as fakes accumulate); real news spreads and is priced as fast as without fakes.
- **E3 Manipulation (honest twin vs pump and dump vs spoofing).** Does a manipulator move the price, for how long, and does it pay? Figure metrics: distortion (price minus the twin's on the same seed) over time and while acting; the premium (manipulator's P&L minus the twin's, at final `P*`); agents' gain; distortion against book depth. Now: the pump peaks at 3.2c within 10 steps and falls below half that within about 100; the spoof lifts the price 0.6c; premiums are -$1,750 (pump) and -$180 (spoof), and agents gain $1,200 and $120.

### 2.3 How the behaviors change with the parameters

The experiments vary the three variables of the question. A sensitivity analysis asks how the key behaviors depend on everything else, in three stages on the Kalshi default:

1. **Screening (Morris elementary effects).** About 12 uncertain parameters (the number of agents `n_agents`, `market_trust`, `noise`, `min_edge`, `patience`, `kelly_frac`, `g`, `w_hearsay`, `w_depth`, spreading chance `beta`, clue accuracy `q`, market maker size), about 130 design points x 20 seeds. Outputs: `|price - P*|`, news half-life, volume, calibration slope, informed P&L. This ranks which parameters matter (mean absolute effect `mu*`) and which act through interactions (spread of effects `sigma`).
2. **One-at-a-time sweeps** for the number of agents and the top three parameters from screening, 4-5 levels x 100 seeds, paired on seeds. For `n_agents` (100, 300, 1000, 3000) the turns per step scale with it, so each agent trades as often.
3. **Robustness of the experiment results.** Rerun each experiment's headline paired difference on about 10 parameter sets that also passed the calibration, and report how often its sign holds.

Best guesses: more agents mean more volume and a smaller `|price - P*|`, flattening past about 1,000 agents, since the private clue pool is fixed and is already almost fully held at 300. `market_trust` and `g` set the calibration slope: higher trust makes the price follow itself more and pushes it from 50c, and `g = 1` (no probability weighting) should cut the favorite-longshot bias by about half; whatever remains comes from the market rules (Snowberg & Wolfers 2010). `noise` and `min_edge` set volume: more noise means more trades and more profit for the informed. `w_hearsay` scales the disinformation distortion about in proportion, and `w_depth` is what spoofing needs to pay: the spoof premium should turn positive only at large `w_depth`.

## 3. Key outputs

Figures F4-F10 exist now (`results/figures/`, drawn by `experiments/figures.py`, with 500-seed data); the rest are planned, with best-guess shapes. Every figure has a title and no subtitle; per-run metrics are violins (the spread over seeds) with a dot at the mean and a bar for its 95% interval.

### Figures

| # | Figure | Axes and marks | Expected look |
|---|---|---|---|
| F1 | **Model Overview** (diagram) | boxes for world and clues, network, 300 agents, order book, market maker, manipulator; arrows for clues, orders, and fills | the architecture in the README, drawn |
| F2 | **One Run** (`plot_run.py`) | x: step (0-2,000); y: price and `P*` (c), news and fake releases marked; panels below for volume and spread | the price follows `P*` in steps after each news clue, lagging by tens of steps and staying closer to 50c |
| F3 | **Calibration Curve** | x: price a fixed time before close, in 10c bins (c); y: share resolving YES (%); diagonal; three series: Kalshi data, model default, model with `g = 1`; 95% bars | both the data and the default bend below the diagonal under 30c and above it over 70c; `g = 1` sits closer to the diagonal |
| F4 | **Speed of News Pricing** | x: steps after a news clue (0-300); y: price move toward the clue (c); mean with 95% band, CLOB and Kalshi; dashed line at the mean `P*` move | both rise and level off near 3.4c, above the 2.4c `P*` line; Kalshi rises more slowly (half-life 43 vs 26 steps) |
| F5 | **Impact of Kalshi Fees** | 2x2 violins, CLOB vs Kalshi: volume (thousand contracts), news half-life (steps), `|price - P*|` (c), best-informed tenth's P&L ($ per agent); paired difference in each corner | Kalshi's volume violin clearly lower (-50k contracts) and half-life higher (+19.5 steps per run); error and informed P&L nearly overlap, with small but significant paired shifts (+0.35c, -$9) |
| F6 | **Distortion from Disinformation** | x: fake clues (1, 2, 4, 10); y: distortion toward the fakes (c); violins per count, means, dashed line through the origin labeled "c per fake"; panels for the whole run and the last quarter | means on a straight line, about 0.65c per fake (whole run) and 1.2c (last quarter); violins widen with more fakes |
| F7 | **Network Position and News** | violins, best- vs least-connected quarter: steps to learn a news clue; P&L ($ per agent, no fakes); paired difference | the best-connected learn 7.5 steps sooner (15.3 vs 22.8); P&L violins overlap, difference +$0.8 +- 2.2 |
| F8 | **Price Distortion from Manipulation** | x: steps after the manipulation starts (0-1,000, log past 10); y: price minus the honest twin's (c); mean with 95% band per strategy; marker where spoofing stops | pump jumps to 3.2c within 10 steps and falls back by about 100; spoof stays under 1c |
| F9 | **Does Manipulation Pay?** | violins, pump vs spoof: distortion while manipulating (c), manipulator's premium ($), agents' gain ($); zero line | pump: premium -$1,750 with a long loss tail, agents gain $1,200; spoof: -$180 and $120 |
| F10 | **Book Depth and Distortion** | scatter per seed: x: depth before manipulation (thousand contracts); y: distortion while manipulating (c); Spearman in the panel title | no protection from depth against the pump (Spearman +0.15); a weak one against spoofing (-0.15) |
| F11 | **Parameter Sensitivity** | Morris plot per output: x: mean absolute effect `mu*`, y: spread `sigma`, one labeled dot per parameter; plus line charts for the sweeps: x: parameter value, y: metric with 95% band | `market_trust`, `g`, and `noise` far right; the rest near zero; error falls and flattens with `n_agents` |
| F12 | **Robustness Across Calibrations** | one row per headline effect; x: paired difference, one dot per accepted calibration, zero line | most effects keep their sign on every set; makers vs takers straddles zero |

### Tables

| # | Table | Columns |
|---|---|---|
| T1 | Calibration patterns | pattern, Kalshi target, model value (default), pass |
| T2 | E1 results | metric (error, Brier, half-life, volume, no-trade share, spread, depth, empty-book share, fees, maker vs taker return, P&L by tier); CLOB; Kalshi; Kalshi - CLOB (paired, 95% CI) |
| T3 | E2 results | disinformation ratio; fakes; steps for news to reach half the agents; news half-life; distortion; last-quarter distortion; `|price - P*|`; Brier |
| T4 | E3 results | metric (premium, distortion while acting and after, peak and time to half peak, money taken from agents and market maker, fees, volume); twin; pump; spoof |
| T5 | Predictions and verdicts | prediction, metric, result with 95% interval, verdict (held, partly, failed) |
| T6 | Parameters | name, meaning, default, source (calibrated, literature, assumption) |

### Equations

- Kalshi taker fee: `fee = ceil(0.07 x C x P x (1 - P))` dollars for `C` contracts at price `P`.
- Benchmark: `logit(P*) = (# YES - # NO real clues held by any agent) x ln(q / (1 - q))`.
- Belief: `logit(b) = L_own + w_hearsay L_heard + w_market logit(price) + w_depth imbalance + noise z`, with `w_market = market_trust / (market_trust + n_own + w_hearsay n_heard)`.
- Sizing: fractional Kelly, `stake = kelly_frac x (p - price) / (1 - price) x cash`, on perceived `logit(p) = g logit(b)`.
- Brier split (no fakes): `E[(price - Y)^2] = E[(P* - Y)^2] + E[(price - P*)^2]`: unavoidable error plus aggregation loss.
- Distortion: `D = mean over steps of s x (price - P*) - same seed without the treatment`, with `s` the treatment's direction.

### Movie

A 60-90 second screen recording of the live GUI (`uv run solara run kalshi_model.gui`) on one seed: the price chasing `P*` while the network lights up as a news clue spreads from its seed agents, then a fake spreading and dragging the price away, then (E3 preset) a pump and the price falling back. It shows the mechanism behind F4, F6, and F8 in one clip for the presentation.
