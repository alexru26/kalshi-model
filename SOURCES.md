# Sources

Each entry has a one-sentence summary of why it matters for this project. Entries marked † were added from memory during design discussions and still need their details checked before the final report.

## Information aggregation foundations

- Hayek, F. A. (1945). The Use of Knowledge in Society. *American Economic Review*, 35(4). Prices summarize knowledge no single person has; the starting idea behind every prediction market.
- Plott, C., & Sunder, S. (1988). Rational expectations and the aggregation of diverse information in laboratory security markets. *Econometrica*, 56(5). Lab markets can aggregate private information, but not always, so failures are worth finding.
- Wolfers, J., & Zitzewitz, E. (2004). Prediction Markets. *Journal of Economic Perspectives*, 18(2). Readable overview of prediction markets for the introduction.
- Wolfers, J., & Zitzewitz, E. (2006). Interpreting prediction market prices as probabilities. NBER WP 12200. Under log utility the price equals the wealth-weighted mean belief, the benchmark our Kelly sizing ties to.
- Manski, C. (2006). Interpreting the predictions of prediction markets. *Economics Letters*, 91(3). Price equals average belief only under strong assumptions.
- Berg, J., Forsythe, R., Nelson, F., & Rietz, T. (2008). Results from a dozen years of election futures markets research. *Handbook of Experimental Economics Results*. Iowa Electronic Markets often beat polls; the classical success story.
- Arrow, K. J., et al. (2008). The promise of prediction markets. *Science*, 320(5878). Short case for prediction markets as forecasting tools.
- Page, L., & Clemen, R. (2013). Do prediction markets produce well-calibrated probability forecasts? *Economic Journal*, 123(568). Markets are well calibrated near resolution, less so far from it.
- Blume, L., & Easley, D. (2006). If you're so smart, why aren't you rich? Belief selection in complete and incomplete markets. *Econometrica*, 74(4). † Markets shift wealth toward traders with more accurate beliefs; justifies linking cash to informedness.
- Kets, W., Pennock, D. M., Sethi, R., & Shah, N. (2014). Betting strategies, market selection, and the wisdom of crowds. *AAAI Conference on Artificial Intelligence*. † Kelly bettors with better beliefs gain wealth share over repeated markets, which improves the price; supports the cash link.
- Grossman, S., & Stiglitz, J. (1980). On the impossibility of informationally efficient markets. *American Economic Review*, 70(3). † Traders learn from prices, and someone must lose to the informed or gathering information does not pay.
- Milgrom, P., & Stokey, N. (1982). Information, trade and common knowledge. *Journal of Economic Theory*, 26(1). † Rational, informed traders alone will not trade with each other, so uninformed agents are essential.

## Market mechanisms

- Hanson, R. (2003). Combinatorial information market design. *Information Systems Frontiers*, 5(1). Introduces market scoring rules, the classical automated market maker.
- Hanson, R. (2007). Logarithmic market scoring rules for modular combinatorial information aggregation. *Journal of Prediction Markets*, 1(1). The LMSR used as our AMM.
- Chen, Y., & Pennock, D. (2007). A utility framework for bounded-loss market makers. *UAI*. Variations on automated market makers.
- Othman, A., Pennock, D., Reeves, D., & Sandholm, T. (2013). A practical liquidity-sensitive automated market maker. *ACM TEAC*. Liquidity-sensitive alternative to the LMSR.
- Kyle, A. S. (1985). Continuous auctions and insider trading. *Econometrica*, 53(6). Classic model of informed traders hiding among noise traders by trading gradually; supports patient informed agents.
- Glosten, L., & Milgrom, P. (1985). Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. *Journal of Financial Economics*, 14(1). Market makers widen spreads to cover losses to informed traders; basis for our market-maker component.
- Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance*, 8(3). † Market makers continuously requote around a reservation price that shifts with inventory.
- Copeland, T., & Galai, D. (1983). Information effects on the bid-ask spread. *Journal of Finance*, 38(5). † A resting limit order is a free option that others can pick off after news.
- Foucault, T. (1999). Order flow composition and trading costs in a dynamic limit order market. *Journal of Financial Markets*, 2(2). † Picking-off risk grows with volatility and widens spreads.
- Foucault, T., Roell, A., & Sandas, P. (2003). Market making with costly monitoring. *Review of Financial Studies*, 16(2). † Traders who monitor less often are picked off more; justifies cancelling only on an agent's own turn.
- Hasbrouck, J., & Saar, G. (2009). Technology and liquidity provision: The blurring of traditional definitions. *Journal of Financial Markets*, 12(2). † Documents "fleeting" limit orders cancelled within seconds.
- Liu, W. M. (2009). Monitoring and limit order submission risks. *Journal of Financial Markets*, 12(1). Cancellations and revisions manage picking-off and non-execution risk, and rise with submission risk.
- Fong, K., & Liu, W. M. (2010). Limit order revisions. *Journal of Banking and Finance*, 34(8). Large orders near the quotes are revised most; most revisions improve priority, so non-execution is the common motive.
- Budish, E., Cramton, P., & Shim, J. (2015). The high-frequency trading arms race. *Quarterly Journal of Economics*, 130(4). † Stale quotes get sniped in a race between cancelling and trading.
- Cont, R., Kukanov, A., & Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics*, 12(1). † Order book imbalance predicts short-term price moves; justifies `w_depth`.

## Modeling

- Gode, D., & Sunder, S. (1993). Allocative efficiency of markets with zero-intelligence traders. *Journal of Political Economy*, 101(1). Random budget-constrained traders already produce sensible prices; used only as a sanity-check population.
- Othman, A. (2008). Zero-intelligence agents in prediction markets. *AAMAS*. Zero-intelligence agents with Beta-distributed beliefs trading single shares in a continuous double auction show a favorite-longshot bias (for example, a mean belief of 0.8 trades at 0.742 under diffuse beliefs); our order-book sanity check (`experiments/othman_check.py`).
- Chakraborty, M., Das, S., & Peabody, J. (2015). Price evolution in a continuous double auction prediction market with a scoring-rule based market maker. *AAAI*. Combines an order book with an automated market maker in one simulation.
- Arthur, W. B., Holland, J., LeBaron, B., Palmer, R., & Tayler, P. (1997). Asset pricing under endogenous expectations in an artificial stock market. *The Economy as an Evolving Complex System II*. Classic complex-systems market model with heterogeneous adaptive agents.
- Grimm, V., et al. (2005). Pattern-oriented modeling of agent-based complex systems. *Science*, 310(5750). † Calibrate an ABM by matching several observed patterns at once; our calibration method.
- Grimm, V., et al. (2020). The ODD protocol for describing agent-based and other simulation models: a second update. *JASSS*, 23(2). Standard format for documenting the model in the report.

## Agent behavior

- Kelly, J. L. (1956). A new interpretation of information rate. *Bell System Technical Journal*, 35(4). Optimal bet sizing for repeated bets; basis of agents' fractional-Kelly target positions.
- Beygelzimer, A., Langford, J., & Pennock, D. (2012). Learning performance of prediction markets with Kelly bettors. *AAMAS*. † Studies prediction markets populated by Kelly bettors, a theoretical reference for our agents.
- Yaniv, I. (2004). Receiving other people's advice: Influence and benefit. *Organizational Behavior and Human Decision Processes*, 93(1). † More knowledgeable people discount advice more; justifies `w_hearsay` falling with `n_clues`.
- DeGroot, M. (1974). Reaching a consensus. *Journal of the American Statistical Association*, 69(345). Weighted averaging of others' estimates; the form of our own-estimate vs. market average.
- Eyster, E., Rabin, M., & Vayanos, D. (2019). Financial markets where traders neglect the informational content of prices. *Journal of Finance*, 74(1). † Traders who underweight what prices reveal trade more and keep markets active; justifies agents weighting the price by less than 1 (`w_market < 1`).
- Enke, B., & Zimmermann, F. (2019). Correlation neglect in belief formation. *Review of Economic Studies*, 86(1). † People treat overlapping signals as independent; justifies agents adding their own evidence to a price that already contains part of it.
- Gjerstad, S., & Dickhaut, J. (1998). Price formation in double auctions. *Games and Economic Behavior*, 22(1). † Traders form beliefs from order book history ("heuristic belief learning").
- Black, F. (1986). Noise. *Journal of Finance*, 41(3). † Trading on noise is what makes markets liquid.
- De Long, J. B., Shleifer, A., Summers, L., & Waldmann, R. (1990). Noise trader risk in financial markets. *Journal of Political Economy*, 98(4). † Uninformed traders persist and affect prices.
- Forsythe, R., Nelson, F., Neumann, G., & Wright, J. (1992). Anatomy of an experimental political stock market. *American Economic Review*, 82(5). † Iowa Electronic Markets traders bet on what they hope will happen; real uninformed traders exist.
- Bikhchandani, S., Hirshleifer, D., & Welch, I. (1992). A theory of fads, fashion, custom, and cultural change as informational cascades. *Journal of Political Economy*, 100(5). Copying others can lock in wrong beliefs; the risk when `market_trust` is high.
- Banerjee, A. (1992). A simple model of herd behavior. *Quarterly Journal of Economics*, 107(3). Companion herding model.
- Tversky, A., & Kahneman, D. (1992). Advances in prospect theory. *Journal of Risk and Uncertainty*, 5(4). People overweight small probabilities; its estimates help set the `g` distribution, though we use the symmetric log-odds form instead of its weighting function.
- Goldstein, W., & Einhorn, H. (1987). Expression theory and the preference reversal phenomena. *Psychological Review*, 94(2). † Introduces the linear-in-log-odds probability weighting function our agents use.
- Gonzalez, R., & Wu, G. (1999). On the shape of the probability weighting function. *Cognitive Psychology*, 38(1). † Estimates the linear-in-log-odds weighting function's curvature and crossover; check its estimates when setting the `g` distribution.
- Barber, B., & Odean, T. (2008). All that glitters: The effect of attention and news on the buying behavior of individual and institutional investors. *Review of Financial Studies*, 21(2). † Individual investors buy attention-grabbing stocks right after news; supports impatient inexperienced agents.
- Snowberg, E., & Wolfers, J. (2010). Explaining the favorite-long shot bias: Is it risk-love or misperceptions? *Journal of Political Economy*, 118(4). The favorite-longshot bias comes from misperceived probabilities; explanation if our rules alone do not produce it.

## Information spread

- Daley, D. J., & Kendall, D. G. (1964). Epidemics and rumours. *Nature*, 204. Classic rumor model with spreaders and stiflers.
- Erdos, P., & Renyi, A. (1959). On random graphs I. *Publicationes Mathematicae*, 6. † Random graphs; our null network.
- Watts, D., & Strogatz, S. (1998). Collective dynamics of small-world networks. *Nature*, 393. Clustered networks with short paths.
- Barabasi, A.-L., & Albert, R. (1999). Emergence of scaling in random networks. *Science*, 286. Scale-free networks with hubs.
- Holme, P., & Kim, B. J. (2002). Growing scale-free networks with tunable clustering. *Physical Review E*, 65(2). † Networks with both hubs and clustering; our default.
- Holland, P., Laskey, K., & Leinhardt, S. (1983). Stochastic blockmodels: First steps. *Social Networks*, 5(2). † Networks with built-in communities; our echo-chamber network.
- Newman, M. (2003). The structure and function of complex networks. *SIAM Review*, 45(2). † Real social networks combine hubs, clustering, and communities.
- Ugander, J., Karrer, B., Backstrom, L., & Marlow, C. (2011). The anatomy of the Facebook social graph. arXiv:1111.4503. † Measured structure of a real large social network.
- Centola, D. (2010). The spread of behavior in an online social network experiment. *Science*, 329(5996). † Clustered networks spread behavior better than random ones.
- Del Vicario, M., et al. (2016). The spreading of misinformation online. *PNAS*, 113(3). † Misinformation spreads inside like-minded communities.
- Cinelli, M., et al. (2021). The echo chamber effect on social media. *PNAS*, 118(9). † Echo chambers across platforms.
- Vosoughi, S., Roy, D., & Aral, S. (2018). The spread of true and false news online. *Science*, 359(6380). † False news spreads faster than true news, mostly through people rather than bots.
- Wardle, C., & Derakhshan, H. (2017). Information Disorder: Toward an interdisciplinary framework. Council of Europe report DGI(2017)09. † Separates misinformation (false, no intent) from disinformation (false, deliberate).
- Ozsoylev, H., Walden, J., Yavuz, M. D., & Bildik, R. (2014). Investor networks in the stock market. *Review of Financial Studies*, 27(5). Central investors in an empirical information network trade earlier and earn more.
- van Bommel, J. (2003). Rumors. *Journal of Finance*, 58(4). An informed trader trades, spreads a rumor to followers, and trades again on the overshoot; motivates trade-before-telling.
- Kogan, S., Moskowitz, T., & Niessner, M. (2023). Social media and financial news manipulation. *Review of Finance*, 27(4). Paid fake articles raised retail trading and prices; exposure reduced trust in all news on those platforms.

## Manipulation

- Allen, F., & Gale, D. (1992). Stock-price manipulation. *Review of Financial Studies*, 5(3). † Separates trade-based, information-based, and action-based manipulation, and shows trade-based manipulation can profit.
- Hanson, R., Oprea, R., & Porter, D. (2006). Information aggregation and manipulation in an experimental market. *Journal of Economic Behavior & Organization*, 60(4). Lab manipulators failed because others traded against them.
- Hanson, R., & Oprea, R. (2009). A manipulator can aid prediction market accuracy. *Economica*, 76(302). A manipulator is a free lunch that draws in informed traders.
- Rhode, P., & Strumpf, K. (2004). Historical presidential betting markets. *Journal of Economic Perspectives*, 18(2). Manipulation attempts in historical betting markets rarely lasted.
- Rothschild, D., & Sethi, R. (2016). Trading strategies and market microstructure: Evidence from a prediction market. *Journal of Prediction Markets*, 10(1). https://papers.ssrn.com/abstract=2322420. A single large Intrade trader apparently held a candidate's price up.
- Rasooly, I., & Rozzi, R. (2025). How manipulable are prediction markets? arXiv:2503.03312. https://arxiv.org/abs/2503.03312. Field experiment: random price pushes persisted but faded faster in markets with more traders and volume; manipulation depends on liquidity.
- Wang, X., & Wellman, M. (2017). Spoofing the limit order book: An agent-based model. *AAMAS*. † Spoofing works in an ABM when traders use order book information.
- Lee, E. J., Eom, K. S., & Park, K. S. (2013). Microstructure-based manipulation: Strategic behavior and performance of spoofing traders. *Journal of Financial Markets*, 16(2). † Empirical evidence of profitable spoofing in Korean markets.
- Benabou, R., & Laroque, G. (1992). Using privileged information to manipulate markets: Insiders, gurus, and credibility. *Quarterly Journal of Economics*, 107(3). † Insiders and gurus can profit from strategically false statements.
- Fortune (Nov 7, 2025). Columbia researchers on wash trading in Polymarket. https://fortune.com/2025/11/07/polymarket-wash-trading-inflated-prediction-markets-columbia-research. Trading volume itself can be manipulated.

## Kalshi-specific

- Kalshi, Fee Schedule (official PDF). https://kalshi.com/docs/kalshi-fee-schedule.pdf. The taker and maker fee formulas, rounding, and free cancellation.
- Kalshi API docs, Fee Rounding. https://docs.kalshi.com/getting_started/fee_rounding. Kalshi keeps a fee accumulator per order across all its fills, so an order filled in pieces pays about what one equivalent fill would; our order book rounds each order's total fee up once.
- Kalshi Trade API v2 (public market data). https://api.elections.kalshi.com/trade-api/v2. Source for the calibration check and the trade-size and spread patterns.
- Kalshi API docs, Cancel Up to 10,000 Orders. https://docs.kalshi.com/api-reference/orders/cancel-up-to-10000-orders. Bulk cancellation is supported, which shows cancel-and-requote is normal practice.
- Burgi, C., Deng, W., & Whelan, K. (2025). Makers and Takers: The Economics of the Kalshi Prediction Market. UCD Economics WP2025_19 / CESifo WP 12122. https://www.ucd.ie/economics/t4media/WP2025_19.pdf. Kalshi prices are informative but show a strong favorite-longshot bias, and makers out-earn takers.

## AI agents in markets

- Galanis, S. (2026). Information Aggregation with AI Agents. arXiv:2604.20050. https://arxiv.org/abs/2604.20050. LLM agents in an LMSR market with private signals; for the discussion section.
- Information Homogenization Induces Herding in Retrieval-Augmented LLM Agent Markets (ICLR 2026). https://www.iclr.cc/virtual/2026/10016534. Agents reading the same news trade alike without communicating.
