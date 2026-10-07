# AGENTS.md

Project memory for agents working in this repository.

## Overview

An agent-based model of one Kalshi binary prediction market. It asks how Kalshi's rules (P(1-P) taker fee rounded up to the cent, maker/taker order book, free cancels) change how fast and how accurately a market absorbs true news, false rumors, and manipulation, compared with a classical fee-free LMSR market.

`PLAN.md` is the source of truth for design, experiments, and build order (Section 9). Read it before writing code. No code exists yet; when tooling is added, record the build and test commands here.

Scope is the Core tier (`PLAN.md` Section 8): E1 fees, E2 news and rumors, E3 brute-force manipulator, plus a favorite-longshot calibration check against public Kalshi data. Nothing from the out-of-scope list unless the user asks.

## Core principles

- Every run has a known right answer: `P*`, the exact Bayesian posterior from released clues. Prices are scored against it.
- Same agents, different market rules (M0-M5). Vary at most two parameters per experiment; write predictions first.
- Money is integer cents. One YES order book; positions are signed YES counts; everything is fully collateralized.
- Conservation invariants (`PLAN.md` Section 9, Step 4) are asserted every step in debug mode and in tests.
- Models take `rng=seed` and draw all randomness from the model. Every configuration runs on the same seed list; each run is one data point.

## Architecture outline

Planned (`PLAN.md` Section 10). Market mechanics (`fees.py`, `lmsr.py`, `orderbook.py`) are plain Python with no Mesa dependency. Mesa covers agents, the model step, data collection, and `batch_run` sweeps in `experiments/`. All parameters live in one `Config` dataclass in `src/config.py`. Figures are drawn from `results/*.csv`.

## Development directives

- Follow the user's global guidelines in `~/AGENTS.md`; in particular, never use the em dash and never add an agent as commit co-author.
- Follow the `PLAN.md` build order; write order-book tests before the order book.
- Describe work as ordered implementation steps, not week-by-week timelines.
- When a decision changes the plan, update `PLAN.md` and remove the matching open question here.

## Open questions

- Commit `.lavish/` (generated review pages) or add it to a `.gitignore`?
- Numeric defaults and the market maker's exact rules are not yet set in `PLAN.md`.
- How the informed agent's "edge beats fee" rule works under M0, which has no bid/ask.
- Exact definition of "reaction half-life".

## Things to remember

- Mesa 3.5 requires Python >= 3.12, uses `rng=` (not the deprecated `seed=`), and `batch_run` uses `spawn`, so the model must be importable at module level.
- No Chrome is installed in this WSL environment; browser screenshots do not work here.
- `.lavish/plan-summary.html` predates the scope cut; regenerate rather than hand-edit.

## Maintaining this file

Keep this file short: only general context that almost every session needs. It is loaded into every session, so each line costs tokens.

- Do not repeat what `PLAN.md` or the code already says; point to the section instead.
- Prefer rewriting or pruning entries over appending. Remove resolved open questions and stale notes.
- When the file grows bloated, or a recurring theme (for example, order-book conventions or experiment workflow) needs more than a few lines, move it into its own project skill with the `skill-creator` skill and leave a one-line pointer here.
