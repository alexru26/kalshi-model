# AGENTS.md

Project memory for agents working in this repository.

## Overview

An agent-based model of one Kalshi binary prediction market. Three experiments: E1 Kalshi rules (AMM vs CLOB vs Kalshi), E2 information spread (network type, disinformation ratio), E3 manipulation (pump-and-dump, spoofing), plus a favorite-longshot calibration check against public Kalshi data.

`PLAN.md` is the source of truth for design, experiments, and build order (Section 4). Read it before writing code. Sources live in `SOURCES.md`. No code exists yet; when tooling is added, record the build and test commands here.

## Core principles

- Every run has a known right answer: `P*`, the exact Bayesian posterior from every real clue any agent holds. Prices are scored against it.
- Experiments change one variable at a time around one shared default configuration. Write predictions first.
- "Agents" means traders only (the manipulator is an agent with extra rules). The market maker and information sources are model components. Agents have no type labels; behavior comes from trait variables.
- Money is integer cents. One YES order book; positions are signed YES counts; everything is fully collateralized.
- Conservation invariants (`PLAN.md` Section 4, Step 4) are asserted every step in debug mode and in tests.
- Every configuration runs on the same seed list; each run is one data point.

## Development directives

- Follow the user's global guidelines; in particular, never use the em dash and never add an agent as commit co-author.
- Follow the `PLAN.md` build order; write order-book tests before the order book.
- Keep `PLAN.md` concise: big points only. Put sources in `SOURCES.md` with a one-sentence summary each.
- When a decision changes the plan, update `PLAN.md` and remove the matching open question here.

## Open questions

- Numeric defaults are starting guesses until the calibration step (`PLAN.md` Section 4, Step 7) freezes them.
- Sources marked † in `SOURCES.md` still need their details verified.

## Things to remember

- Mesa 3.5 requires Python >= 3.12, uses `rng=` (not the deprecated `seed=`), and `batch_run` uses `spawn`, so the model must be importable at module level.
- No Chrome is installed in this WSL environment; browser screenshots do not work here.
- `.lavish/` is gitignored; `.lavish/plan-summary.html` predates the redesign, so regenerate rather than hand-edit.

## Maintaining this file

Keep this file short: only general context that almost every session needs. It is loaded into every session, so each line costs tokens.

- Do not repeat what `PLAN.md` or the code already says; point to the section instead.
- Prefer rewriting or pruning entries over appending. Remove resolved open questions and stale notes.
- When the file grows bloated, or a recurring theme (for example, order-book conventions or experiment workflow) needs more than a few lines, move it into its own project skill with the `skill-creator` skill and leave a one-line pointer here.
