# quorum

[![CI](https://github.com/maisymylod/quorum/actions/workflows/ci.yml/badge.svg)](https://github.com/maisymylod/quorum/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **population simulation engine**. Synthesize a representative population from real
census marginals, simulate how it answers a question, and score the answer against a
published survey result that the simulator never saw.

The claim the repo has to earn is narrow and checkable: *given a question, predict the
population's aggregate answer distribution, and be measurably right.* Every design
decision below serves that claim being verifiable rather than asserted.

## Status

Under construction, built in milestones. What is in `main` today:

- [x] Core objects and contracts: `Agent`, `Population`, `SimulationSpec`, `RunRecord`
- [x] `quorum new` / `quorum validate` and a declarative spec format
- [x] Real ground truth vendored: national census marginals and a scored survey question bank
- [x] Population synthesis: microdata resampling and raking, with fidelity enforced as a gate
- [x] World model: homophilous social graph and bounded-confidence peer influence
- [x] Classical propagation and the baselines a real engine has to beat
- [x] Hybrid LLM and classical response prediction, with cost accounting and a budget guard
- [x] Poststratification, partial pooling, and two independent routes to an interval
- [x] Evaluation harness: backtest, baselines, ablation grid, calibration, generated `EVAL.md`
- [ ] Publication layer: decision-ready report

Numbers will appear in `EVAL.md` only once they come from actually running the code.

## Ground truth

Accuracy is measured against two independent public sources, which is the only reason
the accuracy claim means anything.

| | Source | Role |
|---|---|---|
| Population | American Community Survey 2024 1-Year PUMS, 2.8M adult records representing 267M adults | What a synthesized population is raked to |
| Answers | General Social Survey 2024, 3,309 respondents, 41 scored items | What a prediction is scored against |

The population is built from census data; the answers it is scored against come from a
survey the synthesis never sees. They meet only at a shared attribute taxonomy.

Eleven of those items are **randomized wording experiments**: the same spending question
asked two ways, each to a random half of respondents. Because the halves are randomly
assigned, they describe the same population, so any difference between them is caused by
the wording alone. That is an answer key an ordinary topline cannot be.

The best known of them is welfare. In 2024, **33.4%** said we spend too little on
*"welfare"*; **70.5%** said we spend too little on *"assistance to the poor"*. Same
population, same survey, same year, 37 points apart. A simulator that returns the same
answer for both has not modelled the question at all.

Details and caveats: [`data/vendor/PROVENANCE.md`](data/vendor/PROVENANCE.md).

## The shape of a simulation

```
targets ──▶ synthesize ──▶ world model ──▶ predict ──▶ estimate ──▶ score ──▶ publish
            (raking)       (network,       (hybrid     (post-       (vs real  (report)
                            influence)      LLM +       strat,       survey
                                            classical)  intervals)   topline)
```

Each arrow is a Protocol in [`core/contracts.py`](src/quorum/core/contracts.py), so any
stage can be swapped without touching the others. That is what makes the ablation grid a
matter of configuration rather than code.

## Two design decisions worth calling out

**Populations are columnar, not `list[Agent]`.** The obvious modelling does not survive
scale: at 100k agents every marginal, stratification and reweight becomes a Python loop.
`Population` wraps a single dataframe with an explicit weight column so those operations
are vectorized, and materializing an `Agent` is an explicit, rare act, used where a
single respondent genuinely is the unit, above all prompt rendering.

**Agents return distributions, not choices.** A simulated respondent that reports only
its modal answer throws away the information that makes an aggregate topline calibrated.
Agent responses are distributions over options and aggregation is a weighted average of
them.

## Quickstart

```bash
make install
make demo          # synthesize, ask, estimate, report, from a clean clone
make test          # tests plus the quality gate
make eval          # rerun the ablation grid and regenerate EVAL.md
```

`make demo` runs the welfare wording experiment: the same spending question asked two
ways, against one synthetic population, scored against what a real survey found.

## Capabilities demonstrated

Population synthesis and raking to known marginals; agent-based simulation; hybrid
LLM plus classical modelling; poststratification and hierarchical Bayesian inference;
Monte Carlo uncertainty propagation; evaluation, calibration and accuracy measurement
against real ground truth; cost and latency engineering for large simulated populations;
reproducible, seed-deterministic pipelines.

## License

MIT.
