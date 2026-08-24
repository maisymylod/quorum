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
- [ ] Population synthesis (raking to real ACS marginals) and fidelity gates
- [ ] World model: social graph and bounded-confidence peer influence
- [ ] Hybrid LLM + classical response prediction, with cost accounting
- [ ] Poststratification, hierarchical Bayesian intervals, Monte Carlo
- [ ] Evaluation harness: backtest, baselines, ablation grid, calibration
- [ ] Publication layer: decision-ready report

Numbers will appear in `EVAL.md` only once they come from actually running the code.

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
make test          # tests plus the quality gate
.venv/bin/quorum new turnout --path specs/turnout.yaml
.venv/bin/quorum validate specs/turnout.yaml
```

## Capabilities demonstrated

Population synthesis and raking to known marginals; agent-based simulation; hybrid
LLM plus classical modelling; poststratification and hierarchical Bayesian inference;
Monte Carlo uncertainty propagation; evaluation, calibration and accuracy measurement
against real ground truth; cost and latency engineering for large simulated populations;
reproducible, seed-deterministic pipelines.

## License

MIT.
