# Model card

`quorum` is not a trained model. It is a system that composes a language model with
classical statistics to predict how a population answers a question, and this card
describes that system: what it does, what it was measured on, where the measurement
stops, and what it should not be used for.

## What it is

Given target population marginals and a question, `quorum` synthesizes a weighted
population of agents, asks a language model how a stratified sample of them would
answer, propagates those answers to the rest of the population with a classical
model, poststratifies onto the target cells, and reports the result with an interval.

- **Input** a declarative spec: target marginals, population size, question text and
  response options, predictor settings, budget.
- **Output** a distribution over the response options, with a credible interval, a
  breakdown by demographic dimension, and a run record carrying the spec hash, seed,
  population hash and cost.

## What it was measured on

- **Population targets** American Community Survey 2024 1-Year PUMS, person records,
  US adults aged 18 and over: 2,790,132 records representing 267.2 million people.
- **Answers** General Social Survey 2024, 3,309 respondents, 41 scored items,
  11 randomized wording experiments.
- **Attributes** age band, sex, education, race, marital status.

The two sources are independent. The population is built from census data; the answers
it is scored against come from a survey the synthesis never sees. They meet only at the
shared attribute taxonomy.

## Current state of the numbers

**No accuracy claim in this repository currently rests on a language model.** The
engine ships with an offline stub provider that hashes a prompt into a distribution so
that the pipeline, the tests and the evaluation harness run anywhere with no key and no
network. The stub knows nothing about the world and cannot predict anything.
`EVAL.md`, every published artifact and every run record say so explicitly whenever the
stub produced the answers.

The accuracy gates in `quorum/eval/gates.py` are split accordingly. Gates about
plumbing are enforced on every run. Gates about prediction (skill against the baseline,
calibration, interval coverage, wording-gap sign and magnitude) are enforced only when
a real model answered, and are reported as skipped otherwise. Running
`make eval EVAL_ARGS="--provider anthropic --model claude-opus-5 --check"` activates
them.

## Assumptions that are known to be shaky

**Soft counts in the pooling model.** Partial pooling treats each agent's response
distribution as fractional evidence. The Dirichlet-multinomial machinery underneath is
written for integer draws, so the posterior width rests on an approximation. The first
attempt at fitting the pooling strength by marginal likelihood failed outright for this
reason: it saturated at maximal pooling and flattened every segment onto the global
mean regardless of the gradient in the data. The concentration is now fitted by
matching moments, which sidesteps the likelihood, and the resulting interval is checked
against a bootstrap that shares none of these assumptions. The two do not always agree,
and `interval_agreement` reports the gap rather than hiding it.

**Segment estimates are pooled by construction.** A demographic cell with little
evidence reads close to the population average because it was shrunk there, not because
the simulation found it to be average. `PooledCells.shrinkage` reports how much of each
cell's number came from the prior, and the published report says this on the page.

**Synthesis carries one state's worth of joint structure at national margins.** The
seed microdata supplies the joint distribution between attributes; raking fixes each
one-way margin exactly. Interactions the seed does not represent are not corrected.

**The survey changed mode.** The GSS moved to a largely web-based design in 2021.
Nothing here compares across years, and nothing should without accounting for it.

**Five attributes is a coarse person.** Age band, sex, education, race and marital
status do not determine an opinion, and the engine does not claim they do. Latent
traits disperse agents within a cell; they are noise with a purpose, not knowledge.

## What it should not be used for

- **Any claim about a real, identifiable person or a small group.** The agents are
  synthetic and the smallest meaningful unit is a poststratification cell.
- **Substituting for a survey where the answer matters.** A simulated topline is an
  estimate produced by a model of a population, not a measurement of one.
- **Questions far outside the domain it was scored on.** Everything measured here is
  general-population attitude questions with two to seven ordered options.
- **Establishing that an effect is real.** The report's `resolved` column says whether
  the simulation could distinguish a direction, which is a statement about the
  simulation.

## Reproducibility

Current models reject sampling parameters, so a run cannot be pinned by seed or
temperature. It is pinned by remembering what the model said: every answer is
content-addressed by model, system prompt, user prompt and token limit, and the cache
is committed. With it present, a run replays exactly, offline, for free. Population
synthesis, sampling, trait assignment, graph construction and posterior draws are all
seeded and deterministic; a test asserts that the same spec and seed produce the same
population fingerprint and the same answer.

## Cost

Four levers, each measured rather than asserted, reported by `CostMeter`:

| lever | mechanism |
|---|---|
| Hybrid prediction | the model is called once per archetype, not once per agent, so model cost stops scaling with population size |
| Prompt caching | the shared half of the prompt goes first and is marked cacheable, so repeat input bills at a tenth |
| Batch pricing | metered at half price for work that is not latency-sensitive |
| Response cache | committed, so replays cost nothing |

A budget is checked before each call rather than after, and raises rather than
overspending.

## License

MIT. The vendored data carries its own terms; see
[`data/vendor/PROVENANCE.md`](data/vendor/PROVENANCE.md).
