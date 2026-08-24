# Vendored ground truth

Everything in this directory is derived from public data by
[`scripts/fetch_data.py`](../../scripts/fetch_data.py) and committed so that a fresh
clone can run the whole pipeline offline. Regenerate it with `make data`, which
re-downloads roughly 650 MB of source files into the untracked `data/raw/`.

The two sources are deliberately independent. The population is synthesized from
census marginals; the answers it is scored against come from a survey the synthesis
never sees. They meet only at the shared attribute taxonomy in
[`quorum/data/schema.py`](../../src/quorum/data/schema.py).

## `acs_marginals.json`, `acs_microdata.csv.gz`

American Community Survey 2024 1-Year Public Use Microdata Sample, person records,
US Census Bureau. Public domain as a work of the United States federal government.

Source: <https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/csv_pus.zip>

- **Universe:** US adults aged 18 and over.
- **Weighting:** person weights (`PWGTP`).
- **Coverage:** 2,790,132 person records representing 267.2 million adults.
- **Variables used:** `AGEP`, `SEX`, `SCHL`, `RAC1P`, `MAR`, `PWGTP`.
- `acs_marginals.json` holds the weighted share of every level of every attribute,
  computed over all 2.8 million records, and is what synthesized populations are
  raked to.
- `acs_microdata.csv.gz` is a deterministic 40,000-row subsample of the same records.
  It is the synthesis seed: raking fixes the one-way margins, and the seed is what
  supplies the joint structure between them.

## `gss_questions.json`

General Social Survey, Cross-Sectional Cumulative Data 1972-2024, Release 3A (July
2026), NORC at the University of Chicago.

Source: <https://gss.norc.org/us/en/gss/get-the-data.html>

- **Year scored:** 2024, 3,309 respondents.
- **Weighting:** `wtssps`, the post-stratified weight NORC supplies for 2021 onward.
- **What is vendored:** aggregate statistics only. Weighted toplines, their standard
  errors at the Kish effective sample size, unweighted and effective sample sizes, and
  breakdowns by age band, sex and education where at least 60 respondents support the
  estimate. No respondent level records are redistributed.
- **Question wording** is read verbatim out of the published codebook PDF rather than
  transcribed by hand. Wording is the entire subject of the split-ballot experiments,
  so it is not something the pipeline is allowed to paraphrase.

### The wording experiments

Eleven of the spending items are asked in two randomized forms. Respondents are
assigned at random to one form or the other, so the two arms describe the same
population and any difference between them is caused by the wording alone. That makes
them an answer key an ordinary topline cannot be: a simulator that predicts both arms
identically has not modelled the question at all.

The best known of them is welfare. In 2024, 33.4 percent said we spend too little on
"welfare"; 70.5 percent said we spend too little on "assistance to the poor". Same
population, same survey, same year, 37 points apart.

### A caveat worth stating

The GSS moved to a mixed-mode design with a large web component in 2021. Comparisons
between 2024 and pre-2020 GSS years carry a mode effect that has nothing to do with
anything modelled here. Nothing in this repository makes cross-year comparisons, and
nothing should without accounting for it.
