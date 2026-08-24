#!/usr/bin/env python3
"""Regenerate the vendored ground-truth extracts from their public sources.

Run this to rebuild everything under ``data/vendor/`` from scratch:

    python scripts/fetch_data.py --all

The raw downloads are large (roughly 600 MB of census microdata and 45 MB of survey
data) and are cached under ``data/raw/``, which is not tracked. What lands in
``data/vendor/`` is small, derived, and committed, so a clone can run the whole
pipeline offline. Only aggregate statistics are vendored from the survey; no
respondent level rows are redistributed.

Sources
-------
American Community Survey 2024 1-Year Public Use Microdata Sample (US Census Bureau,
public domain), for the population marginals and the joint structure that synthesis
starts from.

General Social Survey 1972-2024 Cross-Sectional Cumulative Data (NORC at the
University of Chicago), for the survey answers a prediction is scored against.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quorum.data.harmonize import (  # noqa: E402
    ACS_COLUMNS,
    effective_sample_size,
    harmonize_acs,
    harmonize_gss,
    share_standard_errors,
    weighted_shares,
)
from quorum.data.schema import ATTRIBUTES, LEVELS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
VENDOR = ROOT / "data" / "vendor"

ACS_YEAR = 2024
ACS_URL = f"https://www2.census.gov/programs-surveys/acs/data/pums/{ACS_YEAR}/1-Year/csv_pus.zip"
GSS_URL = "https://gss.norc.org/content/dam/gss/get-the-data/documents/stata/GSS_stata.zip"
GSS_DTA = "GSS_stata/gss7224_r3a.dta"
GSS_CODEBOOK = "GSS_stata/GSS 2024 Codebook R3a.pdf"
GSS_YEAR = 2024
GSS_WEIGHT = "wtssps"

#: How many microdata rows to vendor as the synthesis seed. Large enough to carry the
#: joint structure of five attributes, small enough to commit.
SEED_ROWS = 40_000
SEED_SEED = 20260824

#: Dimensions that ground-truth answers are broken out by, for segment level scoring.
SEGMENT_DIMENSIONS = ("age_band", "sex", "education")

#: Minimum unweighted respondents before a segment estimate is published. Below this
#: the segment is too noisy to be an answer key.
MIN_SEGMENT_N = 60

SPENDING_OPTIONS = ("too little", "about right", "too much")
SPENDING_CODES = {1: "too little", 2: "about right", 3: "too much"}

#: The randomized wording splits. Each pair asks the same underlying spending question
#: of a random half of respondents, differing only in how the item is named, so the
#: difference between arms is caused by wording and nothing else.
WORDING_EXPERIMENTS = [
    ("welfare", "natfare", "natfarey"),
    ("foreign_aid", "nataid", "nataidy"),
    ("defense", "natarms", "natarmsy"),
    ("cities", "natcity", "natcityy"),
    ("crime", "natcrime", "natcrimy"),
    ("drugs", "natdrug", "natdrugy"),
    ("education", "nateduc", "nateducy"),
    ("environment", "natenvir", "natenviy"),
    ("health", "natheal", "nathealy"),
    ("race", "natrace", "natracey"),
    ("space", "natspac", "natspacy"),
]

#: Single-form items, included so the backtest is not only wording experiments.
STANDALONE_ITEMS = {
    "natroad": SPENDING_CODES,
    "natsoc": SPENDING_CODES,
    "natchld": SPENDING_CODES,
    "natsci": SPENDING_CODES,
    "natenrgy": SPENDING_CODES,
    "cappun": {1: "favor", 2: "oppose"},
    "gunlaw": {1: "favor", 2: "oppose"},
    "grass": {1: "legal", 2: "not legal"},
    "letdie1": {1: "yes", 2: "no"},
    "fear": {1: "yes", 2: "no"},
    "happy": {1: "very happy", 2: "pretty happy", 3: "not too happy"},
    "satfin": {1: "satisfied", 2: "more or less", 3: "not at all satisfied"},
    "health": {1: "excellent", 2: "good", 3: "fair", 4: "poor"},
    "trust": {1: "can trust", 2: "cannot be too careful", 3: "depends"},
    "helpful": {1: "try to be helpful", 2: "look out for themselves", 3: "depends"},
    "fair": {1: "take advantage", 2: "try to be fair", 3: "depends"},
    "polviews": {
        1: "extremely liberal",
        2: "liberal",
        3: "slightly liberal",
        4: "moderate",
        5: "slightly conservative",
        6: "conservative",
        7: "extremely conservative",
    },
    "partyid": {
        0: "strong democrat",
        1: "not strong democrat",
        2: "independent near democrat",
        3: "independent",
        4: "independent near republican",
        5: "not strong republican",
        6: "strong republican",
    },
    "spanking": {
        1: "strongly agree",
        2: "agree",
        3: "disagree",
        4: "strongly disagree",
    },
}


# -- downloading -------------------------------------------------------------------


def download(url: str, destination: Path) -> Path:
    """Fetch ``url`` to ``destination`` unless it is already there."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"  cached {destination.name} ({destination.stat().st_size / 1e6:.0f} MB)")
        return destination
    print(f"  downloading {url}")
    import httpx

    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_bytes(chunk_size=1 << 20):
                handle.write(chunk)
    print(f"  wrote {destination.name} ({destination.stat().st_size / 1e6:.0f} MB)")
    return destination


# -- census ------------------------------------------------------------------------


def _acs_chunks(archive: Path, chunksize: int = 500_000) -> Iterable[pd.DataFrame]:
    with zipfile.ZipFile(archive) as zf:
        members = sorted(n for n in zf.namelist() if n.endswith(".csv"))
        if not members:
            raise RuntimeError(f"{archive} contains no CSV member")
        for member in members:
            print(f"  reading {member}")
            with zf.open(member) as handle:
                reader = pd.read_csv(
                    io.TextIOWrapper(handle, encoding="utf-8"),
                    usecols=list(ACS_COLUMNS),
                    chunksize=chunksize,
                    dtype="float64",
                )
                for chunk in reader:
                    yield chunk


def build_acs(archive: Path) -> None:
    """Write national adult marginals plus a microdata seed."""
    counts = {a: pd.Series(dtype=float) for a in ATTRIBUTES}
    kept_frames: list[pd.DataFrame] = []
    records = 0
    weight_total = 0.0
    rng = np.random.default_rng(SEED_SEED)
    # Reservoir-free approach: keep a fixed fraction of every chunk, then subsample
    # once at the end. Deterministic and never holds the full file in memory.
    keep_fraction = 0.02

    for chunk in _acs_chunks(archive):
        adults = harmonize_acs(chunk)
        if adults.empty:
            continue
        records += len(adults)
        weight_total += float(adults["weight"].sum())
        for attribute in ATTRIBUTES:
            grouped = adults.groupby(attribute, observed=True)["weight"].sum()
            counts[attribute] = counts[attribute].add(grouped, fill_value=0.0)
        mask = rng.random(len(adults)) < keep_fraction
        if mask.any():
            kept_frames.append(adults.loc[mask])

    marginals = {}
    for attribute in ATTRIBUTES:
        series = counts[attribute].reindex(LEVELS[attribute]).fillna(0.0)
        marginals[attribute] = {
            level: float(share) for level, share in (series / series.sum()).items()
        }

    VENDOR.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": {
            "name": "American Community Survey Public Use Microdata Sample",
            "publisher": "US Census Bureau",
            "year": ACS_YEAR,
            "product": "1-Year PUMS, person records",
            "url": ACS_URL,
            "license": "Public domain (US federal government work)",
            "variables": list(ACS_COLUMNS),
        },
        "universe": f"US adults aged 18 and over, ACS {ACS_YEAR} 1-Year PUMS, person weights (PWGTP)",
        "records": records,
        "population_total": weight_total,
        "marginals": marginals,
    }
    path = VENDOR / "acs_marginals.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  wrote {path.name}: {records:,} records, {weight_total / 1e6:.1f}M adults")

    seed = pd.concat(kept_frames, ignore_index=True)
    if len(seed) > SEED_ROWS:
        take = np.sort(rng.choice(len(seed), size=SEED_ROWS, replace=False))
        seed = seed.iloc[take].reset_index(drop=True)
    seed_path = VENDOR / "acs_microdata.csv.gz"
    seed.to_csv(seed_path, index=False, compression="gzip")
    print(f"  wrote {seed_path.name}: {len(seed):,} rows ({seed_path.stat().st_size / 1e6:.1f} MB)")


# -- survey ------------------------------------------------------------------------


def _extract(archive: Path, member: str) -> Path:
    target = RAW / member
    if not target.exists():
        print(f"  extracting {member}")
        with zipfile.ZipFile(archive) as zf:
            zf.extract(member, RAW)
    return target


def question_wording(codebook: Path) -> dict[str, str]:
    """Pull verbatim item wording out of the published codebook.

    Using the codebook rather than hand-typing question text is the difference
    between simulating the survey that was actually run and simulating a paraphrase
    of it. Wording is the whole point of the split-ballot experiments, so it is not a
    detail that can be approximated.
    """
    text_path = RAW / "gss_codebook.txt"
    if not text_path.exists():
        if shutil.which("pdftotext") is None:
            raise RuntimeError(
                "pdftotext is required to extract question wording "
                "(brew install poppler). Only needed when regenerating vendored data."
            )
        subprocess.run(
            ["pdftotext", "-layout", str(codebook), str(text_path)], check=True
        )
    raw = text_path.read_text(errors="replace")
    wording: dict[str, str] = {}
    # A label starts on the "Label:" line (sometimes on that line, sometimes the next)
    # and runs until the block's next field. Anchoring the end on an explicit set of
    # following-field markers is what stops a label from swallowing the next
    # variable's block, which silently attaches the wrong question text to an item.
    pattern = re.compile(
        r"^Variable:\s+(\w+)\s+Type:[^\n]*\n"
        r"Label:[ \t]*(.*?)"
        r"(?=^\s*(?:Notes:|Text:|Variable:|LABEL\s|PCT\b)|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    for match in pattern.finditer(raw):
        name = match.group(1).lower()
        label = re.sub(r"\s+", " ", match.group(2)).strip()
        if label and name not in wording:
            wording[name] = label
    return wording


#: Item families whose codebook labels carry only the item, with the shared question
#: stem living on one sibling variable. Without this, an item such as "Health" renders
#: as a bare noun phrase and the simulated respondent is answering a different question
#: from the one the survey asked.
ITEM_FAMILIES = (("nat", "natspac"),)


def _family_stem(variable: str, wording: dict[str, str]) -> str:
    for prefix, source in ITEM_FAMILIES:
        if variable.startswith(prefix):
            match = re.search(r"\(([^)]*)\)", wording.get(source, ""))
            if match:
                return match.group(1).strip()
    return ""


def stata_labels(dta: Path) -> dict[str, str]:
    """Variable labels carried inside the data file itself.

    A second, independent record of what each item says. The published codebook is
    the better source for full question wording, but it is not error free: for at
    least one wording experiment it prints the base form's item text against the
    variant. Holding both lets that be detected rather than silently scored.
    """
    reader = pd.read_stata(dta, iterator=True)
    labels = reader.variable_labels()
    cleaned = {}
    for name, label in labels.items():
        label = re.sub(r":\s*(?:Version|Ver)\s+[A-Z]\s*$", "", str(label)).strip()
        cleaned[name.lower()] = label
    return cleaned


def _question_text(variable: str, wording: dict[str, str]) -> str:
    """Render a codebook label as the question a respondent was actually asked."""
    label = wording.get(variable)
    if not label:
        return variable
    match = re.match(r"^.*?\(([^)]*)\)\s*(.*)$", label)
    if match:
        stem, item = match.group(1).strip(), match.group(2).strip()
    elif _family_stem(variable, wording):
        stem, item = _family_stem(variable, wording), label
    else:
        return label
    stem = stem.replace("...", "").strip(" .")
    stem = stem[0].upper() + stem[1:] if stem else stem
    return f"{stem} {item.rstrip('?')}?"


def _item_record(
    frame: pd.DataFrame,
    variable: str,
    codes: dict[int, str],
    wording: dict[str, str],
    experiment: str | None = None,
    arm_label: str | None = None,
) -> dict | None:
    values = pd.to_numeric(frame[variable], errors="coerce").map(codes)
    options = tuple(codes[k] for k in sorted(codes))
    weights = frame["weight"]
    mask = values.isin(options) & weights.notna() & (weights > 0)
    if mask.sum() < 100:
        return None

    shares = weighted_shares(values, weights, options)
    n_eff = effective_sample_size(weights[mask].to_numpy())
    errors = share_standard_errors(shares, n_eff)

    segments: dict[str, dict[str, list[float]]] = {}
    for dimension in SEGMENT_DIMENSIONS:
        by_level: dict[str, list[float]] = {}
        for level in LEVELS[dimension]:
            in_level = mask & (frame[dimension] == level)
            if int(in_level.sum()) < MIN_SEGMENT_N:
                continue
            by_level[level] = [
                float(x)
                for x in weighted_shares(values[in_level], weights[in_level], options)
            ]
        if by_level:
            segments[dimension] = by_level

    return {
        "id": variable,
        "text": _question_text(variable, wording),
        "options": list(options),
        "topline": [float(x) for x in shares],
        "standard_error": [float(x) for x in errors],
        "n": int(mask.sum()),
        "effective_n": float(n_eff),
        "experiment": experiment,
        "arm_label": arm_label,
        "segments": segments,
    }


def build_gss(archive: Path) -> None:
    """Write the scored question bank for one survey year."""
    dta = _extract(archive, GSS_DTA)
    codebook = _extract(archive, GSS_CODEBOOK)
    wording = question_wording(codebook)
    print(f"  parsed wording for {len(wording):,} variables")

    variables = ["year", "age", "sex", "degree", "race", "marital", GSS_WEIGHT]
    variables += [v for _, a, b in WORDING_EXPERIMENTS for v in (a, b)]
    variables += list(STANDALONE_ITEMS)
    print("  reading survey microdata")
    raw = pd.read_stata(dta, columns=sorted(set(variables)), convert_categoricals=False)
    year_frame = raw[raw["year"] == GSS_YEAR]
    frame = harmonize_gss(year_frame, weight_column=GSS_WEIGHT)
    print(f"  {len(frame):,} respondents in {GSS_YEAR}")

    labels = stata_labels(dta)

    questions: list[dict] = []
    experiments: list[dict] = []
    for experiment_id, base, variant in WORDING_EXPERIMENTS:
        arms = []
        base_text = _question_text(base, wording)
        variant_text = _question_text(variant, wording)
        if variant_text == base_text:
            # The codebook printed the base form's item against the variant. Fall
            # back to the item name the data file carries, which is what actually
            # distinguishes the two arms.
            fallback = dict(wording)
            fallback[variant] = labels.get(variant, "")
            variant_text = _question_text(variant, fallback)
            print(f"  {variant}: codebook wording duplicated {base}, using data file label")
        if variant_text == base_text:
            raise RuntimeError(
                f"wording experiment {experiment_id!r} has identical text on both arms; "
                "a wording split with no wording difference cannot be scored"
            )
        for variable, text in ((base, base_text), (variant, variant_text)):
            record = _item_record(
                frame,
                variable,
                SPENDING_CODES,
                wording,
                experiment=experiment_id,
                arm_label=text,
            )
            if record is not None:
                record["text"] = text
                questions.append(record)
                arms.append(variable)
        if len(arms) == 2:
            experiments.append(
                {
                    "id": experiment_id,
                    "label": f"{experiment_id} spending, wording split",
                    "arms": arms,
                    "contrast_option": "too little",
                }
            )

    for variable, codes in STANDALONE_ITEMS.items():
        if variable not in frame.columns:
            continue
        record = _item_record(frame, variable, codes, wording)
        if record is not None:
            questions.append(record)

    unresolved = [q["id"] for q in questions if not q["text"].endswith("?") or q["text"] == q["id"]]
    if unresolved:
        raise RuntimeError(
            f"no verbatim question wording resolved for {unresolved}; "
            "refusing to vendor items whose text was not read from the codebook"
        )

    payload = {
        "source": {
            "name": "General Social Survey, Cross-Sectional Cumulative Data 1972-2024",
            "publisher": "NORC at the University of Chicago",
            "release": "Release 3A (July 2026)",
            "url": GSS_URL,
            "note": "Only aggregate statistics are redistributed here. "
            "Respondent level records are not included; rerun this script to "
            "regenerate them from the published file.",
        },
        "year": GSS_YEAR,
        "weight_variable": GSS_WEIGHT,
        "respondents": int(len(frame)),
        "segment_dimensions": list(SEGMENT_DIMENSIONS),
        "minimum_segment_n": MIN_SEGMENT_N,
        "questions": questions,
        "experiments": experiments,
    }
    VENDOR.mkdir(parents=True, exist_ok=True)
    path = VENDOR / "gss_questions.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"  wrote {path.name}: {len(questions)} questions, "
        f"{len(experiments)} wording experiments"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acs", action="store_true", help="rebuild census marginals and seed")
    parser.add_argument("--gss", action="store_true", help="rebuild the survey question bank")
    parser.add_argument("--all", action="store_true", help="rebuild everything")
    args = parser.parse_args(argv)
    if not (args.acs or args.gss or args.all):
        parser.error("pass --acs, --gss or --all")

    if args.acs or args.all:
        print("census microdata")
        build_acs(download(ACS_URL, RAW / "csv_pus.zip"))
    if args.gss or args.all:
        print("survey data")
        build_gss(download(GSS_URL, RAW / "GSS_stata.zip"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
