from __future__ import annotations

import json

import numpy as np
import pytest

from quorum.data.targets import (
    DEFAULT_MARGINALS,
    DEFAULT_QUESTIONS,
    MarginalTargets,
    QuestionBank,
    load_ground_truth,
)


@pytest.fixture
def marginals_payload() -> dict:
    return {
        "source": {"name": "test"},
        "universe": "test adults",
        "population_total": 1000.0,
        "records": 10,
        "marginals": {"sex": {"male": 0.49, "female": 0.51}},
    }


@pytest.fixture
def questions_payload() -> dict:
    return {
        "source": {"name": "test"},
        "year": 2024,
        "weight_variable": "w",
        "questions": [
            {
                "id": "a",
                "text": "Form A?",
                "options": ["too little", "about right", "too much"],
                "topline": [0.3, 0.4, 0.3],
                "standard_error": [0.01, 0.01, 0.01],
                "n": 900,
                "effective_n": 700.0,
                "experiment": "split",
                "arm_label": "Form A",
                "segments": {"sex": {"male": [0.2, 0.4, 0.4]}},
            },
            {
                "id": "b",
                "text": "Form B?",
                "options": ["too little", "about right", "too much"],
                "topline": [0.7, 0.2, 0.1],
                "standard_error": [0.01, 0.01, 0.01],
                "n": 900,
                "effective_n": 700.0,
                "experiment": "split",
                "arm_label": "Form B",
                "segments": {},
            },
        ],
        "experiments": [
            {"id": "split", "label": "wording split", "arms": ["a", "b"], "contrast_option": "too little"}
        ],
    }


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def test_marginals_load_and_vectorize(tmp_path, marginals_payload):
    targets = MarginalTargets.load(_write(tmp_path, "m.json", marginals_payload))
    np.testing.assert_allclose(targets.vector("sex"), [0.49, 0.51])
    assert targets.attributes == ("sex",)


def test_marginals_reject_an_incomplete_level_set(tmp_path, marginals_payload):
    marginals_payload["marginals"]["sex"] = {"male": 1.0}
    with pytest.raises(ValueError, match="expected"):
        MarginalTargets.load(_write(tmp_path, "m.json", marginals_payload))


def test_marginals_reject_shares_that_do_not_sum_to_one(tmp_path, marginals_payload):
    marginals_payload["marginals"]["sex"] = {"male": 0.4, "female": 0.5}
    with pytest.raises(ValueError, match="sum to"):
        MarginalTargets.load(_write(tmp_path, "m.json", marginals_payload))


def test_marginals_reject_an_unknown_attribute(tmp_path, marginals_payload):
    marginals_payload["marginals"]["height"] = {"tall": 1.0}
    with pytest.raises(ValueError, match="unknown attribute"):
        MarginalTargets.load(_write(tmp_path, "m.json", marginals_payload))


def test_marginals_subset(tmp_path, marginals_payload):
    targets = MarginalTargets.load(_write(tmp_path, "m.json", marginals_payload))
    assert targets.subset(["sex"]).attributes == ("sex",)
    with pytest.raises(KeyError, match="no target marginals"):
        targets.subset(["race"])


def test_question_bank_loads_and_indexes(tmp_path, questions_payload):
    bank = QuestionBank.load(_write(tmp_path, "q.json", questions_payload))
    assert len(bank) == 2
    assert bank.ids == ("a", "b")
    assert bank["a"].share("about right") == pytest.approx(0.4)
    np.testing.assert_allclose(bank["a"].segment("sex", "male"), [0.2, 0.4, 0.4])
    assert bank["a"].segment("sex", "female") is None
    assert [q.id for q in bank] == ["a", "b"]


def test_experiment_gap_is_measured_against_the_first_arm(tmp_path, questions_payload):
    bank = QuestionBank.load(_write(tmp_path, "q.json", questions_payload))
    gaps = bank.experiments[0].gap(bank)
    assert gaps["a"] == pytest.approx(0.0)
    assert gaps["b"] == pytest.approx(0.4)


def test_bank_rejects_a_topline_that_does_not_sum_to_one(tmp_path, questions_payload):
    questions_payload["questions"][0]["topline"] = [0.3, 0.3, 0.3]
    with pytest.raises(ValueError, match="topline sums to"):
        QuestionBank.load(_write(tmp_path, "q.json", questions_payload))


def test_bank_rejects_ragged_options(tmp_path, questions_payload):
    questions_payload["questions"][0]["options"] = ["too little", "too much"]
    with pytest.raises(ValueError, match="disagree in length"):
        QuestionBank.load(_write(tmp_path, "q.json", questions_payload))


def test_bank_rejects_an_experiment_with_an_unknown_arm(tmp_path, questions_payload):
    questions_payload["experiments"][0]["arms"] = ["a", "missing"]
    with pytest.raises(ValueError, match="unknown arms"):
        QuestionBank.load(_write(tmp_path, "q.json", questions_payload))


def test_bank_rejects_an_experiment_whose_arms_disagree_on_options(tmp_path, questions_payload):
    questions_payload["questions"][1]["options"] = ["yes", "no", "maybe"]
    with pytest.raises(ValueError, match="different options"):
        QuestionBank.load(_write(tmp_path, "q.json", questions_payload))


def test_bank_rejects_a_contrast_on_a_missing_option(tmp_path, questions_payload):
    questions_payload["experiments"][0]["contrast_option"] = "unsure"
    with pytest.raises(ValueError, match="not a response option"):
        QuestionBank.load(_write(tmp_path, "q.json", questions_payload))


def test_split_partitions_and_validates_the_holdout(tmp_path, questions_payload):
    bank = QuestionBank.load(_write(tmp_path, "q.json", questions_payload))
    calibration, holdout = bank.split(["b"])
    assert calibration.ids == ("a",)
    assert holdout.ids == ("b",)
    with pytest.raises(KeyError, match="unknown questions"):
        bank.split(["zzz"])


# -- the real vendored files ------------------------------------------------------


def test_vendored_ground_truth_loads():
    targets, bank = load_ground_truth(DEFAULT_MARGINALS, DEFAULT_QUESTIONS)
    assert targets.records > 1_000_000
    assert targets.population_total > 200e6
    assert set(targets.attributes) == {"age_band", "sex", "education", "race", "marital"}
    assert len(bank) >= 30
    assert len(bank.experiments) == 11


def test_vendored_wording_experiments_have_distinct_text():
    bank = QuestionBank.load(DEFAULT_QUESTIONS)
    for experiment in bank.experiments:
        texts = {bank[arm].text for arm in experiment.arms}
        assert len(texts) == len(experiment.arms), experiment.id
        for arm in experiment.arms:
            assert bank[arm].text.endswith("?"), arm


def test_the_welfare_wording_gap_is_present_in_the_ground_truth():
    bank = QuestionBank.load(DEFAULT_QUESTIONS)
    welfare = next(e for e in bank.experiments if e.id == "welfare")
    gaps = welfare.gap(bank)
    # "Assistance to the poor" draws far more support than "welfare". If this ever
    # stops holding, the vendored data has been rebuilt wrongly.
    assert gaps["natfarey"] > 0.30


def test_vendored_marginals_are_plausible_for_us_adults():
    targets = MarginalTargets.load(DEFAULT_MARGINALS)
    sex = targets.marginals["sex"]
    assert 0.45 < sex["male"] < 0.52
    assert targets.marginals["age_band"]["65+"] > 0.15
    assert targets.marginals["marital"]["married"] > 0.4
