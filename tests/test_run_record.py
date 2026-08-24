from __future__ import annotations

import json

import numpy as np

from quorum.core.prediction import Prediction
from quorum.core.run import RunRecord


def test_record_captures_the_spec(spec):
    record = RunRecord.for_spec(spec)
    assert record.name == spec.name
    assert record.spec_fingerprint == spec.fingerprint()
    assert record.seed == spec.seed
    assert record.predictor == spec.predictor.kind
    assert record.quorum_version


def test_reproducibility_key_binds_spec_seed_and_population(spec):
    record = RunRecord.for_spec(spec)
    record.population_fingerprint = "abc123"
    assert record.reproducibility_key() == f"{spec.fingerprint()}:{spec.seed}:abc123"


def test_add_prediction_serializes_interval_and_drops_unserializable_metadata(spec):
    record = RunRecord.for_spec(spec)
    draws = np.random.default_rng(1).dirichlet([10, 10], size=200)
    prediction = Prediction(
        "q1",
        ("Agree", "Disagree"),
        draws.mean(axis=0),
        draws=draws,
        metadata={"archetypes": 12, "array": np.zeros(3)},
    )
    record.add_prediction("default", prediction)
    payload = record.results["default"]
    assert payload["options"] == ["Agree", "Disagree"]
    assert payload["has_uncertainty"] is True
    assert len(payload["interval"]) == 2
    assert payload["metadata"] == {"archetypes": 12}


def test_record_writes_valid_json(tmp_path, spec):
    record = RunRecord.for_spec(spec)
    record.note("stub provider in use")
    path = record.write(tmp_path)
    payload = json.loads(path.read_text())
    assert payload["notes"] == ["stub provider in use"]
    assert payload["name"] == spec.name
