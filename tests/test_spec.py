from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from quorum.core.spec import SimulationSpec


def test_spec_round_trips_through_yaml(spec):
    reloaded = SimulationSpec.from_dict(yaml.safe_load(spec.to_yaml()))
    assert reloaded.fingerprint() == spec.fingerprint()


def test_fingerprint_changes_with_any_field(spec_dict):
    base = SimulationSpec.from_dict(spec_dict)
    changed = SimulationSpec.from_dict({**spec_dict, "seed": spec_dict["seed"] + 1})
    assert base.fingerprint() != changed.fingerprint()


def test_fingerprint_is_stable_across_key_order(spec_dict):
    reordered = dict(reversed(list(spec_dict.items())))
    assert SimulationSpec.from_dict(reordered).fingerprint() == SimulationSpec.from_dict(spec_dict).fingerprint()


def test_unknown_keys_are_rejected(spec_dict):
    with pytest.raises(ValidationError):
        SimulationSpec.from_dict({**spec_dict, "populaton": {}})


def test_stratification_dimensions_must_exist(spec_dict):
    spec_dict["predictor"]["stratify_by"] = ["income_band"]
    with pytest.raises(ValidationError, match="income_band"):
        SimulationSpec.from_dict(spec_dict)


def test_estimator_dimensions_must_exist(spec_dict):
    spec_dict["estimator"]["dimensions"] = ["nonsense"]
    with pytest.raises(ValidationError, match="nonsense"):
        SimulationSpec.from_dict(spec_dict)


def test_duplicate_attributes_are_rejected(spec_dict):
    spec_dict["population"]["attributes"] = ["age_band", "age_band"]
    with pytest.raises(ValidationError, match="duplicates"):
        SimulationSpec.from_dict(spec_dict)


def test_empty_attributes_are_rejected(spec_dict):
    spec_dict["population"]["attributes"] = []
    with pytest.raises(ValidationError, match="at least one dimension"):
        SimulationSpec.from_dict(spec_dict)


def test_scenario_needs_two_options(spec_dict):
    spec_dict["scenario"]["options"] = ["Agree"]
    with pytest.raises(ValidationError, match="at least two"):
        SimulationSpec.from_dict(spec_dict)


def test_scenario_rejects_duplicate_options(spec_dict):
    spec_dict["scenario"]["options"] = ["Agree", "Agree"]
    with pytest.raises(ValidationError, match="duplicates"):
        SimulationSpec.from_dict(spec_dict)


def test_scenario_needs_a_prompt_or_an_arm(spec_dict):
    spec_dict["scenario"]["prompt"] = ""
    with pytest.raises(ValidationError, match="either a prompt"):
        SimulationSpec.from_dict(spec_dict)


def test_scenario_rejects_duplicate_arm_ids(spec_dict):
    spec_dict["scenario"]["arms"] = [
        {"id": "a", "label": "A", "prompt": "one"},
        {"id": "a", "label": "B", "prompt": "two"},
    ]
    with pytest.raises(ValidationError, match="duplicate ids"):
        SimulationSpec.from_dict(spec_dict)


def test_arm_prompts_defaults_to_a_single_unframed_arm(spec):
    assert spec.scenario.arm_prompts() == {"default": "Do you agree?"}


def test_arm_prompts_uses_arm_ids_when_present(spec_dict):
    spec_dict["scenario"]["arms"] = [
        {"id": "welfare", "label": "welfare", "prompt": "spending on welfare"},
        {"id": "poor", "label": "the poor", "prompt": "assistance to the poor"},
    ]
    prompts = SimulationSpec.from_dict(spec_dict).scenario.arm_prompts()
    assert set(prompts) == {"welfare", "poor"}


def test_spec_is_frozen(spec):
    with pytest.raises(ValidationError):
        spec.name = "renamed"


def test_from_yaml_reads_a_file(tmp_path, spec):
    path = tmp_path / "sim.yaml"
    path.write_text(spec.to_yaml())
    assert SimulationSpec.from_yaml(path).name == spec.name


def test_from_yaml_rejects_a_non_mapping(tmp_path):
    path = tmp_path / "sim.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        SimulationSpec.from_yaml(path)
