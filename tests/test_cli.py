from __future__ import annotations

import pytest

from quorum.cli import main
from quorum.core.spec import SimulationSpec


def test_version(capsys):
    assert main(["version"]) == 0
    assert "quorum" in capsys.readouterr().out


def test_new_scaffolds_a_valid_spec(tmp_path, capsys):
    path = tmp_path / "sim.yaml"
    assert main(["new", "turnout", "--path", str(path)]) == 0
    spec = SimulationSpec.from_yaml(path)
    assert spec.name == "turnout"
    assert spec.scenario.question_id == "turnout"
    assert "wrote" in capsys.readouterr().out


def test_new_refuses_to_clobber(tmp_path, capsys):
    path = tmp_path / "sim.yaml"
    main(["new", "turnout", "--path", str(path)])
    assert main(["new", "turnout", "--path", str(path)]) == 1
    assert "--force" in capsys.readouterr().err
    assert main(["new", "turnout", "--path", str(path), "--force"]) == 0


def test_validate_summarizes_a_good_spec(tmp_path, capsys, spec):
    path = tmp_path / "sim.yaml"
    path.write_text(spec.to_yaml())
    assert main(["validate", str(path)]) == 0
    out = capsys.readouterr().out
    assert "valid" in out
    assert spec.fingerprint() in out


def test_validate_reports_field_errors(tmp_path, capsys):
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\nscenario:\n  question_id: q\n  options: [only_one]\n")
    assert main(["validate", str(path)]) == 1
    assert "scenario" in capsys.readouterr().err


def test_validate_reports_a_missing_file(capsys):
    assert main(["validate", "does-not-exist.yaml"]) == 1
    assert "could not read" in capsys.readouterr().err


def test_no_command_is_an_error():
    with pytest.raises(SystemExit):
        main([])
