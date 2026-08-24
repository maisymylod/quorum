"""The run record: everything needed to explain or reproduce a result.

A simulated number is only worth anything if you can say exactly what produced it.
Every run writes one of these next to its artifacts: the spec hash, the seed, the
population fingerprint, the provider and model actually used, what it cost, how long
it took, and the resulting prediction. Two runs that agree on ``spec_fingerprint``,
``seed`` and ``population_fingerprint`` must agree on the answer, and the determinism
test in CI asserts exactly that.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from quorum.core.prediction import Prediction
from quorum.core.spec import SimulationSpec


@dataclass(slots=True)
class RunRecord:
    """Provenance for one simulation run."""

    name: str
    spec_fingerprint: str
    seed: int
    population_fingerprint: str = ""
    population_size: int = 0
    predictor: str = ""
    provider: str = ""
    model: str = ""
    llm_calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    wall_seconds: float = 0.0
    quorum_version: str = ""
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    results: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @classmethod
    def for_spec(cls, spec: SimulationSpec) -> "RunRecord":
        from quorum import __version__

        return cls(
            name=spec.name,
            spec_fingerprint=spec.fingerprint(),
            seed=spec.seed,
            predictor=spec.predictor.kind,
            provider=spec.predictor.provider.name,
            model=spec.predictor.provider.model,
            quorum_version=__version__,
        )

    def add_prediction(self, arm: str, prediction: Prediction, level: float = 0.90) -> None:
        interval = prediction.interval(level)
        self.results[arm] = {
            "question_id": prediction.question_id,
            "options": list(prediction.options),
            "distribution": [round(float(p), 6) for p in prediction.distribution],
            "interval": [[round(float(lo), 6), round(float(hi), 6)] for lo, hi in interval],
            "level": level,
            "has_uncertainty": prediction.has_uncertainty,
            "metadata": {k: v for k, v in prediction.metadata.items() if _jsonable(v)},
        }

    def note(self, message: str) -> None:
        self.notes.append(message)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def write(self, directory: str | Path, filename: str = "run.json") -> Path:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        path = out / filename
        path.write_text(self.to_json() + "\n")
        return path

    def reproducibility_key(self) -> str:
        """The triple that must determine the answer."""
        return f"{self.spec_fingerprint}:{self.seed}:{self.population_fingerprint}"


def _jsonable(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True
