"""A content-addressed cache of model answers.

The model is not a deterministic function of its input and sampling parameters are no
longer accepted, so a run cannot be made reproducible by pinning a seed. It is made
reproducible by remembering what the model said. Every answer is keyed by a hash of
everything that produced it (model, system prompt, user prompt, token limit), and the
cache is committed to the repository, so the demo replays real model output offline,
for free, and gives the same numbers on every machine.

Storage is one JSONL file per model. Append-only means a run adds lines rather than
rewriting a blob, which keeps the diff of a new experiment readable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from quorum.predict.provider import Completion


def cache_key(model: str, system: str, prompt: str, max_tokens: int) -> str:
    """Hash of everything that determined an answer."""
    payload = "\x1f".join([model, system, prompt, str(max_tokens)])
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(slots=True)
class ResponseCache:
    """Reads and writes model answers keyed by their inputs."""

    directory: Path
    entries: dict[str, Completion] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    _loaded: set[str] = field(default_factory=set)

    @classmethod
    def open(cls, directory: str | Path) -> "ResponseCache":
        return cls(directory=Path(directory))

    def _path(self, model: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in model)
        return self.directory / f"{safe}.jsonl"

    def load(self, model: str) -> int:
        """Read one model's cache file into memory. Returns the entry count."""
        if model in self._loaded:
            return len(self.entries)
        self._loaded.add(model)
        path = self._path(model)
        if not path.exists():
            return len(self.entries)
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            self.entries[record["key"]] = Completion.from_dict(record["completion"])
        return len(self.entries)

    def get(self, model: str, system: str, prompt: str, max_tokens: int) -> Completion | None:
        self.load(model)
        key = cache_key(model, system, prompt, max_tokens)
        found = self.entries.get(key)
        if found is None:
            self.misses += 1
            return None
        self.hits += 1
        return found

    def put(
        self, model: str, system: str, prompt: str, max_tokens: int, completion: Completion
    ) -> str:
        """Record an answer and append it to the model's cache file."""
        self.load(model)
        key = cache_key(model, system, prompt, max_tokens)
        self.entries[key] = completion
        self.directory.mkdir(parents=True, exist_ok=True)
        record = {"key": key, "completion": completion.as_dict()}
        with self._path(model).open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return key

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def __len__(self) -> int:
        return len(self.entries)


class NullCache:
    """A cache that remembers nothing. For tests and for deliberately fresh runs."""

    hits = 0
    misses = 0
    hit_rate = 0.0

    def get(self, model: str, system: str, prompt: str, max_tokens: int) -> Completion | None:
        return None

    def put(
        self, model: str, system: str, prompt: str, max_tokens: int, completion: Completion
    ) -> str:
        return ""

    def __len__(self) -> int:
        return 0
