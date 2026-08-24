# Response cache

Real model answers, keyed by a hash of everything that produced them: model, system
prompt, user prompt and token limit. Committed on purpose.

Current models reject sampling parameters, so a run cannot be made reproducible by
pinning a seed or a temperature. It is made reproducible by remembering what the model
said. With these files present, `make demo` replays real model output offline, for
free, and gives the same numbers on every machine.

One JSONL file per model, append-only, so a new experiment shows up as added lines
rather than a rewritten blob.

The offline stub is not cached. It is already a deterministic function of its prompt,
so persisting it would store a hash and nothing else.
