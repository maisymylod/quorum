"""Command line entry point.

The commands are the workflow: scaffold a spec, check it, run it, score it, publish it.
"""

from __future__ import annotations

import argparse
import sys
from importlib import resources
from pathlib import Path

from quorum import __version__


def _cmd_version(_: argparse.Namespace) -> int:
    print(f"quorum {__version__}")
    return 0


def _cmd_new(args: argparse.Namespace) -> int:
    """Scaffold a new simulation spec.

    The point of the spec being a file rather than code is that standing up a new
    simulation is a copy-and-edit, which is what makes this something a non-author can
    do without opening the engine.
    """
    template = resources.files("quorum.templates").joinpath("simulation.yaml").read_text()
    destination = Path(args.path or f"{args.name}.yaml")
    if destination.exists() and not args.force:
        print(f"refusing to overwrite {destination} (pass --force)", file=sys.stderr)
        return 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template.replace("{name}", args.name))
    print(f"wrote {destination}")
    print(f"next: edit the scenario block, then `quorum validate {destination}`")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from pydantic import ValidationError

    from quorum.core.spec import SimulationSpec

    try:
        spec = SimulationSpec.from_yaml(args.spec)
    except ValidationError as exc:
        print(f"{args.spec} is not a valid simulation spec:\n", file=sys.stderr)
        for error in exc.errors():
            location = ".".join(str(p) for p in error["loc"])
            print(f"  {location}: {error['msg']}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"could not read {args.spec}: {exc}", file=sys.stderr)
        return 1

    arms = spec.scenario.arm_prompts()
    print(f"{args.spec}: valid")
    print(f"  name          {spec.name}")
    print(f"  fingerprint   {spec.fingerprint()}")
    print(f"  seed          {spec.seed}")
    print(f"  population    {spec.population.size:,} agents over {len(spec.population.attributes)} attributes")
    print(f"  question      {spec.scenario.question_id} ({len(spec.scenario.options)} options, {len(arms)} arm(s))")
    print(f"  predictor     {spec.predictor.kind} via {spec.predictor.provider.name}:{spec.predictor.provider.model}")
    print(f"  budget        ${spec.budget.max_usd:.2f} / {spec.budget.max_calls:,} calls")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quorum",
        description="Population simulation engine: synthesize an audience, ask it a "
        "question, and score the answer against known truth.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_version = subparsers.add_parser("version", help="print the installed version")
    p_version.set_defaults(func=_cmd_version)

    p_new = subparsers.add_parser("new", help="scaffold a new simulation spec")
    p_new.add_argument("name", help="simulation name, used for the file and question id")
    p_new.add_argument("--path", help="destination path (default: <name>.yaml)")
    p_new.add_argument("--force", action="store_true", help="overwrite an existing file")
    p_new.set_defaults(func=_cmd_new)

    p_validate = subparsers.add_parser("validate", help="check a spec and summarize it")
    p_validate.add_argument("spec", help="path to a simulation spec")
    p_validate.set_defaults(func=_cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
