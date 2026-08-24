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


def _cmd_run(args: argparse.Namespace) -> int:
    """Run one simulation spec and write its artifacts."""
    from quorum.core.spec import SimulationSpec
    from quorum.exec.runner import Simulation

    spec = SimulationSpec.from_yaml(args.spec)
    result = Simulation(spec, root=args.root).run()
    destination = Path(args.root) / spec.output.dir / spec.name.replace(":", "_")
    path = result.record.write(destination)

    print(f"{spec.name}  ({result.record.reproducibility_key()})")
    print(f"  population   {len(result.population):,} agents, "
          f"marginal fidelity {result.fidelity.max_deviation:.1e}")
    for arm, prediction in result.predictions.items():
        shares = ", ".join(
            f"{o} {v:.1%}" for o, v in zip(prediction.options, prediction.distribution)
        )
        print(f"  {arm:<12} {shares}")
        if prediction.has_uncertainty:
            interval = prediction.interval(spec.estimator.level)
            band = ", ".join(f"[{lo:.1%}, {hi:.1%}]" for lo, hi in interval)
            print(f"  {'':<12} {int(spec.estimator.level * 100)}%: {band}")
    print(f"  cost         ${result.record.cost_usd:.4f} over "
          f"{result.record.llm_calls:,} calls in {result.record.wall_seconds:.1f}s")
    for note in result.record.notes:
        print(f"  note         {note}")
    print(f"  wrote        {path}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    """Run the ablation grid over the question bank and regenerate the report."""
    import json

    from quorum.data.targets import MarginalTargets, QuestionBank
    from quorum.eval import report
    from quorum.eval.configurations import BASE_SPEC, DEFAULT_GRID, ENGINES, build_spec
    from quorum.eval.harness import Backtest

    root = Path(args.root)
    bank = QuestionBank.load(root / args.questions)
    targets = MarginalTargets.load(root / BASE_SPEC["population"]["targets"])
    engines = args.engines or list(DEFAULT_GRID)
    unknown = [e for e in engines if e not in ENGINES]
    if unknown:
        print(f"unknown engines {unknown}; known: {sorted(ENGINES)}", file=sys.stderr)
        return 1

    provider = {"name": args.provider, "model": args.model}
    factory = lambda engine, group, arms: build_spec(engine, group, arms, provider=provider)
    backtest = Backtest(bank, factory, root=str(root), targets=targets)

    results = {}
    for engine in engines:
        result = backtest.run(engine, only=args.only)
        results[engine] = result
        summary = result.summary()
        print(
            f"{engine:<30} mae {summary['mae']:.4f}  "
            f"gap_mae {summary.get('gap_mae', float('nan')):.4f}  "
            f"${summary['cost_usd']:.4f}  {int(summary['llm_calls']):,} calls"
        )

    text = report.render(
        results,
        bank_year=bank.year,
        provider=args.provider,
        model=args.model,
        population_size=BASE_SPEC["population"]["size"],
        archetypes=BASE_SPEC["predictor"]["archetypes"],
    )
    (root / "EVAL.md").write_text(text)
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "eval.json").write_text(
        json.dumps({name: r.summary() for name, r in results.items()}, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {root / 'EVAL.md'} and {artifacts / 'eval.json'}")
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

    p_run = subparsers.add_parser("run", help="run one simulation spec")
    p_run.add_argument("spec", help="path to a simulation spec")
    p_run.add_argument("--root", default=".", help="repository root for data paths")
    p_run.set_defaults(func=_cmd_run)

    p_eval = subparsers.add_parser(
        "eval", help="run the ablation grid over the question bank and write EVAL.md"
    )
    p_eval.add_argument("--root", default=".", help="repository root for data paths")
    p_eval.add_argument(
        "--questions", default="data/vendor/gss_questions.json", help="question bank"
    )
    p_eval.add_argument("--provider", default="stub", choices=["stub", "anthropic"])
    p_eval.add_argument("--model", default="stub", help="model id for a live provider")
    p_eval.add_argument("--engines", nargs="*", help="subset of the grid to run")
    p_eval.add_argument("--only", nargs="*", help="subset of question groups to score")
    p_eval.set_defaults(func=_cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
