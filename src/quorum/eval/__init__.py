"""Evaluation: whether any of this works, measured against published answers."""

from quorum.eval import metrics, report
from quorum.eval.configurations import DEFAULT_GRID, ENGINES, build_spec
from quorum.eval.harness import Backtest, BacktestResult, ExperimentScore, QuestionScore

__all__ = [
    "Backtest",
    "BacktestResult",
    "DEFAULT_GRID",
    "ENGINES",
    "ExperimentScore",
    "QuestionScore",
    "build_spec",
    "metrics",
    "report",
]
