"""Small inline SVG charts.

Hand-rolled rather than plotted with a library, for one reason that matters more than
elegance: a published artifact has to be a single file that opens anywhere, with no
network, no fonts to fetch and no image assets beside it. Inline SVG gives that, and
it keeps a plotting library out of the dependency set and out of CI.

Two chart types, because two are enough. A stacked bar shows one distribution; a
dot-and-interval plot shows an estimate with what is known about it, which is the
shape almost everything here takes.
"""

from __future__ import annotations

from html import escape

import numpy as np

#: Sequential, colour-blind safe, and ordered, because the response options are.
PALETTE = ("#1b4965", "#5fa8d3", "#bee9e8", "#f4a259", "#bc4b51", "#8a6fa8", "#7d8491")

_TEXT = "#1f2933"
_MUTED = "#5b6470"
_RULE = "#d7dce2"


def _colour(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def stacked_bar(
    values: np.ndarray, labels: tuple[str, ...], width: int = 620, height: int = 34
) -> str:
    """One distribution as a single stacked bar, with in-bar percentages."""
    values = np.asarray(values, dtype=float)
    parts = []
    x = 0.0
    for index, (value, label) in enumerate(zip(values, labels)):
        span = float(value) * width
        parts.append(
            f'<rect x="{x:.2f}" y="0" width="{max(span, 0):.2f}" height="{height}" '
            f'fill="{_colour(index)}"><title>{escape(label)}: {value:.1%}</title></rect>'
        )
        if span > 46:
            parts.append(
                f'<text x="{x + span / 2:.2f}" y="{height / 2 + 4:.1f}" '
                f'text-anchor="middle" font-size="12" font-weight="600" '
                f'fill="{"#ffffff" if index % len(PALETTE) < 2 else _TEXT}">'
                f"{value:.0%}</text>"
            )
        x += span
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'role="img" aria-label="response distribution">{"".join(parts)}</svg>'
    )


def interval_plot(
    labels: tuple[str, ...],
    points: np.ndarray,
    intervals: np.ndarray,
    width: int = 620,
    row_height: int = 30,
    domain: tuple[float, float] | None = None,
) -> str:
    """Estimates with their intervals, one row each.

    The zero line is drawn whenever the domain spans it, because for a contrast that
    line is the whole question.
    """
    points = np.asarray(points, dtype=float)
    intervals = np.asarray(intervals, dtype=float)
    low = float(min(intervals[:, 0].min(), points.min()))
    high = float(max(intervals[:, 1].max(), points.max()))
    if domain is not None:
        low, high = domain
    if high - low < 1e-9:
        low, high = low - 0.05, high + 0.05
    pad = 0.08 * (high - low)
    low, high = low - pad, high + pad

    left = 150
    plot_width = width - left - 60
    height = row_height * len(labels) + 26

    def x_of(value: float) -> float:
        return left + (value - low) / (high - low) * plot_width

    parts = []
    if low < 0 < high:
        parts.append(
            f'<line x1="{x_of(0):.1f}" y1="6" x2="{x_of(0):.1f}" y2="{height - 20}" '
            f'stroke="{_RULE}" stroke-width="1" stroke-dasharray="3 3"/>'
        )
    for index, label in enumerate(labels):
        y = 18 + index * row_height
        lo, hi = intervals[index]
        parts.append(
            f'<text x="{left - 12}" y="{y + 4}" text-anchor="end" font-size="13" '
            f'fill="{_TEXT}">{escape(label)}</text>'
        )
        parts.append(
            f'<line x1="{x_of(lo):.1f}" y1="{y}" x2="{x_of(hi):.1f}" y2="{y}" '
            f'stroke="{_colour(index)}" stroke-width="3" stroke-linecap="round" '
            f'opacity="0.45"/>'
        )
        parts.append(
            f'<circle cx="{x_of(points[index]):.1f}" cy="{y}" r="5" '
            f'fill="{_colour(index)}"/>'
        )
        parts.append(
            f'<text x="{x_of(hi) + 10:.1f}" y="{y + 4}" font-size="12" '
            f'fill="{_MUTED}">{points[index]:+.1%}</text>'
            if low < 0 < high
            else f'<text x="{x_of(hi) + 10:.1f}" y="{y + 4}" font-size="12" '
            f'fill="{_MUTED}">{points[index]:.1%}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'role="img" aria-label="estimates with intervals">{"".join(parts)}</svg>'
    )


def legend(labels: tuple[str, ...]) -> str:
    items = "".join(
        f'<span class="key"><i style="background:{_colour(i)}"></i>{escape(label)}</span>'
        for i, label in enumerate(labels)
    )
    return f'<div class="legend">{items}</div>'
