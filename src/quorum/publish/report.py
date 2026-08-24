"""Turning a run into something a person can decide from.

Raw agent output is a matrix. What a reader needs is the answer, how firmly it is
held, who inside the population differs from it, what changing the wording did, and
enough provenance to know what they are looking at. This module builds exactly that,
as one self-contained HTML file that opens anywhere with no network, plus a markdown
version for anything that reads text.

The provenance is not a footnote. Every artifact states which provider produced the
answers and separates what is real (the census marginals, the survey wording, the
model's answers) from what is constructed (the agents themselves, and any arm the
survey never ran). An artifact that looks authoritative without saying where its
numbers came from is worse than no artifact.
"""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path

import numpy as np

from quorum.exec.runner import SimulationResult
from quorum.publish.charts import interval_plot, legend, stacked_bar
from quorum.publish.contrast import Contrast

STYLE = """
:root { --ink:#1f2933; --muted:#5b6470; --rule:#e3e7eb; --bg:#ffffff; --panel:#f7f9fa; }
* { box-sizing:border-box; }
body { margin:0; padding:40px 24px 72px; background:var(--bg); color:var(--ink);
  font:16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width:820px; margin:0 auto; }
h1 { font-size:26px; line-height:1.25; margin:0 0 6px; letter-spacing:-0.01em; }
h2 { font-size:15px; text-transform:uppercase; letter-spacing:0.08em; color:var(--muted);
  margin:44px 0 14px; font-weight:600; }
.question { font-size:18px; color:var(--ink); margin:0 0 4px; }
.sub { color:var(--muted); font-size:14px; margin:0; }
.arm { border:1px solid var(--rule); border-radius:10px; padding:18px 20px; margin:0 0 14px;
  background:var(--panel); }
.arm h3 { margin:0 0 4px; font-size:16px; font-weight:600; }
.arm p.wording { margin:0 0 14px; color:var(--muted); font-size:14px; }
table { border-collapse:collapse; width:100%; font-size:14px; }
th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--rule); }
th { color:var(--muted); font-weight:600; font-size:13px; text-transform:uppercase;
  letter-spacing:0.04em; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
.legend { display:flex; flex-wrap:wrap; gap:16px; margin:10px 0 0; font-size:13px;
  color:var(--muted); }
.key { display:inline-flex; align-items:center; gap:6px; }
.key i { width:11px; height:11px; border-radius:2px; display:inline-block; }
.callout { border-left:3px solid #bc4b51; background:#fdf3f3; padding:12px 16px;
  border-radius:0 8px 8px 0; font-size:14px; margin:0 0 24px; }
.provenance { margin-top:12px; font-size:13px; color:var(--muted); }
.provenance dt { font-weight:600; color:var(--ink); float:left; width:170px; clear:left; }
.provenance dd { margin:0 0 6px 180px; }
.scroll { overflow-x:auto; }
footer { margin-top:52px; padding-top:18px; border-top:1px solid var(--rule);
  color:var(--muted); font-size:13px; }
code { font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:13px; }
"""

STUB_CALLOUT = (
    "These answers came from the offline stub, which hashes the prompt into a "
    "distribution and knows nothing about the world. Nothing on this page is a "
    "prediction about anyone."
)


def contrasts(result: SimulationResult) -> list[Contrast]:
    """Every arm compared against the first, which is the survey's own reference form."""
    arms = list(result.predictions)
    if len(arms) < 2:
        return []
    reference = arms[0]
    return [
        Contrast.between(reference, result.predictions[reference], arm, result.predictions[arm])
        for arm in arms[1:]
    ]


def _arm_section(result: SimulationResult, arm: str) -> str:
    prediction = result.predictions[arm]
    level = result.spec.estimator.level
    scenario = next(a for a in result.spec.scenario.arms if a.id == arm)

    rows = ["<tr><th>response</th><th class='num'>share</th>"]
    rows.append(
        f"<th class='num'>{int(level * 100)}% interval</th></tr>"
        if prediction.has_uncertainty
        else "</tr>"
    )
    interval = prediction.interval(level)
    for index, option in enumerate(prediction.options):
        cells = [f"<td>{escape(option)}</td>", f"<td class='num'>{prediction.distribution[index]:.1%}</td>"]
        if prediction.has_uncertainty:
            lo, hi = interval[index]
            cells.append(f"<td class='num'>{lo:.1%} to {hi:.1%}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return f"""<section class="arm">
  <h3>{escape(scenario.label or arm)}</h3>
  <p class="wording">{escape(scenario.prompt)}</p>
  {stacked_bar(prediction.distribution, prediction.options)}
  {legend(prediction.options)}
  <div class="scroll"><table>{''.join(rows)}</table></div>
</section>"""


def _contrast_section(result: SimulationResult) -> str:
    found = contrasts(result)
    if not found:
        return ""
    level = result.spec.estimator.level
    blocks = []
    for contrast in found:
        interval = contrast.interval(level)
        rows = [
            "<tr><th>response</th><th class='num'>shift</th>"
            f"<th class='num'>{int(level * 100)}% interval</th><th>resolved</th></tr>"
        ]
        for index, option in enumerate(contrast.options):
            lo, hi = interval[index]
            resolved = "yes" if contrast.resolves(option, level) else "no"
            rows.append(
                f"<tr><td>{escape(option)}</td>"
                f"<td class='num'>{contrast.difference[index]:+.1%}</td>"
                f"<td class='num'>{lo:+.1%} to {hi:+.1%}</td><td>{resolved}</td></tr>"
            )
        blocks.append(
            f"""<section class="arm">
  <h3>{escape(contrast.other)} against {escape(contrast.reference)}</h3>
  {interval_plot(contrast.options, contrast.difference, interval)}
  <div class="scroll"><table>{''.join(rows)}</table></div>
</section>"""
        )
    return "\n".join(blocks)


def _segment_section(result: SimulationResult, arm: str) -> str:
    prediction = result.predictions[arm]
    if not prediction.segments:
        return ""
    blocks = []
    for dimension, levels in prediction.segments.items():
        header = "".join(f"<th class='num'>{escape(o)}</th>" for o in prediction.options)
        rows = [f"<tr><th>{escape(dimension.replace('_', ' '))}</th>{header}</tr>"]
        for level, values in levels.items():
            cells = "".join(f"<td class='num'>{v:.1%}</td>" for v in np.asarray(values))
            rows.append(f"<tr><td>{escape(level)}</td>{cells}</tr>")
        blocks.append(f"<div class='scroll'><table>{''.join(rows)}</table></div>")
    return "\n".join(blocks)


def _provenance(result: SimulationResult) -> str:
    record = result.record
    spec = result.spec
    real = [
        "population marginals from published census microdata",
        "question wording taken verbatim from the survey codebook",
    ]
    constructed = [
        f"{len(result.population):,} agents, synthesized and raked to those marginals",
        "latent traits, which disperse agents within a demographic cell",
    ]
    if spec.world.dynamics.enabled:
        constructed.append("a social graph and peer influence over it")
    if record.provider == "stub":
        constructed.append("the answers themselves, which came from the offline stub")
    else:
        real.append(f"answers from {record.model}, cached so this run replays exactly")

    items = {
        "Provider": f"<code>{escape(record.provider)}</code> / <code>{escape(record.model)}</code>",
        "Spec fingerprint": f"<code>{escape(record.spec_fingerprint)}</code>",
        "Population": f"<code>{escape(record.population_fingerprint)}</code>, "
        f"{record.population_size:,} agents, seed {record.seed}",
        "Marginal fidelity": f"{result.fidelity.max_deviation:.1e} worst-case share error",
        "Cost": f"${record.cost_usd:.4f} over {record.llm_calls:,} model calls "
        f"({record.cache_hits:,} replayed from cache)",
        "Wall time": f"{record.wall_seconds:.1f}s",
        "Real": "; ".join(real),
        "Constructed": "; ".join(constructed),
    }
    body = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in items.items())
    return f'<dl class="provenance">{body}</dl>'


def render_html(result: SimulationResult) -> str:
    """The decision-ready artifact, as one self-contained file."""
    spec = result.spec
    arms = list(result.predictions)
    stamped = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    callout = f'<p class="callout">{STUB_CALLOUT}</p>' if result.record.provider == "stub" else ""

    sections = [_arm_section(result, arm) for arm in arms]
    contrast_block = _contrast_section(result)
    segments = _segment_section(result, arms[0])

    body = [
        f"<h1>{escape(spec.scenario.question_id)}</h1>",
        f'<p class="sub">{len(result.population):,} simulated respondents, '
        f"{len(arms)} question form{'s' if len(arms) != 1 else ''}, {stamped}</p>",
        callout,
        "<h2>What the population said</h2>",
        *sections,
    ]
    if contrast_block:
        body += [
            "<h2>What changing the wording did</h2>",
            '<p class="sub">Both forms were put to the same population, so any '
            "difference between them is caused by the wording. "
            '"Resolved" means the interval excludes zero, which is a statement about '
            "what the simulation could distinguish, not about the world.</p>",
            contrast_block,
        ]
    if segments:
        body += [
            "<h2>Who differs from the average</h2>",
            f'<p class="sub">Breakdown of <code>{escape(arms[0])}</code>. '
            "Segment estimates are pooled toward the overall answer in proportion to "
            "how little evidence each cell has, so a thin cell reads close to the "
            "average by construction rather than by finding.</p>",
            segments,
        ]
    body += ["<h2>Provenance</h2>", _provenance(result)]
    body += [
        "<footer>Generated by quorum. Every figure on this page comes from running "
        "the simulation described above; nothing here was entered by hand.</footer>"
    ]

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(spec.name)}</title>
<style>{STYLE}</style></head>
<body><main>{''.join(body)}</main></body></html>
"""


def render_markdown(result: SimulationResult) -> str:
    """The same content as text, for anything that does not render HTML."""
    spec = result.spec
    level = spec.estimator.level
    lines = [f"# {spec.scenario.question_id}", ""]
    if result.record.provider == "stub":
        lines += [f"> {STUB_CALLOUT}", ""]
    lines += [
        f"{len(result.population):,} simulated respondents, "
        f"{len(result.predictions)} question form(s).",
        "",
    ]
    for arm, prediction in result.predictions.items():
        scenario = next(a for a in spec.scenario.arms if a.id == arm)
        lines += [f"## {scenario.label or arm}", "", f"_{scenario.prompt}_", ""]
        header = "| response | share |"
        divider = "|---|---|"
        if prediction.has_uncertainty:
            header += f" {int(level * 100)}% interval |"
            divider += "---|"
        lines += [header, divider]
        interval = prediction.interval(level)
        for index, option in enumerate(prediction.options):
            row = f"| {option} | {prediction.distribution[index]:.1%} |"
            if prediction.has_uncertainty:
                row += f" {interval[index][0]:.1%} to {interval[index][1]:.1%} |"
            lines.append(row)
        lines.append("")

    for contrast in contrasts(result):
        lines += [f"## {contrast.other} against {contrast.reference}", ""]
        lines += ["| response | shift | resolved |", "|---|---|---|"]
        for index, option in enumerate(contrast.options):
            resolved = "yes" if contrast.resolves(option, level) else "no"
            lines.append(f"| {option} | {contrast.difference[index]:+.1%} | {resolved} |")
        lines.append("")

    record = result.record
    lines += [
        "## Provenance",
        "",
        f"- Provider `{record.provider}` / `{record.model}`",
        f"- Spec `{record.spec_fingerprint}`, population `{record.population_fingerprint}`, "
        f"seed {record.seed}",
        f"- Marginal fidelity {result.fidelity.max_deviation:.1e}",
        f"- ${record.cost_usd:.4f} over {record.llm_calls:,} calls, "
        f"{record.wall_seconds:.1f}s",
        "",
    ]
    return "\n".join(lines)


def publish(result: SimulationResult, destination: str | Path) -> dict[str, Path]:
    """Write every requested format and return where each landed."""
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    formats = result.spec.output.formats
    if "html" in formats:
        path = out / "report.html"
        path.write_text(render_html(result))
        written["html"] = path
    if "markdown" in formats:
        path = out / "report.md"
        path.write_text(render_markdown(result))
        written["markdown"] = path
    if "json" in formats:
        written["json"] = result.record.write(out)
    return written
