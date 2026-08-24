"""Publication: raw agent output turned into a decision-ready artifact."""

from quorum.publish.contrast import Contrast
from quorum.publish.report import publish, render_html, render_markdown

__all__ = ["Contrast", "publish", "render_html", "render_markdown"]
