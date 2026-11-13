"""HTML results view — a readable run report with no frontend build.

``render_run_html`` turns an ``EvalRun`` into a single self-contained
HTML page (inline CSS, no JavaScript, no React): suite, mean score,
pass counts and a per-case table with every scorer's score and detail.
It is what ``GET /runs/{id}/html`` serves and what the CLI ``report``
command writes to a file.
"""

from __future__ import annotations

import html

from .models import EvalRun


def render_run_html(run: EvalRun) -> str:
    rows = []
    for result in run.results:
        mark = "PASS" if result.passed else "FAIL"
        scorer_lines = "<br>".join(
            f"{html.escape(s.scorer_type)}: {s.score:.2f} — {html.escape(s.detail)}"
            for s in result.scores
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(result.case_name)}</td>"
            f"<td class=\"{'pass' if result.passed else 'fail'}\">{mark}</td>"
            f"<td>{result.mean_score:.2f}</td>"
            f"<td>{html.escape(result.output)}</td>"
            f"<td>{scorer_lines}</td>"
            f"<td>{result.latency_ms:.0f} ms</td>"
            "</tr>"
        )
    body = "\n".join(rows) if rows else "<tr><td colspan=\"6\">No cases in this run.</td></tr>"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Eval run — {html.escape(run.suite_name)}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }}
  h1 {{ font-size: 1.4rem; }}
  .summary {{ margin-bottom: 1rem; }}
  .pass {{ color: #0a7d2c; font-weight: 600; }}
  .fail {{ color: #b3261e; font-weight: 600; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 0.5rem; text-align: left; vertical-align: top; }}
  th {{ background: #f2f2f2; }}
</style>
</head>
<body>
<h1>Eval run — {html.escape(run.suite_name)}</h1>
<p class="summary">Model: <strong>{html.escape(run.model)}</strong> &middot;
Run: <code>{html.escape(run.id)}</code> &middot;
{run.passed_count}/{run.total_count} cases passed &middot;
mean score <strong>{run.mean_score:.2f}</strong> &middot;
started {html.escape(run.started_at.isoformat())}</p>
<table>
<tr><th>Case</th><th>Result</th><th>Score</th><th>Output</th><th>Scorers</th><th>Latency</th></tr>
{body}
</table>
</body>
</html>
"""
