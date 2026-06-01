from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="data_manifest/full_live_public_summary.json")
    parser.add_argument("--out", default="artifacts/reports/paper.html")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    feature = summary.get("feature_store", {})
    visual = summary.get("visual_audit", {})
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Deposit Franchise and Duration Gap</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem auto; max-width: 980px; line-height: 1.45; }}
    code, pre {{ background: #f5f5f5; padding: 0.1rem 0.25rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 0.35rem; text-align: left; }}
  </style>
</head>
<body>
  <h1>Deposit Franchise and Duration Gap in U.S. Bank Equities</h1>
  <p>This is a reproducible report scaffold generated from public-safe manifests.
  Full manuscript prose is intentionally kept out of this build step.</p>
  <h2>Frozen Sample</h2>
  <ul>
    <li>Rows: {html.escape(str(feature.get("rows", "n/a")))}</li>
    <li>PERMNOs: {html.escape(str(feature.get("permnos", "n/a")))}</li>
    <li>Months: {html.escape(str(feature.get("month_min", "n/a")))} to {html.escape(str(feature.get("month_max", "n/a")))}</li>
    <li>Public-information lag: {html.escape(str(feature.get("lag_days", "n/a")))} days</li>
    <li>Visual audit: {html.escape(str(visual.get("passed", "n/a")))} / {html.escape(str(visual.get("figures", "n/a")))} passed</li>
  </ul>
  <h2>Manifest Snapshot</h2>
  <pre>{html.escape(json.dumps(summary, indent=2)[:20000])}</pre>
</body>
</html>
"""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
