from __future__ import annotations

import argparse
import html
from pathlib import Path


def _markdownish_to_html(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    in_list = False
    in_code = False
    code_lines: list[str] = []

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                out.append("<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>")
                code_lines = []
                in_code = False
            else:
                close_list()
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not stripped:
            close_list()
            continue
        if stripped.startswith("#"):
            close_list()
            level = min(len(stripped) - len(stripped.lstrip("#")), 3)
            body = stripped[level:].strip()
            out.append(f"<h{level}>{html.escape(body)}</h{level}>")
        elif stripped.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append("<li>" + html.escape(stripped[2:]) + "</li>")
        else:
            close_list()
            out.append("<p>" + html.escape(stripped) + "</p>")
    close_list()
    if in_code:
        out.append("<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--docs", default=None, help="Optional explicit markdown file path.")
    parser.add_argument("--out-dir", default="artifacts/site")
    parser.add_argument("--out", default=None, help="Optional explicit HTML output path.")
    args = parser.parse_args()
    index = Path(args.docs) if args.docs else Path(args.docs_dir) / "index.md"
    out_path = Path(args.out) if args.out else Path(args.out_dir) / "index.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = index.read_text(encoding="utf-8") if index.exists() else "# Missing docs/index.md"
    html_doc = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>Deposit Duration Gap Docs</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:980px;margin:2rem auto;line-height:1.5}"
        "h1,h2,h3{line-height:1.2} code,pre{background:#f6f8fa;border-radius:6px}"
        "pre{padding:1rem;overflow:auto} table{border-collapse:collapse}"
        "p,li{font-size:1rem}</style></head><body>"
        + _markdownish_to_html(text)
        + "</body></html>"
    )
    out_path.write_text(html_doc, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
