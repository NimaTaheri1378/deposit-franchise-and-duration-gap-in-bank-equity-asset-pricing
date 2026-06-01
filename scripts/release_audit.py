from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_EXTENSIONS = {
    ".cff",
    ".css",
    ".example",
    ".html",
    ".json",
    ".md",
    ".qmd",
    ".sbatch",
    ".sh",
    ".sql",
    ".toml",
    ".txt",
    ".yml",
    ".yaml",
    ".py",
}
FORBIDDEN_EXTENSIONS = {
    ".feather",
    ".h5",
    ".hdf5",
    ".joblib",
    ".onnx",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
}
FORBIDDEN_PREFIXES = ("artifacts/", "cache/", "data/", "logs/", "processed/", "raw/")
REQUIRED_README_IMAGES = (
    "figures/static/hero_figure.png",
    "figures/static/duration_uninsured_contour_proxy.png",
    "figures/static/march_2023_event_car.png",
    "figures/static/strategy_cumulative_return.png",
    "figures/static/prediction_decile_returns.png",
    "figures/static/fragility_duration_heatmap.png",
    "figures/static/interpretability_permutation_importance.png",
)
SECRET_PATTERNS = (
    (
        "api_or_secret_assignment",
        re.compile(
            r"(?i)\b(api[_-]?key|access[_-]?key|secret[_-]?key|client[_-]?secret|"
            r"password|passwd|token)\b\s*[:=]\s*[^\s\"']{8,}"
        ),
    ),
    ("email", re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")),
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("long_hex", re.compile(r"\b[a-fA-F0-9]{32,}\b")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)
ALLOWLIST = {
    ".env.example": {"api_or_secret_assignment", "email"},
}


def _git_files(args: list[str]) -> list[str]:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).splitlines()


def _tracked_files() -> list[str]:
    return _git_files(["ls-files"])


def _git_visible_files() -> list[str]:
    return _git_files(["ls-files", "--cached", "--others", "--exclude-standard"])


def _is_text_path(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS or path.name == ".gitkeep"


def check_tracked_file_boundary(tracked: list[str]) -> list[str]:
    problems = []
    for rel in tracked:
        normalized = rel.replace("\\", "/")
        path = Path(normalized)
        if normalized.startswith(FORBIDDEN_PREFIXES):
            problems.append(f"forbidden tracked data/cache path: {rel}")
        if path.suffix.lower() in FORBIDDEN_EXTENSIONS:
            problems.append(f"forbidden tracked binary/data artifact: {rel}")
        if path.name in {".env", ".pgpass"} or "pgpass" in path.name.lower():
            problems.append(f"forbidden tracked credential path: {rel}")
        if normalized == "Deposit Franchise and Duration Gap in Bank Equity Asset Pricing.md":
            problems.append(f"forbidden tracked internal proposal: {rel}")
    return problems


def check_secrets(paths: list[str]) -> list[str]:
    findings = []
    for rel in paths:
        path = ROOT / rel
        if not path.is_file() or not _is_text_path(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        allowed = ALLOWLIST.get(path.name, set())
        for name, pattern in SECRET_PATTERNS:
            if name in allowed:
                continue
            if pattern.search(text):
                findings.append(f"possible {name}: {rel}")
    return findings


def check_readme_images() -> list[str]:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme)
    missing = [ref for ref in refs if ref.startswith("figures/") and not (ROOT / ref).exists()]
    required_missing = [ref for ref in REQUIRED_README_IMAGES if not (ROOT / ref).exists()]
    return [f"missing README image: {ref}" for ref in sorted(set(missing + required_missing))]


def check_public_outputs() -> list[str]:
    problems = []
    summary_path = ROOT / "data_manifest" / "full_live_public_summary.json"
    if not summary_path.exists():
        problems.append("missing aggregate result summary: data_manifest/full_live_public_summary.json")
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("feature_store", {}).get("rows", 0) <= 0:
            problems.append("aggregate summary has no feature-store row count")
        if summary.get("visual_audit", {}).get("passed") != summary.get("visual_audit", {}).get("figures"):
            problems.append("visual audit summary is not all-pass")

    for html_path in (ROOT / "figures" / "interactive").glob("*.html"):
        text = html_path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"(?i)\b(permno|rssd|gvkey|pgpass|password|token|secret)\b", text):
            problems.append(f"interactive output contains blocked identifier/secret term: {html_path}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Emit machine-readable audit output.")
    args = parser.parse_args()

    tracked = _tracked_files()
    visible = _git_visible_files()
    problems = (
        check_tracked_file_boundary(tracked)
        + check_secrets(visible)
        + check_readme_images()
        + check_public_outputs()
    )
    result = {
        "status": "pass" if not problems else "fail",
        "tracked_files": len(tracked),
        "git_visible_files": len(visible),
        "problems": problems,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif problems:
        print("release_audit: FAIL")
        for problem in problems:
            print(f"- {problem}")
    else:
        print(
            "release_audit: PASS "
            f"({len(tracked)} tracked files, {len(visible)} git-visible files checked)"
        )
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
