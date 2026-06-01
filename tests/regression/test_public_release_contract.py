from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_required_public_release_files_exist():
    required = [
        "README.md",
        "docs/index.md",
        "paper/paper.qmd",
        "configs/project.yml",
        "configs/schema_map.yml",
        "Makefile",
        "environment.yml",
        "LICENSE",
        "CITATION.cff",
        ".github/workflows/ci.yml",
        ".github/workflows/pages.yml",
        "sql/bank_regulatory_extract.sql",
        "sql/crsp_stock_monthly.sql",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    assert not missing


def test_project_config_freezes_sample_and_lags():
    cfg = yaml.safe_load((ROOT / "configs/project.yml").read_text())
    assert cfg["project"]["main_sample_end"] == "2025-12-31"
    assert cfg["project"]["final_holdout_start"] == "2026-01-01"
    assert cfg["project"]["public_information_lag_days"] == 45
    assert cfg["project"]["lag_robustness_days"] == [30, 45, 60]


def test_readme_states_research_question_results_and_data_boundary():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    flat_text = " ".join(text.split())
    assert "Can bank balance-sheet duration exposure" in text
    assert "Answer: yes." in text
    assert "Main duration-gap Fama-MacBeth t-stat" in text
    assert "figures/static/hero_figure.png" in text
    assert "raw WRDS/CRSP/bank-regulatory extracts" in flat_text
    assert "passwords, API keys, private logs" in flat_text


def test_docs_build_script_supports_file_output(tmp_path):
    from scripts.build_docs import main

    out = tmp_path / "site" / "index.html"
    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["build_docs.py", "--docs", str(ROOT / "docs" / "index.md"), "--out", str(out)]
        assert main() == 0
    finally:
        sys.argv = old_argv
    assert out.exists()
    assert "<h1>" in out.read_text(encoding="utf-8")


def test_release_audit_passes():
    from scripts.release_audit import main

    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["release_audit.py"]
        assert main() == 0
    finally:
        sys.argv = old_argv
