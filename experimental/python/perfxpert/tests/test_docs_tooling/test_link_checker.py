"""Unit coverage for the docs link checker."""

import importlib.util
import subprocess
import sys
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "link-checker.py"
_SPEC = importlib.util.spec_from_file_location("link_checker", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
link_checker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(link_checker)


def test_strict_link_checker_reports_missing_anchor(tmp_path: Path) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text("# Real Heading\n\n[bad](#missing-heading)\n", encoding="utf-8")

    broken = link_checker.find_broken_links(tmp_path, validate_anchors=True)

    assert broken == [
        {
            "file": "doc.md",
            "line": 3,
            "link": "#missing-heading",
            "text": "bad",
        }
    ]


def test_strict_link_checker_accepts_heading_anchors(tmp_path: Path) -> None:
    target = tmp_path / "target.md"
    source = tmp_path / "source.md"
    target.write_text("# First Run — `perfxpert init`\n", encoding="utf-8")
    source.write_text("[ok](target.md#first-run-perfxpert-init)\n", encoding="utf-8")

    assert link_checker.find_broken_links(tmp_path, validate_anchors=True) == []


def test_strict_cli_enables_anchor_validation(tmp_path: Path) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text("# Real Heading\n\n[bad](#missing-heading)\n", encoding="utf-8")

    non_strict = subprocess.run(
        [sys.executable, str(_SCRIPT), str(tmp_path)],
        check=False,
        text=True,
        capture_output=True,
    )
    strict = subprocess.run(
        [sys.executable, str(_SCRIPT), str(tmp_path), "--strict"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert non_strict.returncode == 0, non_strict.stdout + non_strict.stderr
    assert strict.returncode == 1
    assert '"doc.md",3,"#missing-heading","bad"' in strict.stdout
