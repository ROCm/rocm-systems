# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pytest hooks for mutation pipeline tests (session summary, step log, GITHUB_STEP_SUMMARY)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def mutation_session_reports(request: pytest.FixtureRequest) -> list:
    """Collect :class:`~mutate_and_test.models.ShaderReport` for session-end ``mutation_report.md``."""
    lst: list = []
    request.session._mutation_session_reports = lst  # type: ignore[attr-defined]
    return lst


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "mutation_pipeline: async data hazard mutation integration tests"
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    reports = getattr(session, "_mutation_session_reports", None)
    if not reports:
        return

    from mutate_and_test import write_json_report, write_markdown_report
    from mutate_and_test.report_writer import extract_summary_section_markdown

    arch = os.environ.get("TARGET_ARCH", "gfx950")
    root = Path(session.config.rootpath)
    json_path = root / "mutation_report.json"
    md_path = root / "mutation_report.md"
    write_json_report(reports, json_path, arch=arch)
    write_markdown_report(reports, md_path, arch=arch)
    full_text = md_path.read_text(encoding="utf-8")
    summary_only = extract_summary_section_markdown(full_text)
    print("\n::group::Mutation testing — Summary (from mutation_report.md)")
    print(summary_only)
    print("::endgroup::")

    # Job "Summary" tab (not the step log): requires appending markdown to this file.
    gh_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if gh_summary:
        with open(gh_summary, "a", encoding="utf-8") as f:
            f.write("## Mutation testing\n\n")
            f.write(summary_only)
            f.write("\n\n")
