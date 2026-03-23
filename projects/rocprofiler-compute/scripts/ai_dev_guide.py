#!/usr/bin/env python3
##############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################
"""Validate .ai/ layout, guide docs, and skills for rocprofiler-compute."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Project root = parent of scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_PATHS: tuple[str, ...] = (
    "scripts/ai_dev_guide.py",
    ".ai/README.md",
    ".ai/CLAUDE.md",
    ".ai/ROADMAP.md",
    ".ai/guide/workflow.md",
    ".ai/guide/taxonomy.md",
    ".ai/rules/core.md",
    ".ai/rules/anti_patterns.md",
    ".ai/rules/profiling_infra.md",
    ".ai/rules/security.md",
    ".ai/rules/tools_policy.md",
    ".ai/standards/python.md",
    ".ai/standards/cpp.md",
    ".ai/standards/cmake.md",
    ".ai/standards/agent_output.md",
    ".ai/prompts/default.md",
    ".ai/prompts/run_session.md",
    ".ai/review/checklist.md",
    ".ai/skills/index.md",
    ".ai/skills/add_feature.md",
    ".ai/skills/fix_bug.md",
    ".ai/skills/write_test.md",
    ".ai/skills/add_experimental_cli.md",
    ".ai/skills/update_soc_or_counters.md",
    ".ai/skills/analyze_or_roofline.md",
    ".ai/skills/execution_graph_trace.md",
    ".ai/skills/optimize_performance.md",
    ".ai/skills/native_or_cmake.md",
    ".ai/skills/fix_build_failure.md",
    ".ai/skills/code_review.md",
    ".claude/skills/code-reviewer/SKILL.md",
    "AGENTS.md",
    "CLAUDE.md",
    ".cursor/rules/rocprofiler-compute-ai.mdc",
    ".github/copilot-instructions.md",
    "docs/AI_GUIDE.md",
    ".claude/README.md",
    ".claude/settings.json",
    ".claude/capabilities.md",
    ".claude/tools-policy.md",
    ".claude/rules/claude-guide.md",
    ".claude/hooks/bash_guard.py",
)

SKILL_HEADINGS: tuple[str, ...] = (
    "## Goal",
    "## Steps",
    "## Constraints",
    "## Output",
)

_INDEX_LINK_RE = re.compile(r"\]\(([^)]+\.md)\)")


def _skills_linked_from_index() -> list[str]:
    index_path = PROJECT_ROOT / ".ai" / "skills" / "index.md"
    index_body = index_path.read_text(encoding="utf-8")
    linked_md = _INDEX_LINK_RE.findall(index_body)
    names = sorted(set(linked_md))
    return [n for n in names if "/" not in n and not n.startswith(".")]


def check_required_paths(verbose: bool) -> list[str]:
    errors: list[str] = []
    for rel in REQUIRED_PATHS:
        path = PROJECT_ROOT / rel
        if not path.is_file():
            errors.append(f"Missing required file: {rel}")
        elif verbose:
            print(f"OK {rel}")
    return errors


def check_skill_index_links(verbose: bool) -> list[str]:
    errors: list[str] = []
    linked = _skills_linked_from_index()
    skills_dir = PROJECT_ROOT / ".ai" / "skills"
    for name in linked:
        path = skills_dir / name
        if not path.is_file():
            errors.append(f"Skill index links to missing file: .ai/skills/{name}")
        elif verbose:
            print(f"OK index -> skills/{name}")
    return errors


def check_skill_templates(verbose: bool) -> list[str]:
    errors: list[str] = []
    skills_dir = PROJECT_ROOT / ".ai" / "skills"
    for path in sorted(skills_dir.glob("*.md")):
        if path.name == "index.md":
            continue
        text = path.read_text(encoding="utf-8")
        for heading in SKILL_HEADINGS:
            if heading not in text:
                errors.append(f"{path.relative_to(PROJECT_ROOT)}: missing section {heading!r}")
        if verbose:
            print(f"OK template {path.relative_to(PROJECT_ROOT)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate .ai/ guide layout.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print checked paths.")
    args = parser.parse_args()

    if not (PROJECT_ROOT / ".ai" / "README.md").is_file():
        print("Run from rocprofiler-compute tree (scripts/ must live under project root).", file=sys.stderr)
        return 2

    all_errors: list[str] = []
    all_errors.extend(check_required_paths(args.verbose))
    all_errors.extend(check_skill_index_links(args.verbose))
    all_errors.extend(check_skill_templates(args.verbose))

    if all_errors:
        print("ai_dev_guide: failed", file=sys.stderr)
        for line in all_errors:
            print(f"  {line}", file=sys.stderr)
        return 1

    if args.verbose:
        print("ai_dev_guide: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
