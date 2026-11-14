# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Minimal, strict reader for a workflow's pull_request trigger, for tests.

No pyyaml: the job that runs these tests installs only pytest. Anything the
reader cannot vouch for raises AssertionError instead of being skipped.
"""

import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TYPES = {"opened", "synchronize", "reopened"}

# Conditions that depend only on earlier steps' status, never on the event.
STATUS_ONLY_IFS = {"always()", "!cancelled()", "success()", "failure()"}


@dataclass(frozen=True)
class PullRequestTrigger:
    branches: set[str]
    paths: list[str]
    types: set[str]
    # Every job- and step-level `if:` condition, `${{ }}` stripped.
    ifs: list[str]


def pull_request_trigger(workflow: Path) -> PullRequestTrigger:
    lines = [
        re.sub(r"\s+#.*$", "", l)
        for l in workflow.read_text(encoding="utf-8").splitlines()
    ]
    start = next(i for i, l in enumerate(lines) if re.match(r"^  pull_request:\s*$", l))
    filters: dict[str, list[str]] = {}
    key = None
    for line in lines[start + 1 :]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith("    "):
            break
        if m := re.match(r"^    ([\w-]+):\s*(.*)$", line):
            key, inline = m.groups()
            if key not in ("branches", "paths", "types"):
                raise AssertionError(f"{workflow}: unsupported pull_request key {key}")
            filters[key] = re.findall(r"[^\[\],'\"\s]+", inline)
        elif key and (item := re.match(r"^\s+-\s*['\"]?([^'\"\s]+)", line)):
            filters[key].append(item.group(1))
        else:
            raise AssertionError(f"{workflow}: cannot parse {line!r}")
    paths = filters.get("paths", [])
    if any(p.startswith("!") for p in paths):
        raise AssertionError(f"{workflow}: negated paths {paths}")

    jobs = next((i for i, l in enumerate(lines) if l == "jobs:"), len(lines))
    ifs = [
        re.sub(r"^\$\{\{\s*(.*?)\s*\}\}$", r"\1", m.group(1).strip())
        for l in lines[jobs:]
        if (m := re.match(r"^\s+(?:-\s+)?if:\s*(.*)$", l))
    ]
    return PullRequestTrigger(
        branches=set(filters.get("branches", [])),
        paths=paths,
        types=set(filters.get("types", [])) or DEFAULT_TYPES,
        ifs=ifs,
    )


def job_lines(workflow: Path) -> dict[str, list[str]]:
    """Stripped, comment-free lines of each top-level job, by job id."""
    jobs: dict[str, list[str]] = {}
    job, in_jobs = None, False
    for line in workflow.read_text(encoding="utf-8").splitlines():
        line = re.sub(r"\s+#.*$", "", line)
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            in_jobs, job = line == "jobs:", None
        elif in_jobs and (m := re.match(r"^  ([\w-]+):$", line)):
            job = m.group(1)
            jobs[job] = []
        elif job:
            jobs[job].append(line.strip())
    return jobs
