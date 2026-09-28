# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Write the summary, RESULT markdown, and log archive."""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import NoReturn


from .command import emit, fmt_dur, log, run_start, step_times


def write_summary(summary_log: Path, facts: dict) -> None:
    """Write the concise summary to a file and echo it to the console/detailed log."""
    facts["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
    order = [
        "result",
        "mode",
        "validation",
        "error",
        "pytest_exit",
        "tests_total",
        "passed",
        "failed",
        "errors",
        "skipped",
        "duration_sec",
        "tier",
        "run_labels",
        "mpi",
        "reruns",
        "rerun_failed_attempts",
        "cmake_args",
        "variant",
        "gpu_arch",
        "host",
        "scheduler",
        "tarball",
        "rocm_version",
        "rocprofsys_version",
        "sha256_verified",
        "sha256",
        "rocm_updated",
        "tarball_url",
        "branch",
        "git_revision",
        "git_subject",
        "source_changed",
        "examples_rebuilt",
        "examples_installed",
        "disabled_examples",
        "build_dir",
        "work_dir",
        "rocm_prefix",
        "detailed_log",
        "rocm_sanity_log",
        "pip_freeze_log",
        "failures_log",
        "started",
        "finished",
        "command",
    ]
    lines = ["rocprof-sys nightly tarball test - SUMMARY", "=" * 44]
    for key in order:
        if key in facts:
            lines.append(f"{key:22s}: {facts[key]}")

    failed_tests = facts.get("_failed_tests") or []
    if failed_tests:
        lines.append("")
        lines.append(f"failed tests ({len(failed_tests)}):")
        for name in failed_tests[:50]:
            lines.append(f"  - {name}")
        if len(failed_tests) > 50:
            lines.append(f"  ... and {len(failed_tests) - 50} more (see failures log)")

    if step_times():
        lines.append("")
        lines.append("step timings:")
        for title, dur in step_times():
            lines.append(f"  {fmt_dur(dur):>9}  {title}")
        lines.append(f"  {fmt_dur(time.monotonic() - run_start()):>9}  TOTAL wall-clock")

    text = "\n".join(lines)

    with open(summary_log, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")

    emit("\n" + text)
    emit(f"\n[nightly-test] Summary written to: {summary_log}")

    # Gap 8: human-readable Markdown result. Gap 7: archive logs + latest pointers.
    write_result_md(summary_log, facts)
    archive_logs(summary_log, facts)


def write_result_md(summary_log: Path, facts: dict) -> None:
    """Write a Markdown RESULT file (mirrors the QA RESULT_*.md) next to the logs."""
    workdir = summary_log.parent
    stamp = summary_log.stem.replace("summary-", "")
    md = workdir / f"RESULT-{stamp}.md"

    def g(key, default="-"):
        return facts.get(key, default)

    lines = [
        "# rocprofiler-systems tarball validation",
        "",
        f"- Result: **{g('result')}**",
        f"- Mode: {g('mode')}",
        f"- Finished (UTC-local): {g('finished')}",
        f"- Host: {g('host')}",
        f"- Scheduler: {g('scheduler')}",
        f"- Work dir: {g('work_dir')}",
        "",
        "## Under test",
        f"- Tarball: {g('tarball')}",
        f"- ROCm version: {g('rocm_version')}",
        f"- rocprof-sys version: {g('rocprofsys_version')}",
        f"- SHA-256 verified: {g('sha256_verified')}",
        f"- Source branch: {g('branch')}",
        f"- Source commit: {g('git_revision')}  {g('git_subject', '')}",
        "",
        "## Test results",
        f"- pytest exit: {g('pytest_exit')}",
        f"- total / passed / failed / errors / skipped: "
        f"{g('tests_total')} / {g('passed')} / {g('failed')} / "
        f"{g('errors')} / {g('skipped')}",
        f"- tier: {g('tier')}",
        f"- reruns (inline): {g('reruns', 0)}",
        f"- failed-rerun attempts: {g('rerun_failed_attempts', 0)}",
    ]
    failed_tests = facts.get("_failed_tests") or []
    if failed_tests:
        lines += ["", "## Failed tests"]
        lines += [f"- {t}" for t in failed_tests[:100]]
        if len(failed_tests) > 100:
            lines.append(f"- ... and {len(failed_tests) - 100} more")

    if step_times():
        lines += ["", "## Stage timings"]
        lines += [f"- {title}: {fmt_dur(dur)}" for title, dur in step_times()]

    lines += [
        "",
        "## Logs",
        f"- Detailed: {g('detailed_log')}",
        f"- Summary: {summary_log}",
        f"- ROCm sanity: {g('rocm_sanity_log')}",
        f"- pip freeze: {g('pip_freeze_log')}",
        f"- Failures: {g('failures_log')}",
    ]
    md.write_text("\n".join(lines) + "\n")
    facts["result_md"] = str(md)
    log(f"RESULT written to: {md}")


def archive_logs(summary_log: Path, facts: dict) -> None:
    """Bundle this run's log artifacts into a tar.gz and update 'latest' pointers."""
    workdir = summary_log.parent
    stamp = summary_log.stem.replace("summary-", "")
    members = [
        workdir / f"detailed-{stamp}.log",
        workdir / f"summary-{stamp}.log",
        workdir / f"RESULT-{stamp}.md",
        workdir / f"rocm-sanity-{stamp}.log",
        workdir / f"pip-freeze-{stamp}.txt",
        workdir / f"failures-{stamp}.log",
        workdir / "pytest-results.xml",
    ] + sorted(workdir.glob("pytest-rerun-*.xml"))

    archive = workdir / f"logs-{stamp}.tar.gz"
    try:
        with tarfile.open(archive, "w:gz") as tf:
            for m in members:
                if m.is_file():
                    tf.add(m, arcname=m.name)
    except Exception as exc:  # noqa: BLE001
        log(f"WARNING: could not create log archive: {exc}")
        return

    # 'latest' pointers so automation/QA can find the newest run at a fixed path.
    (workdir / "latest-summary.txt").write_text(str(summary_log) + "\n")
    (workdir / "latest-archive.txt").write_text(str(archive) + "\n")
    if facts.get("result_md"):
        (workdir / "latest-result.txt").write_text(str(facts["result_md"]) + "\n")
    facts["log_archive"] = str(archive)
    log(f"Log archive: {archive}")
