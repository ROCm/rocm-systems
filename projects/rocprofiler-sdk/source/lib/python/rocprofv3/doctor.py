# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""Public entry points for rocprofv3-doctor: run the checks, render a report.

The ``source/bin/rocprofv3-doctor`` script is a thin argparse wrapper over this
module; all of the interesting logic lives here so it can be unit-tested
without spawning a subprocess.
"""

from __future__ import absolute_import

import datetime
import json

from rocprofv3 import doctor_layout
from rocprofv3.doctor_env import SystemAccessor
from rocprofv3.doctor_registry import (
    GROUP_TITLES,
    get_checks,
    resolve_run_set,
    run_checks,
    SelectionError,
    validate_dependency_graph,
)
from rocprofv3.doctor_result import (
    Result,
    STATUS_ERROR,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_SKIP,
    STATUS_VALUES,
    STATUS_WARN,
    SEV_INFO,
)

# Importing the check modules is what populates the registry. Keep these
# imports after the framework imports and never remove them: they look unused
# to a linter but they are the registration mechanism.
from rocprofv3 import doctor_checks_install  # noqa: F401
from rocprofv3 import doctor_checks_driver  # noqa: F401
from rocprofv3 import doctor_checks_runtime  # noqa: F401
from rocprofv3 import doctor_checks_counters  # noqa: F401
from rocprofv3 import doctor_checks_environ  # noqa: F401
from rocprofv3 import doctor_checks_container  # noqa: F401
from rocprofv3 import doctor_checks_python_env  # noqa: F401
from rocprofv3 import doctor_checks_filesystem  # noqa: F401
from rocprofv3 import doctor_checks_tools  # noqa: F401
from rocprofv3 import doctor_checks_smoke  # noqa: F401

# Bump only on a backward-incompatible change to the JSON layout.
SCHEMA_VERSION = 1

EXIT_OK = 0
EXIT_FAILURES = 1
EXIT_TOOL_ERROR = 2

_COLORS = {
    STATUS_PASS: "\033[32m",
    STATUS_WARN: "\033[33m",
    STATUS_FAIL: "\033[31m",
    STATUS_SKIP: "\033[90m",
    STATUS_ERROR: "\033[35m",
}
_RESET = "\033[0m"
_BOLD = "\033[1m"

_TAGS = {
    STATUS_PASS: "[ PASS ]",
    STATUS_WARN: "[ WARN ]",
    STATUS_FAIL: "[ FAIL ]",
    STATUS_SKIP: "[ SKIP ]",
    STATUS_ERROR: "[ ERR  ]",
}

_ID_COLUMN = 34


def validate_registry():
    """Validate the dependency graph of the globally registered checks."""
    validate_dependency_graph()


def resolve_rocm_root(script_path, override=None):
    """Locate the ROCm root to inspect; return ``(root, how_it_was_found)``."""
    return doctor_layout.detect_rocm_root(
        SystemAccessor(rocm_root=None), script_path, override
    )


def build_accessor(rocm_root, tool_version=None, rocm_root_source=None):
    return SystemAccessor(
        rocm_root=rocm_root,
        tool_version=tool_version,
        rocm_root_source=rocm_root_source,
    )


def installation_kind(accessor):
    """How the inspected ROCm tree was installed (a doctor_layout.INSTALL_* value)."""
    return doctor_layout.install_kind(accessor)


def describe_installation(accessor):
    """One-line summary of the inspected root, for the report header."""
    parts = []
    if accessor.rocm_root_source:
        parts.append("found via {}".format(accessor.rocm_root_source))
    parts.append(doctor_layout.INSTALL_KIND_TITLES[installation_kind(accessor)])
    return "{}  ({})".format(accessor.rocm_root, "; ".join(parts))


def select_checks(only=None, skip=None, include_default_disabled=False):
    return resolve_run_set(
        get_checks(),
        only=only,
        skip=skip,
        include_default_disabled=include_default_disabled,
    )


def run(accessor, only=None, skip=None, include_default_disabled=False):
    """Run the selected checks; return a list of ``(check, result)`` pairs."""
    selected = select_checks(
        only=only, skip=skip, include_default_disabled=include_default_disabled
    )
    return run_checks(accessor, selected)


def summarize(outcomes):
    """Count outcomes by status."""
    summary = {
        STATUS_PASS: 0,
        STATUS_WARN: 0,
        STATUS_FAIL: 0,
        STATUS_SKIP: 0,
        STATUS_ERROR: 0,
        "total": 0,
    }
    for _, result in outcomes:
        summary[result.status] += 1
        summary["total"] += 1
    return summary


def exit_code(outcomes):
    """1 when a check failed; else 2 when a check itself broke; else 0.

    A failure is a finding about the system and outranks a doctor bug, which
    only means one check could not reach a conclusion.
    """
    statuses = set(result.status for _, result in outcomes)
    if STATUS_FAIL in statuses:
        return EXIT_FAILURES
    if STATUS_ERROR in statuses:
        return EXIT_TOOL_ERROR
    return EXIT_OK


# ----------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------
def _colorize(text, status, use_color):
    if not use_color:
        return text
    return "{}{}{}".format(_COLORS.get(status, ""), text, _RESET)


def _indent_block(text, prefix):
    lines = []
    for line in text.splitlines():
        lines.append("{}{}".format(prefix, line) if line else "")
    return "\n".join(lines)


def _group_heading(group):
    title = GROUP_TITLES.get(group, group)
    return "{}\n{}".format(title, "=" * len(title))


def render_text(
    outcomes,
    header=None,
    use_color=False,
    verbose=False,
    quiet=False,
):
    """Render the grouped human-readable report."""
    lines = []
    if header:
        lines.append(header)
        lines.append("")

    current_group = None
    for check, result in outcomes:
        if quiet and result.status not in (STATUS_FAIL, STATUS_WARN):
            continue

        if check.group != current_group:
            current_group = check.group
            if lines and lines[-1] != "":
                lines.append("")
            heading = _group_heading(check.group)
            if use_color:
                heading = "{}{}{}".format(_BOLD, heading, _RESET)
            lines.append(_indent_block(heading, "  "))

        tag = _colorize(_TAGS[result.status], result.status, use_color)
        lines.append("  {}  {}  {}".format(tag, check.id.ljust(_ID_COLUMN), check.title))

        show_detail = verbose or result.status != STATUS_PASS
        if show_detail and result.detail:
            lines.append(_indent_block(result.detail, "             "))
        if verbose and result.data:
            for key in sorted(result.data.keys()):
                lines.append(
                    _indent_block("{}: {}".format(key, result.data[key]), "             ")
                )

    summary = summarize(outcomes)
    lines.append("")
    lines.append(
        "  Summary: {} passed, {} warning(s), {} failed, {} skipped{} "
        "({} total)".format(
            summary[STATUS_PASS],
            summary[STATUS_WARN],
            summary[STATUS_FAIL],
            summary[STATUS_SKIP],
            (
                ", {} check error(s)".format(summary[STATUS_ERROR])
                if summary[STATUS_ERROR]
                else ""
            ),
            summary["total"],
        )
    )

    actionable = []
    for check, result in outcomes:
        if result.status in (STATUS_FAIL, STATUS_ERROR):
            actionable.append((check, result))
    for check, result in outcomes:
        if result.status == STATUS_WARN and (
            result.remediation or check.severity != SEV_INFO
        ):
            actionable.append((check, result))

    if actionable:
        lines.append("")
        lines.append(_indent_block(_group_heading("How to Fix"), "  "))
        for check, result in actionable:
            lines.append("")
            tag = _colorize(
                "[{}]".format(result.status.upper()), result.status, use_color
            )
            lines.append("  {} {} - {}".format(tag, check.id, check.title))
            if result.detail:
                lines.append(_indent_block(result.detail, "      "))
            if result.remediation:
                lines.append("")
                lines.append(_indent_block(result.remediation, "        "))

    return "\n".join(lines) + "\n"


def build_report(
    outcomes,
    tool_version=None,
    rocm_version=None,
    rocm_root=None,
    rocm_root_source=None,
    install_kind=None,
):
    """Build the JSON-serializable report dictionary."""
    summary = summarize(outcomes)
    checks = []
    for check, result in outcomes:
        entry = {
            "id": check.id,
            "group": check.group,
            "title": check.title,
            "severity": check.severity,
            "probe": check.probe,
            "depends": list(check.depends),
        }
        entry.update(result.to_dict())
        checks.append(entry)

    return {
        "schema_version": SCHEMA_VERSION,
        "tool_version": tool_version or "",
        "rocm_version": rocm_version or "",
        "rocm_root": rocm_root or "",
        "rocm_root_source": rocm_root_source or "",
        "install_kind": install_kind or "",
        "timestamp_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": {
            "pass": summary[STATUS_PASS],
            "warn": summary[STATUS_WARN],
            "fail": summary[STATUS_FAIL],
            "skip": summary[STATUS_SKIP],
            "error": summary[STATUS_ERROR],
            "total": summary["total"],
        },
        "checks": checks,
    }


def render_json(
    outcomes,
    tool_version=None,
    rocm_version=None,
    rocm_root=None,
    rocm_root_source=None,
    install_kind=None,
):
    report = build_report(
        outcomes,
        tool_version=tool_version,
        rocm_version=rocm_version,
        rocm_root=rocm_root,
        rocm_root_source=rocm_root_source,
        install_kind=install_kind,
    )
    return json.dumps(report, indent=2, sort_keys=False, default=str) + "\n"


def render_check_list():
    """Render the ``--list-checks`` output."""
    lines = []
    current_group = None
    for check in get_checks():
        if check.group != current_group:
            current_group = check.group
            if lines:
                lines.append("")
            lines.append(
                "{} ({})".format(GROUP_TITLES.get(check.group, check.group), check.group)
            )
        suffix = "" if check.default_enabled else "  [opt-in: --run-smoke-test]"
        lines.append(
            "  {}  {:<8} {}{}".format(
                check.id.ljust(_ID_COLUMN), check.probe, check.title, suffix
            )
        )
    lines.append("")
    lines.append(
        "probe: passive = reads files and the environment; process = starts ROCm "
        "code in a child process; gpu = initializes the GPU runtime"
    )
    lines.append("opt-in checks also run when selected by name with --only")
    return "\n".join(lines) + "\n"


__all__ = [
    "EXIT_FAILURES",
    "EXIT_OK",
    "EXIT_TOOL_ERROR",
    "Result",
    "SCHEMA_VERSION",
    "SelectionError",
    "STATUS_ERROR",
    "STATUS_FAIL",
    "STATUS_PASS",
    "STATUS_SKIP",
    "STATUS_VALUES",
    "STATUS_WARN",
    "SystemAccessor",
    "build_accessor",
    "build_report",
    "describe_installation",
    "exit_code",
    "get_checks",
    "installation_kind",
    "render_check_list",
    "render_json",
    "render_text",
    "resolve_rocm_root",
    "run",
    "select_checks",
    "summarize",
    "validate_registry",
]
