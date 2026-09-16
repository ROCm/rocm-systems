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

"""Check registry, ``@register`` decorator, filtering, and the runner loop."""

from __future__ import absolute_import

import fnmatch
import traceback

from rocprofv3.doctor_result import (
    PROBE_PASSIVE,
    PROBE_VALUES,
    Result,
    STATUS_ERROR,
    STATUS_FAIL,
    STATUS_SKIP,
    SEV_ERROR,
    SEVERITY_VALUES,
)


class SelectionError(ValueError):
    """An --only pattern matched no check: almost always a typo."""


# Human-readable names for each check group, used as section headings in the
# text report. Groups not listed here fall back to their raw id.
GROUP_TITLES = {
    "install": "Installation",
    "driver": "Driver / Device",
    "runtime": "Runtime Stack",
    "counters": "Counter Collection",
    "environ": "Environment Variables",
    "container": "Container / Virtualization",
    "python": "Python Environment",
    "filesystem": "Filesystem",
    "tools": "Companion Tools",
    "smoke": "Live Smoke Tests",
}


class Check(object):
    """A single registered diagnostic.

    ``fn`` is a callable taking a ``SystemAccessor`` and returning a ``Result``.
    """

    __slots__ = (
        "id",
        "group",
        "title",
        "severity",
        "fn",
        "depends",
        "order",
        "default_enabled",
        "probe",
    )

    def __init__(
        self,
        id,  # noqa: A002  (mirrors the decorator keyword)
        group,
        title,
        severity,
        fn,
        depends,
        order,
        default_enabled=True,
        probe=PROBE_PASSIVE,
    ):
        if severity not in SEVERITY_VALUES:
            raise ValueError("invalid severity {!r} for check {!r}".format(severity, id))
        if probe not in PROBE_VALUES:
            raise ValueError("invalid probe {!r} for check {!r}".format(probe, id))
        self.id = id
        self.group = group
        self.title = title
        self.severity = severity
        self.fn = fn
        self.depends = list(depends) if depends else []
        self.order = order
        self.default_enabled = default_enabled
        self.probe = probe

    def __repr__(self):
        return "Check(id={!r}, order={!r})".format(self.id, self.order)


class Registry(object):
    """A collection of checks.

    A class rather than bare module state so that unit tests can build an
    isolated registry (for cycle detection, ordering, runner semantics) without
    perturbing the global one used by the real checks.
    """

    def __init__(self):
        self._checks = []

    def register(
        self,
        id,  # noqa: A002
        group,
        title,
        severity=SEV_ERROR,
        depends=None,
        order=50,
        default_enabled=True,
        probe=PROBE_PASSIVE,
    ):
        """Decorator factory used as ``@registry.register(id=..., ...)``."""

        def decorator(fn):
            self.add(
                Check(
                    id=id,
                    group=group,
                    title=title,
                    severity=severity,
                    fn=fn,
                    depends=depends,
                    order=order,
                    default_enabled=default_enabled,
                    probe=probe,
                )
            )
            return fn

        return decorator

    def add(self, check):
        for existing in self._checks:
            if existing.id == check.id:
                raise ValueError("duplicate check id {!r}".format(check.id))
        self._checks.append(check)
        return check

    def get_checks(self):
        """All registered checks sorted by ``(order, id)`` for stable output."""
        return sorted(self._checks, key=lambda c: (c.order, c.id))

    def groups(self):
        """Group ids in first-appearance order of their lowest-ordered check."""
        seen = []
        for check in self.get_checks():
            if check.group not in seen:
                seen.append(check.group)
        return seen

    def validate_dependency_graph(self):
        """Raise ``ValueError`` on unknown dependencies or dependency cycles."""
        ids = set()
        for check in self._checks:
            ids.add(check.id)

        for check in self._checks:
            for dep in check.depends:
                if dep not in ids:
                    raise ValueError(
                        "check {!r} depends on unknown check {!r}".format(check.id, dep)
                    )

        # Kahn's algorithm: repeatedly strip nodes with no unmet dependency.
        # Anything left over participates in a cycle.
        remaining = {}
        for check in self._checks:
            remaining[check.id] = set(check.depends)

        progressed = True
        while progressed and remaining:
            progressed = False
            ready = []
            for check_id, deps in remaining.items():
                if not deps:
                    ready.append(check_id)
            for check_id in ready:
                del remaining[check_id]
                progressed = True
            for deps in remaining.values():
                deps.difference_update(ready)

        if remaining:
            raise ValueError(
                "dependency cycle detected among checks: {}".format(
                    ", ".join(sorted(remaining.keys()))
                )
            )


# the global registry populated by the doctor_checks_* modules at import time
REGISTRY = Registry()


def register(*args, **kwargs):
    """Register a check in the global registry."""
    return REGISTRY.register(*args, **kwargs)


def get_checks():
    return REGISTRY.get_checks()


def validate_dependency_graph():
    return REGISTRY.validate_dependency_graph()


def _matches(check, patterns):
    """True if the check id or group matches any of the glob patterns."""
    for pattern in patterns:
        if fnmatch.fnmatch(check.id, pattern) or fnmatch.fnmatch(check.group, pattern):
            return True
    return False


def resolve_run_set(checks, only=None, skip=None, include_default_disabled=False):
    """Filter ``checks`` down to the set that should actually be executed.

    ``only`` and ``skip`` are lists of glob patterns matched against both the
    check id and the check group. Multiple patterns within a flag are OR'd;
    ``skip`` is applied after ``only``. Checks marked ``default_enabled=False``
    (the smoke group) run when ``include_default_disabled`` is set or when an
    ``only`` pattern names them -- either is an explicit opt-in.

    Prerequisites of a selected check are added to the run (unless ``skip``
    excludes them), so ``--only runtime.libraries-load`` still verifies that
    the libraries it loads exist first. Raises ``SelectionError`` when an
    ``only`` pattern matches no check at all.
    """
    only = list(only) if only else []
    skip = list(skip) if skip else []

    unmatched = [
        pattern
        for pattern in only
        if not any(_matches(check, [pattern]) for check in checks)
    ]
    if unmatched:
        raise SelectionError(
            "--only {} matches no check; see --list-checks".format(
                ", ".join(repr(pattern) for pattern in unmatched)
            )
        )

    by_id = dict((check.id, check) for check in checks)
    wanted = set()
    for check in checks:
        if only and not _matches(check, only):
            continue
        if not check.default_enabled:
            if not include_default_disabled and not _matches(check, only):
                continue
        wanted.add(check.id)

    # pull in prerequisites transitively
    pending = list(wanted)
    while pending:
        for dep in by_id[pending.pop()].depends:
            if dep not in wanted and dep in by_id:
                wanted.add(dep)
                pending.append(dep)

    return [
        check
        for check in checks
        if check.id in wanted and not (skip and _matches(check, skip))
    ]


def execution_order(check_list):
    """``check_list`` reordered so every check runs after its prerequisites.

    Registration order (``order``, then id) is kept wherever the dependency
    graph allows, so the report still reads group by group.
    """
    remaining = list(check_list)
    present = set(check.id for check in remaining)
    done = set()
    ordered = []
    while remaining:
        for index, check in enumerate(remaining):
            if all(dep in done or dep not in present for dep in check.depends):
                ordered.append(remaining.pop(index))
                done.add(check.id)
                break
        else:
            raise ValueError(
                "dependency cycle among checks: {}".format(
                    ", ".join(check.id for check in remaining)
                )
            )
    return ordered


def run_isolated(check, accessor):
    """Run one check, converting any escaping exception into an ``error``.

    A buggy or environment-surprised check must never take down the report,
    and must not be mistaken for a problem with the system: the result is
    ``error`` (a doctor bug), not ``fail``.
    """
    try:
        result = check.fn(accessor)
    except Exception:  # noqa: BLE001 -- isolation is the entire point
        return Result(
            status=STATUS_ERROR,
            detail="the check itself raised an unhandled exception; this says "
            "nothing about the system",
            remediation=(
                "This is a rocprofv3-doctor bug; please report it with the "
                "traceback shown by --verbose or --format json."
            ),
            data={"traceback": traceback.format_exc()},
        )
    if not isinstance(result, Result):
        return Result(
            status=STATUS_ERROR,
            detail="check returned {!r} instead of a Result".format(
                type(result).__name__
            ),
            remediation=(
                "This is a rocprofv3-doctor bug; please report it against the "
                "check id shown above."
            ),
        )
    return result


def run_checks(accessor, check_list):
    """Run checks after their prerequisites, propagating skips.

    Returns a list of ``(check, result)`` pairs in execution order. A check
    whose prerequisite failed, errored, or was itself skipped reports ``skip``
    naming that prerequisite, so a single root cause produces one failure
    rather than a cascade of red. A prerequisite excluded from the run by
    ``--skip`` also skips its dependents: nothing vouches for it.
    """
    results = {}
    ordered = []
    for check in execution_order(check_list):
        blocker = None
        reason = None
        for dep_id in check.depends:
            dep_result = results.get(dep_id)
            if dep_result is None:
                blocker = dep_id
                reason = "skipped: prerequisite {!r} was excluded from this run"
                break
            if dep_result.status in (STATUS_FAIL, STATUS_SKIP, STATUS_ERROR):
                blocker = dep_id
                reason = "skipped: depends on {!r} which did not pass"
                break

        if blocker is not None:
            result = Result(
                status=STATUS_SKIP,
                detail=reason.format(blocker),
                data={"skipped_because_of": blocker},
            )
        else:
            result = run_isolated(check, accessor)

        results[check.id] = result
        ordered.append((check, result))
    return ordered
