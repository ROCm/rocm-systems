#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Repeat selected ctest tests for a fixed wall-clock budget and tally the outcomes.

This runs as a single RUN_SERIAL ctest test. It lists the tests of the same build tree with
``ctest --show-only=json-v1`` and runs each selected test's command itself rather than through
a nested ctest, because a nested ctest would write the Testing/ logs that the outer dashboard
run still has open.

The budget cannot produce a false TIMEOUT: a test starts with its full timeout only if that fits
in the remaining budget. Near the end, a test whose duration has already been observed may still
start with its timeout cut down to what remains; if that cut timeout expires it is reported as
CUT, which is not a failure. A test that genuinely times out twice is dropped for the rest of
the run so that one hang cannot consume the budget of every other test.

Must stay Python 3.6 compatible: rhel-8.8 CI images ship 3.6 as python3.
"""

from __future__ import print_function

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

# The kernel-replay concurrency stress test plus the tests that failed on the four
# remove-callbacks PRs, and the per-service queue_hooks unit tests.
DEFAULT_SELECT = [
    r"^tests\.integration\.execute\.test-kernel-replay-concurrency$",
    r"^tests\.integration\.execute\..*kernel-replay-local-context",
    r"^rocprofv3-test-kernel-replay(-interval)?-generate$",
    r"^tests\.integration\.execute\.rocprofv3-test-rocshmem-tracing$",
    r"^tests\.integration\.execute\.rocprofv3-test-attachment-attach-once-att$",
    r"^tests\.integration\.execute\.rocprofv3-test-hip-streams-per-thread$",
    r"^tests\.integration\.execute\.rocprofv3-test-roctx-pause-resume",
    r"^tests\.integration\.execute\.anytime-",
    # counters_queue_hooks, spm_queue_hooks, pc_sampling_queue_hooks, ThreadTraceQueueHooks
    r"(?i)^unit\.[a-z_]*queue_?hooks[a-z_]*\.",
    r"^unit\.kernel_replay_",
]

# Never select this harness (recursion) or validation tests, whose inputs are the output of an
# execute test from a different iteration.
DEFAULT_EXCLUDE = [r"^stress\.", r"\.validate\.", r"-validate$"]

BAD = ("FAIL", "TIMEOUT", "SIGNAL", "SETUP_FAILED")

# abseil's failure signal handler (enabled via ROCPROFILER_FAILURE_SIGNAL_HANDLER) starts its
# report with this marker; when present it is the part of the output worth keeping.
CRASH_MARKER = "*** SIG"


def log(msg):
    print(msg)
    sys.stdout.flush()


def compile_patterns(patterns):
    out = []
    for pat in patterns:
        try:
            out.append(re.compile(pat))
        except re.error:
            # ctest regexes are not all valid Python regexes; fall back to a literal match
            out.append(re.compile(re.escape(pat)))
    return out


def any_match(patterns, text):
    return any(p.search(text) for p in patterns)


class Test(object):
    def __init__(self, entry):
        self.name = entry["name"]
        self.command = entry.get("command") or []
        self.props = {}
        for prop in entry.get("properties") or []:
            self.props[prop["name"]] = prop["value"]

    def prop_list(self, key):
        value = self.props.get(key)
        if value is None:
            return []
        return value if isinstance(value, list) else [value]

    @property
    def disabled(self):
        return bool(self.props.get("DISABLED", False))

    @property
    def timeout(self):
        value = self.props.get("TIMEOUT")
        try:
            return float(value) if value else None
        except (TypeError, ValueError):
            return None


def list_tests(ctest, test_dir):
    out = subprocess.check_output(
        [ctest, "--test-dir", test_dir, "--show-only=json-v1"], stderr=subprocess.PIPE
    )
    data = json.loads(out.decode("utf-8", "replace"))
    return [Test(entry) for entry in data.get("tests", [])]


def build_env(test):
    env = dict(os.environ)
    for item in test.prop_list("ENVIRONMENT"):
        key, sep, value = item.partition("=")
        if sep:
            env[key] = value
    return env


def classify(test, returncode, output):
    """Reproduce ctest's verdict for the properties this test tree uses."""
    if returncode is not None and returncode < 0:
        return "SIGNAL"

    skip = compile_patterns(test.prop_list("SKIP_REGULAR_EXPRESSION"))
    if skip and any_match(skip, output):
        return "SKIP"

    passing = compile_patterns(test.prop_list("PASS_REGULAR_EXPRESSION"))
    ok = any_match(passing, output) if passing else returncode == 0

    failing = compile_patterns(test.prop_list("FAIL_REGULAR_EXPRESSION"))
    if failing and any_match(failing, output):
        ok = False

    if test.props.get("WILL_FAIL", False):
        ok = not ok

    return "PASS" if ok else "FAIL"


def run_test(test, timeout, default_cwd):
    cwd = test.props.get("WORKING_DIRECTORY") or default_cwd
    start = time.monotonic()
    try:
        proc = subprocess.Popen(
            test.command,
            cwd=cwd,
            env=build_env(test),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as err:
        return "FAIL", None, "failed to launch: {}".format(err), 0.0

    try:
        raw, _ = proc.communicate(timeout=timeout)
        status = None
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        raw, _ = proc.communicate()
        status = "TIMEOUT"

    elapsed = time.monotonic() - start
    output = raw.decode("utf-8", "replace")
    if status is None:
        status = classify(test, proc.returncode, output)
    return status, proc.returncode, output, elapsed


def excerpt(output, max_lines, max_bytes):
    """The crash report if the fault handler printed one, otherwise the tail."""
    idx = output.find(CRASH_MARKER)
    lines = (output[idx:] if idx >= 0 else output).splitlines()
    lines = lines[:max_lines] if idx >= 0 else lines[-max_lines:]
    text = "\n".join(lines)
    if len(text) > max_bytes:
        text = text[:max_bytes] + "\n[... excerpt truncated ...]"
    return text


class Setups(object):
    """Runs the fixture setups a selected test requires, once per iteration."""

    def __init__(self, all_tests):
        self.by_fixture = {}
        for test in all_tests:
            for fixture in test.prop_list("FIXTURES_SETUP"):
                self.by_fixture.setdefault(fixture, []).append(test)
        self.iteration = None
        self.results = {}

    def ensure(self, test, iteration, timeout_cap, default_cwd):
        if iteration != self.iteration:
            self.iteration, self.results = iteration, {}
        for fixture in test.prop_list("FIXTURES_REQUIRED"):
            for setup in self.by_fixture.get(fixture, []):
                if setup.name == test.name:
                    continue
                if setup.name not in self.results:
                    timeout = min(setup.timeout or timeout_cap, timeout_cap)
                    status = run_test(setup, timeout, default_cwd)[0]
                    self.results[setup.name] = status
                if self.results[setup.name] not in ("PASS", "SKIP"):
                    return setup.name
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ctest", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--budget-seconds", type=float, default=300.0)
    parser.add_argument("--per-test-timeout-cap", type=float, default=120.0)
    parser.add_argument("--quarantine-after-timeouts", type=int, default=2)
    parser.add_argument("--select", action="append", default=None)
    parser.add_argument("--exclude", action="append", default=None)
    parser.add_argument("--max-output-bytes", type=int, default=36000)
    parser.add_argument(
        "--list", action="store_true", help="print the selection and exit"
    )
    args = parser.parse_args()

    start = time.monotonic()
    deadline = start + args.budget_seconds

    try:
        all_tests = list_tests(args.ctest, args.test_dir)
    except (subprocess.CalledProcessError, OSError, ValueError) as err:
        log("STRESS ERROR: could not list tests in {}: {}".format(args.test_dir, err))
        return 2

    select = compile_patterns(args.select or DEFAULT_SELECT)
    exclude = compile_patterns(args.exclude or DEFAULT_EXCLUDE)
    chosen = [
        t
        for t in all_tests
        if any_match(select, t.name) and not any_match(exclude, t.name) and not t.disabled
    ]

    log(
        "STRESS selected {} of {} tests, budget {:.0f}s".format(
            len(chosen), len(all_tests), args.budget_seconds
        )
    )
    for test in chosen:
        log("  - {}".format(test.name))
    if args.list:
        return 0
    if not chosen:
        log("STRESS ERROR: no tests matched the selection; test names may have changed")
        return 2

    counts = dict(
        (t.name, dict((k, 0) for k in ("runs", "PASS", "SKIP", "CUT") + BAD))
        for t in chosen
    )
    longest = {}
    failures = []
    shown = set()
    printed = [0]
    quarantined = {}
    timeouts = dict((t.name, 0) for t in chosen)
    setups = Setups(all_tests)
    iteration = 0

    def emit(text):
        if printed[0] + len(text) <= args.max_output_bytes:
            log(text)
            printed[0] += len(text) + 1
            return True
        return False

    while True:
        iteration += 1
        ran = 0
        tally = {}
        for test in chosen:
            if test.name in quarantined:
                continue
            timeout = min(
                test.timeout or args.per_test_timeout_cap, args.per_test_timeout_cap
            )
            remaining = deadline - time.monotonic()
            cut = False
            if remaining < timeout:
                seen = longest.get(test.name)
                if seen is None or remaining < max(3.0 * seen, 5.0):
                    continue
                timeout, cut = remaining, True

            broken_setup = setups.ensure(
                test, iteration, args.per_test_timeout_cap, args.test_dir
            )
            if broken_setup:
                status, rc, output = (
                    "SETUP_FAILED",
                    None,
                    "setup {} failed".format(broken_setup),
                )
            else:
                status, rc, output, took = run_test(test, timeout, args.test_dir)
                if status == "TIMEOUT" and cut:
                    status = "CUT"
                elif status != "TIMEOUT":
                    longest[test.name] = max(took, longest.get(test.name, 0.0))
            ran += 1

            entry = counts[test.name]
            entry["runs"] += 1
            entry[status] += 1
            tally[status] = tally.get(status, 0) + 1

            if status in BAD:
                failures.append(
                    {
                        "test": test.name,
                        "status": status,
                        "iteration": iteration,
                        "rc": rc,
                    }
                )
                key = (test.name, status)
                if key not in shown:
                    shown.add(key)
                    emit(
                        "---- first {} of {} (iteration {}, rc={}) ----\n{}".format(
                            status, test.name, iteration, rc, excerpt(output, 80, 6000)
                        )
                    )
            if status == "TIMEOUT":
                timeouts[test.name] += 1
                if timeouts[test.name] >= args.quarantine_after_timeouts:
                    quarantined[test.name] = iteration

        if not ran:
            iteration -= 1
            break
        emit(
            "iteration {} at {:.0f}s: {}".format(
                iteration,
                time.monotonic() - start,
                ", ".join("{} {}".format(v, k) for k, v in sorted(tally.items())),
            )
        )

    elapsed = time.monotonic() - start
    width = max(len(t.name) for t in chosen)
    log("")
    log(
        "STRESS SUMMARY: {} iterations in {:.0f}s (budget {:.0f}s)".format(
            iteration, elapsed, args.budget_seconds
        )
    )
    columns = ("runs", "PASS", "FAIL", "TIMEOUT", "SIGNAL", "SETUP_FAILED", "SKIP", "CUT")
    labels = ("runs", "pass", "fail", "tmo", "sig", "setup", "skip", "cut")
    row = "  {:<{w}}" + " {:>5}" * len(columns) + "{}"
    log(row.format("test", *(labels + ("",)), w=width))
    for test in chosen:
        c = counts[test.name]
        note = (
            "  (dropped after iteration {})".format(quarantined[test.name])
            if test.name in quarantined
            else ""
        )
        log(row.format(test.name, *(tuple(c[k] for k in columns) + (note,)), w=width))

    result = {
        "iterations": iteration,
        "elapsed_s": round(elapsed, 1),
        "budget_s": args.budget_seconds,
        "counts": counts,
        "quarantined": quarantined,
        "failures": failures[:50],
        "failures_total": len(failures),
    }
    log(
        "STRESS_RESULT_JSON: " + json.dumps(result, sort_keys=True, separators=(",", ":"))
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
