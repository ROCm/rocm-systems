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

The budget cannot produce a false TIMEOUT: when a test's full timeout does not fit in the
remaining budget it starts with its timeout cut down to what remains, and if that cut timeout
expires it is reported as CUT, which is not a failure. A test that has already been seen to
finish is only started that way if the remainder is comfortably longer than its longest run; a
test that has never finished needs --min-unseen-start seconds, so tests whose timeout exceeds
the whole budget still get run, and a CUT before a test ever finished is flagged as a possible
hang. A test that genuinely times out twice is dropped for the rest of the run so that one hang
cannot consume the budget of every other test. Selected tests that never ran are listed.

Before a timed-out test is killed, the harness records where each of its threads is blocked
(from /proc) and sends it SIGABRT, so that with ROCPROFILER_FAILURE_SIGNAL_HANDLER=1 the output
also shows the stack of the thread that took the signal.

Must stay Python 3.6 compatible: rhel-8.8 CI images ship 3.6 as python3.
"""

from __future__ import print_function

import argparse
import json
import os
import random
import re
import signal
import subprocess
import sys
import time

# The kernel-replay concurrency stress test plus the tests that failed on the four
# remove-callbacks PRs, the per-service queue_hooks unit tests, and the external correlation id
# samples, whose many-threaded dispatches all go through the queue interceptor the PRs change
# and which stalled once in a coverage build. rocprofv3-test-rocshmem-tracing
# is left out: on ubuntu and rhel-8.8 its first run in a job passes and nearly every re-run hangs
# until its timeout, so repeating it measures leftover state from the previous run and spends the
# budget the other tests need.
DEFAULT_SELECT = [
    r"^tests\.integration\.execute\.test-kernel-replay-concurrency$",
    r"^tests\.integration\.execute\..*kernel-replay-local-context",
    r"^rocprofv3-test-kernel-replay(-interval)?-generate$",
    r"^tests\.integration\.execute\.rocprofv3-test-attachment-attach-once-att$",
    r"^tests\.integration\.execute\.rocprofv3-test-hip-streams-per-thread$",
    r"^tests\.integration\.execute\.rocprofv3-test-roctx-pause-resume",
    r"^tests\.integration\.execute\.anytime-",
    # counters_queue_hooks, spm_queue_hooks, pc_sampling_queue_hooks, ThreadTraceQueueHooks
    r"(?i)^unit\.[a-z_]*queue_?hooks[a-z_]*\.",
    r"^unit\.kernel_replay_",
    r"^external-correlation-id-request",
]

# Never select this harness (recursion) or validation tests, whose inputs are the output of an
# execute test from a different iteration.
DEFAULT_EXCLUDE = [r"^stress\.", r"\.validate\.", r"-validate$"]

BAD = ("FAIL", "TIMEOUT", "SIGNAL", "SETUP_FAILED")

# abseil's failure signal handler (enabled via ROCPROFILER_FAILURE_SIGNAL_HANDLER) starts its
# report with this marker; when present it is the part of the output worth keeping.
CRASH_MARKER = "*** SIG"
STATES_MARKER = "---- thread states at timeout ----"

# A test that times out gets SIGABRT first so the fault handler prints the stack of the thread
# that takes it, and SIGKILL once this many seconds pass without it exiting.
ABORT_GRACE_SECONDS = 10.0

# x86_64 numbers of the system calls a hung thread is usually blocked in
SYSCALL_NAMES = {
    "0": "read",
    "7": "poll",
    "16": "ioctl",
    "35": "nanosleep",
    "61": "wait4",
    "202": "futex",
    "230": "clock_nanosleep",
    "232": "epoll_wait",
    "271": "ppoll",
    "281": "epoll_pwait",
}


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


def read_proc(path):
    try:
        with open(path) as ifs:
            return ifs.read().strip()
    except OSError:
        return "?"


def group_pids(pgid):
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        # the fields after the parenthesized command name start with: state ppid pgrp
        fields = read_proc("/proc/{}/stat".format(entry)).rsplit(")", 1)[-1].split()
        if len(fields) > 2 and fields[2] == str(pgid):
            pids.append(int(entry))
    return sorted(pids)


def thread_states(pgid):
    """Where each thread of a hung test is blocked, read from /proc without a debugger."""
    lines = [STATES_MARKER]
    for pid in group_pids(pgid):
        cmdline = read_proc("/proc/{}/cmdline".format(pid)).replace("\0", " ")
        lines.append("pid {}: {}".format(pid, cmdline[:200]))
        try:
            tids = sorted(int(tid) for tid in os.listdir("/proc/{}/task".format(pid)))
        except OSError:
            continue
        for tid in tids:
            base = "/proc/{}/task/{}/".format(pid, tid)
            syscall = read_proc(base + "syscall").split()
            name = SYSCALL_NAMES.get(syscall[0], syscall[0]) if syscall else "?"
            lines.append(
                "  tid {} {:<16} wchan={} syscall={}".format(
                    tid, read_proc(base + "comm"), read_proc(base + "wchan"), name
                )
            )
    return "\n".join(lines)


def signal_group(pgid, signum):
    try:
        os.killpg(pgid, signum)
    except OSError:
        pass


def run_test(test, timeout, default_cwd, dump_on_timeout=False):
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

    states = ""
    try:
        raw, _ = proc.communicate(timeout=timeout)
        status = None
    except subprocess.TimeoutExpired:
        status, raw = "TIMEOUT", None
        if dump_on_timeout:
            states = thread_states(proc.pid)
            signal_group(proc.pid, signal.SIGABRT)
            try:
                raw, _ = proc.communicate(timeout=ABORT_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                pass
        signal_group(proc.pid, signal.SIGKILL)
        if raw is None:
            raw, _ = proc.communicate()

    elapsed = time.monotonic() - start
    output = raw.decode("utf-8", "replace")
    if status is None:
        status = classify(test, proc.returncode, output)
    if states:
        output = "{}\n{}".format(output, states)
    return status, proc.returncode, output, elapsed


def excerpt(output, max_lines, max_bytes):
    """The crash report if the fault handler printed one (with the lines leading up to it),
    otherwise the tail, followed by the thread states recorded at a timeout."""
    states = ""
    sidx = output.find(STATES_MARKER)
    if sidx >= 0:
        output, states = output[:sidx], output[sidx:]
    idx = output.find(CRASH_MARKER)
    if idx >= 0:
        lines = output[:idx].splitlines()[-10:] + output[idx:].splitlines()[:max_lines]
    else:
        lines = output.splitlines()[-max_lines:]
    if states:
        lines += states.splitlines()[:max_lines]
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
    parser.add_argument("--min-unseen-start", type=float, default=30.0)
    parser.add_argument("--quarantine-after-timeouts", type=int, default=2)
    parser.add_argument("--select", action="append", default=None)
    parser.add_argument("--exclude", action="append", default=None)
    parser.add_argument("--max-output-bytes", type=int, default=36000)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="run the selection in a different order every iteration, so a failure that "
        "depends on the test before it gets different neighbours",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="seed for --shuffle (default: random)"
    )
    parser.add_argument(
        "--list", action="store_true", help="print the selection and exit"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="always exit 0; a failing test makes ctest fail the CI step and skip the steps "
        "after it, so findings are read from the recorded output (CDash) instead",
    )
    args = parser.parse_args()
    code = run(args)
    return 0 if args.report_only else code


def run(args):
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

    seed = args.seed if args.seed is not None else random.randrange(1 << 31)
    rng = random.Random(seed)
    log(
        "STRESS selected {} of {} tests, budget {:.0f}s{}".format(
            len(chosen),
            len(all_tests),
            args.budget_seconds,
            ", shuffled with --seed {}".format(seed) if args.shuffle else "",
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
    cut_unseen = []
    timeouts = dict((t.name, 0) for t in chosen)
    setups = Setups(all_tests)
    iteration = 0
    previous = None

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
        order = list(chosen)
        if args.shuffle:
            rng.shuffle(order)
        for test in order:
            if test.name in quarantined:
                continue
            timeout = min(
                test.timeout or args.per_test_timeout_cap, args.per_test_timeout_cap
            )
            remaining = deadline - time.monotonic()
            seen = longest.get(test.name)
            cut = False
            if remaining < timeout:
                needed = args.min_unseen_start if seen is None else max(3.0 * seen, 5.0)
                if remaining < needed:
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
                status, rc, output, took = run_test(
                    test, timeout, args.test_dir, dump_on_timeout=not cut
                )
                if status == "TIMEOUT" and cut:
                    status = "CUT"
                    if seen is None and test.name not in cut_unseen:
                        cut_unseen.append(test.name)
                        emit(
                            "---- {} was cut after {:.0f}s and has never finished; possible "
                            "hang ----".format(test.name, took)
                        )
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
                        "after": previous,
                    }
                )
                key = (test.name, status)
                if key not in shown:
                    shown.add(key)
                    emit(
                        "---- first {} of {} (iteration {}, rc={}, after {}) ----\n{}".format(
                            status,
                            test.name,
                            iteration,
                            rc,
                            previous,
                            excerpt(output, 80, 6000),
                        )
                    )
            previous = test.name
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

    never_ran = [t.name for t in chosen if counts[t.name]["runs"] == 0]
    if never_ran:
        log("{} selected tests never ran:".format(len(never_ran)))
        for name in never_ran:
            log("  - {}".format(name))
    for name in cut_unseen:
        log("cut before it ever finished (possible hang): {}".format(name))

    result = {
        "iterations": iteration,
        "elapsed_s": round(elapsed, 1),
        "budget_s": args.budget_seconds,
        "seed": seed if args.shuffle else None,
        "counts": counts,
        "never_ran": never_ran,
        "cut_unseen": cut_unseen,
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
