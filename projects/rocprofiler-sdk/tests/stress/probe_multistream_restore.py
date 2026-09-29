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

"""Does a context save/restore explain multistream's one-wave dispatches counting two waves?

counter-collection.test_counter_values fails about once in a few hundred runs, always the same
way: one of the 10,000 one-wave `add` dispatches in tests/bin/multistream reports SQ_WAVES_sum
== 2. The app's own overlap check never fires. Hypothesis: its 100 streams oversubscribe the
hardware queues, the scheduler preempts the running wave (context save/restore), and SQ_WAVES
counts the restored wave a second time.

This runs multistream under rocprofv3 repeatedly for a wall-clock budget, collecting SQ_WAVES
with SQ_WAVES_RESTORED and SQ_WAVES_SAVED for every dispatch, and classifies each `add`
dispatch:

  normal                     SQ_WAVES == 1, nothing saved or restored
  inflated_with_restore      SQ_WAVES >= 2 and SQ_WAVES_RESTORED >= 1  (hypothesis holds)
  inflated_without_restore   SQ_WAVES >= 2 and SQ_WAVES_RESTORED == 0  (a real overlap)
  restore_without_inflation  SQ_WAVES == 1 but a wave was saved or restored  (refutes it)
  zero_waves                 SQ_WAVES == 0  (an undercount)
  missing_counter            a counter has no row for this dispatch

All three counters must come from the same pass. Giving rocprofv3 one --pmc per counter makes
it replay the whole application once per counter, each run writing its own pass_N/ output, so
the counters of one dispatch would come from different runs and could not be compared.

It exits non-zero whenever anything other than `normal` is seen. Under --report-only, as
registered in CI, it always exits 0 so it cannot fail the CI step and skip the steps after it;
the verdict is read from the PROBE_RESULT_JSON line in the recorded output instead.

Must stay Python 3.6 compatible: rhel-8.8 CI images ship 3.6 as python3.
"""

from __future__ import print_function

import argparse
import collections
import csv
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

COUNTERS = ("SQ_WAVES", "SQ_WAVES_RESTORED", "SQ_WAVES_SAVED")
CLASSES = (
    "normal",
    "inflated_with_restore",
    "inflated_without_restore",
    "restore_without_inflation",
    "zero_waves",
    "missing_counter",
)
# the app's only kernel; the runtime's copy kernels also appear in the output and are ignored
KERNEL_RE = re.compile(r"(^|[^A-Za-z0-9_])add($|[^A-Za-z0-9_])")


def log(msg):
    print(msg)
    sys.stdout.flush()


def find_column(header, *words):
    """Index of the first column whose lower-cased name contains every word."""
    for idx, name in enumerate(header):
        low = name.lower()
        if all(w in low for w in words):
            return idx
    return None


def read_counters(csv_path):
    """dispatch_id -> {"kernel": name, counter: summed value}, summing dimension rows."""
    with open(csv_path) as ifs:
        rows = csv.reader(ifs)
        header = next(rows)
        col = {
            "dispatch": find_column(header, "dispatch", "id"),
            "kernel": find_column(header, "kernel", "name"),
            "counter": find_column(header, "counter", "name"),
            "value": find_column(header, "counter", "value"),
        }
        missing = [k for k, v in col.items() if v is None]
        if missing:
            raise ValueError("columns {} not found in {}".format(missing, header))
        table = collections.defaultdict(lambda: collections.defaultdict(float))
        for row in rows:
            if not row:
                continue
            entry = table[int(row[col["dispatch"]])]
            entry["kernel"] = row[col["kernel"]]
            entry[row[col["counter"]]] += float(row[col["value"]])
    return header, table


def describe(csv_path, run_dir, header, add_entries):
    """Print what the first parsed run contained, so a misread shows up in the log."""
    log("first run: {}".format(os.path.relpath(csv_path, run_dir)))
    log("  header: {}".format(",".join(header)))
    names = sorted(set(k for _, e in add_entries for k in e if k != "kernel"))
    for name in names:
        values = [e[name] for _, e in add_entries if name in e]
        log(
            "  {}: {} of {} add dispatches, sum {:g}, min {:g}, max {:g}".format(
                name, len(values), len(add_entries), sum(values), min(values), max(values)
            )
        )
    absent = [name for name in COUNTERS if name not in names]
    if absent:
        log("  no rows at all for: {}".format(", ".join(absent)))


def classify(entry):
    if any(name not in entry for name in COUNTERS):
        return "missing_counter"
    waves = int(round(entry.get("SQ_WAVES", 0.0)))
    restored = int(round(entry.get("SQ_WAVES_RESTORED", 0.0)))
    saved = int(round(entry.get("SQ_WAVES_SAVED", 0.0)))
    if waves == 0:
        return "zero_waves"
    if waves >= 2:
        return "inflated_with_restore" if restored >= 1 else "inflated_without_restore"
    if restored or saved:
        return "restore_without_inflation"
    return "normal"


def run_once(args, run_dir, timeout):
    cmd = [args.rocprofv3, "--pmc"] + list(COUNTERS)
    cmd += ["--output-format", "csv", "-d", run_dir, "-o", "out", "--", args.app]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True
    )
    try:
        raw, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        raw, _ = proc.communicate()
        return "TIMEOUT", raw.decode("utf-8", "replace")
    output = raw.decode("utf-8", "replace")
    return ("OK" if proc.returncode == 0 else "RC={}".format(proc.returncode)), output


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rocprofv3", required=True)
    parser.add_argument("--app", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--budget-seconds", type=float, default=240.0)
    parser.add_argument("--run-timeout", type=float, default=90.0)
    parser.add_argument("--max-events", type=int, default=40)
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
    if not os.path.isdir(args.workdir):
        os.makedirs(args.workdir)
    start = time.monotonic()
    deadline = start + args.budget_seconds
    totals = dict((k, 0) for k in CLASSES)
    events = []
    run_failures = []
    runs = 0
    longest = None
    described = False

    log(
        "PROBE rocprofv3 --pmc {} on {} for {:.0f}s".format(
            " ".join(COUNTERS), os.path.basename(args.app), args.budget_seconds
        )
    )
    while True:
        remaining = deadline - time.monotonic()
        # never start a run the budget would cut off; the first run always goes ahead
        if longest is not None and remaining < 1.5 * longest:
            break
        if remaining <= 0:
            break
        runs += 1
        run_dir = os.path.join(args.workdir, "run-{}".format(runs))
        shutil.rmtree(run_dir, ignore_errors=True)
        t0 = time.monotonic()
        status, output = run_once(
            args, run_dir, min(args.run_timeout, max(remaining, 5.0))
        )
        took = time.monotonic() - t0
        longest = took if longest is None else max(longest, took)

        found = sorted(
            glob.glob(
                os.path.join(run_dir, "**", "*counter_collection.csv"), recursive=True
            )
        )
        if status == "OK" and len(found) > 1:
            status = "{} counter CSVs, expected one: {}".format(
                len(found), ", ".join(os.path.relpath(f, run_dir) for f in found)
            )
        if status != "OK" or not found:
            run_failures.append({"run": runs, "status": status, "csv": bool(found)})
            if len(run_failures) <= 2:
                tail = "\n".join(output.splitlines()[-40:])
                log(
                    "---- run {} failed ({}; csv found: {}) ----\n{}".format(
                        runs, status, bool(found), tail
                    )
                )
            if len(run_failures) >= 3 and runs == len(run_failures):
                log("PROBE ERROR: every rocprofv3 run so far failed; stopping early")
                break
            shutil.rmtree(run_dir, ignore_errors=True)
            continue

        try:
            header, table = read_counters(found[0])
        except (ValueError, OSError, StopIteration) as err:
            run_failures.append(
                {"run": runs, "status": "PARSE: {}".format(err), "csv": True}
            )
            log("---- run {} output could not be parsed: {} ----".format(runs, err))
            shutil.rmtree(run_dir, ignore_errors=True)
            continue

        add_entries = [
            (dispatch_id, entry)
            for dispatch_id, entry in sorted(table.items())
            if KERNEL_RE.search(entry.get("kernel", ""))
        ]
        if not described and add_entries:
            describe(found[0], run_dir, header, add_entries)
            described = True

        counts = dict((k, 0) for k in CLASSES)
        for dispatch_id, entry in add_entries:
            kind = classify(entry)
            counts[kind] += 1
            if kind != "normal" and len(events) < args.max_events:
                event = {"run": runs, "dispatch_id": dispatch_id, "class": kind}
                for name in COUNTERS:
                    event[name] = entry.get(name)
                events.append(event)
        for k in CLASSES:
            totals[k] += counts[k]
        log(
            "run {} ({:.1f}s): {} add dispatches{}".format(
                runs,
                took,
                sum(counts.values()),
                "".join(", {} {}".format(counts[k], k) for k in CLASSES[1:] if counts[k]),
            )
        )
        shutil.rmtree(run_dir, ignore_errors=True)

    elapsed = time.monotonic() - start
    log("")
    log("PROBE SUMMARY: {} runs in {:.0f}s".format(runs, elapsed))
    for k in CLASSES:
        log("  {:<27} {}".format(k, totals[k]))
    if run_failures:
        log("  runs that produced no usable output: {}".format(len(run_failures)))
    for ev in events:
        values = dict(
            (name, "-" if ev[name] is None else "{:g}".format(ev[name]))
            for name in COUNTERS
        )
        log(
            "  event run={} dispatch={} {}: SQ_WAVES={} RESTORED={} SAVED={}".format(
                ev["run"],
                ev["dispatch_id"],
                ev["class"],
                values["SQ_WAVES"],
                values["SQ_WAVES_RESTORED"],
                values["SQ_WAVES_SAVED"],
            )
        )
    result = {
        "runs": runs,
        "elapsed_s": round(elapsed, 1),
        "totals": totals,
        "events": events,
        "run_failures": run_failures,
    }
    log("PROBE_RESULT_JSON: " + json.dumps(result, sort_keys=True, separators=(",", ":")))

    interesting = sum(totals[k] for k in CLASSES[1:])
    if runs == len(run_failures):
        return 2
    return 1 if (interesting or run_failures) else 0


if __name__ == "__main__":
    sys.exit(main())
