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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Run the formal models, or skip cleanly when the checkers are not installed.

Neither CBMC nor TLC is a build dependency of rocprofiler-sdk. These checks are a
development aid: on a machine that has the tools they run in seconds and catch semantic
regressions that no GPU-free test can reach; everywhere else they report SKIP and cost
nothing. Exit code 2 is ctest's SKIP_RETURN_CODE for this directory.

  ./run_formal.py cbmc     bounded model checking of the C-level state machines
  ./run_formal.py tlc      exhaustive model checking of the replay isolation window
"""

import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SKIP = 2


def _skip(msg):
    print(f"SKIP: {msg}")
    return SKIP


def _run(cmd, cwd=None):
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def _present(*names):
    """A model is only checked on branches that carry the code it models.

    The callback-removal lineage has no kernel_replay, so the replay models are simply
    absent there rather than duplicated or stubbed.
    """
    missing = [n for n in names if not os.path.isfile(os.path.join(HERE, n))]
    if missing:
        print(f"  [--] skipping {', '.join(missing)}: not present on this branch")
        return False
    return True


def _expect(label, proc, must_succeed):
    """CBMC/TLC return non-zero when a property is violated.

    Several of these models are expected to FAIL: they encode a known gap, and a model
    that stopped failing would mean the gap had silently changed shape. So each case
    declares the verdict it expects rather than simply demanding success.
    """
    ok = (proc.returncode == 0)
    good = (ok == must_succeed)
    want = "hold" if must_succeed else "be violated"
    got = "held" if ok else "was violated"
    print(f"  [{'ok' if good else 'FAIL'}] {label}: expected to {want}, {got}")
    if not good:
        print(proc.stdout[-4000:])
        print(proc.stderr[-2000:])
    return good


def run_cbmc():
    exe = shutil.which("cbmc")
    if not exe:
        return _skip("cbmc not installed (apt-get install cbmc)")

    results = []

    if _present("refcount.c"):
        # Serialization refcount: sound when every caller pairs acquire/release ...
        results.append(
            _expect(
                "refcount, balanced callers",
                _run([exe, "refcount.c", "--unwind", "8", "--bounds-check",
                      "--pointer-check", "--signed-overflow-check"], cwd=HERE),
                must_succeed=True,
            )
        )
        # ... and P3 breaks the moment one is not. Documents where the argument rests.
        results.append(
            _expect(
                "refcount, unbalanced caller (known hazard)",
                _run([exe, "refcount.c", "-DALLOW_UNBALANCED", "--unwind", "8"], cwd=HERE),
                must_succeed=False,
            )
        )

    if _present("perpass.c"):
        # Per-pass isolation holds for the override-aware services; PC sampling breaks it.
        results.append(
            _expect(
                "per-pass isolation incl. PC sampling (known gap)",
                _run([exe, "perpass.c", "--unwind", "8"], cwd=HERE),
                must_succeed=False,
            )
        )

    if not results:
        return _skip("no CBMC models present on this branch")
    return 0 if all(results) else 1


def run_tlc():
    if not _present("ReplayIsolation.tla", "Guarded.cfg", "Bypass.cfg"):
        return _skip("no TLA+ models present on this branch")
    java = shutil.which("java")
    if not java:
        return _skip("java not installed")
    jar = os.environ.get("TLA2TOOLS_JAR", os.path.join(HERE, "tla2tools.jar"))
    if not os.path.isfile(jar):
        return _skip(
            "tla2tools.jar not found; set TLA2TOOLS_JAR or drop the jar in this directory"
        )

    base = [java, "-XX:+UseParallelGC", "-cp", jar, "tlc2.TLC", "-workers", "2"]
    results = [
        _expect(
            "replay isolation window, guards in place",
            _run(base + ["-config", "Guarded.cfg", "ReplayIsolation"], cwd=HERE),
            must_succeed=True,
        ),
        _expect(
            "replay isolation window, graph fast path bypassing the lock (known hazard)",
            _run(base + ["-config", "Bypass.cfg", "ReplayIsolation"], cwd=HERE),
            must_succeed=False,
        ),
    ]
    return 0 if all(results) else 1


def main(argv):
    if len(argv) != 2 or argv[1] not in ("cbmc", "tlc"):
        print(__doc__)
        return 1
    return run_cbmc() if argv[1] == "cbmc" else run_tlc()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
