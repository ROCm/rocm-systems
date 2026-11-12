# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""The checked-in generated files match what their generators produce.

hrr_api_args.h, hip_playback_generated.cpp and hip_capture_generated.cpp come
from gen_hrr_api_args.py, and tests/integration/hrr_api_matrix_expectations.h
comes from derive_manifest.py + check_matrix.py --emit-cxx. A change to the
generator, to a manual handler set or to hip_api_trace.hpp that does not
regenerate them leaves a tree that builds and passes the default suite while the
API matrix and the dispatch tables disagree about what an API does at replay.

This regenerates each into a scratch directory and compares it with the
checked-in copy. Nothing is written into the tree. The matrix half needs PyYAML,
as the matrix tools do, and is skipped without it.

Run: python3 -m unittest discover -s projects/hrr/tools/tests
"""

import difflib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent
HRR = TOOLS.parent
CLR_GENERATED = HRR.parent / "clr" / "hipamd" / "src" / "hrr" / "hip_capture_generated.cpp"
MATRIX = TOOLS / "api-matrix"
EXPECTATIONS = HRR / "tests" / "integration" / "hrr_api_matrix_expectations.h"

try:
    import yaml  # noqa: F401
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False


def _run(args):
    proc = subprocess.run([sys.executable] + [str(a) for a in args],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("%s failed (exit %d):\n%s%s" %
                           (args[0], proc.returncode, proc.stdout, proc.stderr))


def _diff(name, committed, regenerated, regen_hint):
    if committed.read_text() == regenerated.read_text():
        return None
    delta = list(difflib.unified_diff(
        committed.read_text().splitlines(), regenerated.read_text().splitlines(),
        "checked in", "regenerated", lineterm="", n=0))
    return "%s is stale; %s\n%s" % (name, regen_hint, "\n".join(delta[:20]))


class GeneratedFilesCurrent(unittest.TestCase):
    def test_generator_outputs_match_the_checked_in_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _run([TOOLS / "gen_hrr_api_args.py", "--silent",
                  "--output-header", out / "hrr_api_args.h",
                  "--output-capture", out / "capture.cpp",
                  "--output-playback", out / "playback.cpp"])
            hint = "run projects/hrr/tools/gen_hrr_api_args.py"
            stale = [d for d in (
                _diff("include/hrr/hrr_api_args.h", HRR / "include" / "hrr" / "hrr_api_args.h",
                      out / "hrr_api_args.h", hint),
                _diff("playback/hip_playback_generated.cpp", HRR / "playback" / "hip_playback_generated.cpp",
                      out / "playback.cpp", hint),
                _diff("clr/hipamd/src/hrr/hip_capture_generated.cpp", CLR_GENERATED,
                      out / "capture.cpp", hint),
            ) if d]
            self.assertEqual(stale, [], "\n\n".join(stale))

    @unittest.skipUnless(HAVE_YAML, "PyYAML is required for the matrix tools")
    def test_api_matrix_header_matches_the_derived_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            # derive_manifest.py exits non-zero when the classification drifted
            # from api_classes_baseline.json, which is itself a stale-file failure.
            _run([MATRIX / "derive_manifest.py", "--quiet", "-o", out / "api_classes.json"])
            _run([MATRIX / "check_matrix.py", "--manifest", out / "api_classes.json",
                  "--emit-cxx", out / "hrr_api_matrix_expectations.h"])
            stale = _diff("tests/integration/hrr_api_matrix_expectations.h", EXPECTATIONS,
                          out / "hrr_api_matrix_expectations.h",
                          "run tools/api-matrix/derive_manifest.py, then check_matrix.py --emit-cxx")
            self.assertIsNone(stale, stale)


if __name__ == "__main__":
    unittest.main()
