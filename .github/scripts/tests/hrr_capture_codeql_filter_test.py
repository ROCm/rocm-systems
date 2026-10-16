# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for hrr_capture_codeql_filter.py.

unittest rather than pytest-only, so the CodeQL job can run them in its ROCm
image, which has Python but no pytest.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import hrr_capture_codeql_filter as capture_filter


def result(rule, uri, line=1):
    return {
        "ruleId": rule,
        "message": {"text": f"{rule} here"},
        "locations": [{"physicalLocation": {
            "artifactLocation": {"uri": uri, "uriBaseId": "%SRCROOT%"},
            "region": {"startLine": line}}}],
    }


# The shape of the real run on develop: the rocclr header alerts the capture
# files pull in, next to one capture alert of the kind #12253 fixed.
WRITER = "projects/clr/hipamd/src/hrr/hip_capture_writer.cpp"
ROCCLR = [
    result("cpp/integer-multiplication-cast-to-long",
           "projects/clr/rocclr/elf/elfio/elfio.hpp", 594),
    result("cpp/incorrect-allocation-error-handling",
           "projects/clr/rocclr/device/device.hpp", 1434),
    result("cpp/suspicious-sizeof", "projects/clr/rocclr/platform/command.hpp", 1990),
]


def sarif(results):
    return {"version": "2.1.0", "runs": [{"tool": {"driver": {"name": "CodeQL"}},
                                          "results": results}]}


class FilterTest(unittest.TestCase):
    def test_rocclr_header_results_are_dropped(self):
        doc = sarif(list(ROCCLR))
        self.assertEqual(capture_filter.filter_sarif(doc), [])
        self.assertEqual(doc["runs"][0]["results"], [])

    def test_capture_results_are_kept(self):
        capture = result("cpp/world-writable-file-creation", WRITER, 450)
        doc = sarif(ROCCLR + [capture])
        self.assertEqual(capture_filter.filter_sarif(doc), [capture])
        self.assertEqual(doc["runs"][0]["results"], [capture])

    def test_a_directory_sharing_the_prefix_is_not_capture(self):
        doc = sarif([result("cpp/x", "projects/clr/hipamd/src/hrr_other/a.cpp"),
                     result("cpp/x", "projects/clr/hipamd/src/hip_module.cpp")])
        self.assertEqual(capture_filter.filter_sarif(doc), [])

    def test_absolute_uri_inside_capture_is_kept(self):
        capture = result("cpp/x", "file:///__w/rocm-systems/rocm-systems/" + WRITER)
        self.assertEqual(capture_filter.filter_sarif(sarif([capture])), [capture])

    def test_result_without_location_is_kept(self):
        orphan = {"ruleId": "cpp/x", "message": {"text": "no location"}}
        self.assertEqual(capture_filter.filter_sarif(sarif([orphan])), [orphan])

    def test_every_run_is_filtered(self):
        capture = result("cpp/x", WRITER)
        doc = {"runs": [{"results": list(ROCCLR)}, {"results": [capture]}]}
        self.assertEqual(capture_filter.filter_sarif(doc), [capture])
        self.assertEqual(doc["runs"][0]["results"], [])


class MainTest(unittest.TestCase):
    def run_main(self, results):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp, "in.sarif"), Path(tmp, "out.sarif")
            src.write_text(json.dumps(sarif(results)), encoding="utf-8")
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = capture_filter.main(["filter", str(src), str(dst)])
            return rc, out.getvalue(), err.getvalue(), json.loads(dst.read_text())

    def test_prints_zero_when_only_headers_alert(self):
        rc, out, err, written = self.run_main(list(ROCCLR))
        self.assertEqual((rc, out.strip(), err), (0, "0", ""))
        self.assertEqual(written["runs"][0]["results"], [])

    def test_prints_count_and_lists_capture_alerts(self):
        capture = result("cpp/world-writable-file-creation", WRITER, 450)
        rc, out, err, written = self.run_main(ROCCLR + [capture])
        self.assertEqual((rc, out.strip()), (0, "1"))
        self.assertIn(f"{WRITER}:450: cpp/world-writable-file-creation", err)
        self.assertEqual(written["runs"][0]["results"], [capture])
        # The rest of the document is untouched, so upload-sarif still takes it.
        self.assertEqual(written["runs"][0]["tool"]["driver"]["name"], "CodeQL")

    def test_wrong_arguments_are_refused(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(capture_filter.main(["filter", "only-one"]), 2)


if __name__ == "__main__":
    unittest.main()
