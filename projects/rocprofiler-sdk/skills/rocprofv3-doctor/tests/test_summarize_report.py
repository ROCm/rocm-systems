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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


"""GPU-free unit tests for the rocprofv3 --doctor report summary script."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_report.py"


def load_script():
    spec = importlib.util.spec_from_file_location("summarize_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


summarize = load_script()


def check(check_id, status, severity="error", detail="", remediation="", data=None):
    return {
        "id": check_id,
        "group": check_id.split(".")[0],
        "title": check_id + " title",
        "severity": severity,
        "probe": "passive",
        "depends": [],
        "status": status,
        "detail": detail,
        "remediation": remediation,
        "data": data or {},
    }


def report(checks, **extra):
    summary = {key: 0 for key in ("pass", "warn", "fail", "skip", "error")}
    for entry in checks:
        summary[entry["status"]] += 1
    summary["total"] = len(checks)
    body = {
        "schema_version": 1,
        "tool_version": "1.4.0",
        "rocm_version": "",
        "rocm_root": "/opt/rocm/core-10.0",
        "rocm_root_source": "ROCM_PATH",
        "install_kind": "therock-package",
        "summary": summary,
        "checks": checks,
    }
    body.update(extra)
    return body


def run(body, *args):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "doctor.json"
        if isinstance(body, str):
            path.write_text(body)
        else:
            path.write_text(json.dumps(body))
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = summarize.main([str(path)] + list(args))
        return code, stdout.getvalue(), stderr.getvalue()


KFD_CHAIN = [
    check(
        "driver.kfd-device",
        "fail",
        detail="/dev/kfd does not exist",
        remediation="sudo modprobe amdgpu",
    ),
    check(
        "driver.kfd-readable",
        "skip",
        data={"skipped_because_of": "driver.kfd-device"},
    ),
    check(
        "counters.cap-perfmon",
        "skip",
        severity="info",
        data={"skipped_because_of": "driver.kfd-readable"},
    ),
    check("install.rocm-root", "pass"),
]


class SummarizeReportTest(unittest.TestCase):
    def test_header_names_root_and_install_kind(self):
        code, out, _ = run(report(KFD_CHAIN))
        self.assertEqual(code, 0)
        self.assertIn("`/opt/rocm/core-10.0` (found via ROCM_PATH; therock-package)", out)
        self.assertIn("1 failed, 2 skipped", out)

    def test_skip_chain_is_folded_into_its_root_cause(self):
        _, out, _ = run(report(KFD_CHAIN))
        causes = out.split("## Root causes to fix")[1].split("##")[0]
        self.assertIn("[FAIL] driver.kfd-device", causes)
        self.assertIn("Blocks: counters.cap-perfmon, driver.kfd-readable", causes)
        self.assertNotIn("[SKIP]", out)
        self.assertIn("sudo modprobe amdgpu", causes)

    def test_failures_come_before_warnings(self):
        checks = [
            check("tools.rocpd", "warn", severity="warning", detail="rocpd missing"),
            check("install.sdk-library", "fail", detail="sdk missing"),
        ]
        _, out, _ = run(report(checks))
        self.assertLess(out.index("install.sdk-library"), out.index("tools.rocpd"))

    def test_info_warning_is_a_note_not_a_cause(self):
        checks = [
            check(
                "runtime.att-decoder-library",
                "warn",
                severity="info",
                detail="decoder not found",
            )
        ]
        _, out, _ = run(report(checks))
        self.assertIn("None: no check failed", out)
        notes = out.split("## Notes (informational)")[1]
        self.assertIn("runtime.att-decoder-library: decoder not found", notes)

    def test_doctor_error_is_kept_apart_from_system_findings(self):
        checks = [check("python.pandas", "error", detail="the check itself raised")]
        _, out, _ = run(report(checks))
        self.assertIn("None: no check failed", out)
        self.assertIn("## Doctor errors (not findings about this system)", out)
        self.assertIn("rocprofv3 --doctor bug", out)

    def test_diagnoses_show_confidence_and_evidence(self):
        diagnosis = {
            "id": "hsa.out-of-resources",
            "confidence": "high",
            "summary": "HSA reported OUT_OF_RESOURCES (likely cause: /dev/kfd)",
            "evidence": "HSA_STATUS_ERROR_OUT_OF_RESOURCES: ...",
        }
        uncertain = dict(diagnosis, confidence="possible", summary="maybe memory")
        checks = [
            check(
                "counters.avail-enumeration",
                "warn",
                severity="warning",
                data={"diagnoses": [diagnosis, uncertain]},
            )
        ]
        _, out, _ = run(report(checks))
        self.assertIn("Diagnosis (likely): HSA reported", out)
        self.assertIn("Diagnosis (possible): maybe memory", out)
        self.assertIn("Evidence: `HSA_STATUS_ERROR_OUT_OF_RESOURCES: ...`", out)

    def test_excluded_prerequisite_is_reported_as_not_verified(self):
        checks = [
            check(
                "runtime.libraries-load",
                "skip",
                data={"skipped_because_of": "install.sdk-library"},
            )
        ]
        _, out, _ = run(report(checks))
        gaps = out.split("## Not verified by this run")[1]
        self.assertIn(
            "runtime.libraries-load not checked: their prerequisite "
            "install.sdk-library was excluded",
            gaps,
        )

    def test_smoke_coverage_is_stated_either_way(self):
        _, without, _ = run(report(KFD_CHAIN))
        self.assertIn("rocprofv3 itself was not started", without)
        with_smoke = KFD_CHAIN + [check("smoke.rocprofv3-launcher", "pass")]
        _, out, _ = run(report(with_smoke))
        self.assertIn("Kernel tracing itself is not verified", out)

    def test_output_file_matches_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "summary.md"
            code, out, _ = run(report(KFD_CHAIN), "--output", str(target))
            self.assertEqual(code, 0)
            self.assertEqual(target.read_text(), out)

    def test_unreadable_inputs_exit_2_with_a_message(self):
        cases = {
            "not json": "is not JSON",
            json.dumps({"hello": 1}): "is not a rocprofv3 --doctor report",
            json.dumps(report(KFD_CHAIN, schema_version=2)): "schema_version 2",
        }
        for text, message in cases.items():
            code, out, err = run(text)
            self.assertEqual(code, 2, text)
            self.assertEqual(out, "")
            self.assertIn(message, err)

    def test_missing_file_exits_2(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = summarize.main(["/nonexistent/doctor.json"])
        self.assertEqual(code, 2)
        self.assertIn("cannot read", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
