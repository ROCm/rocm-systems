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

"""GPU-free unit tests for the PC sampling analysis script."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_pc_sampling.py"


def load_module():
    spec = importlib.util.spec_from_file_location("analyze_pc_sampling", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analyze = load_module()


def _stochastic_json(*, code_object_id: int = 0, wave_issued: bool = False) -> dict:
    return {
        "rocprofiler-sdk-tool": [
            {
                "kernel_symbols": [
                    {
                        "kernel_name": "my_kernel",
                        "formatted_kernel_name": "my_kernel(int*)",
                    }
                ],
                "strings": {
                    "pc_sample_instructions": ["s_waitcnt vmcnt(0)"],
                    "pc_sample_comments": ["/tmp/example.cpp:42"],
                },
                "buffer_records": {
                    "kernel_dispatch": [
                        {
                            "dispatch_info": {
                                "dispatch_id": 1,
                                "kernel_id": 0,
                            }
                        }
                    ],
                    "pc_sample_stochastic": [
                        {
                            "inst_index": 0,
                            "record": {
                                "dispatch_id": 1,
                                "wave_issued": wave_issued,
                                "inst_type": "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST",
                                "snapshot": {
                                    "stall_reason": (
                                        "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_WAITCNT"
                                    )
                                },
                                "pc": {
                                    "code_object_id": code_object_id,
                                    "code_object_offset": 4096,
                                },
                            },
                        }
                    ],
                },
            }
        ]
    }


def _host_trap_json() -> dict:
    return {
        "rocprofiler-sdk-tool": [
            {
                "kernel_symbols": [],
                "strings": {
                    "pc_sample_instructions": ["v_add_f32 v0, v1, v2"],
                    "pc_sample_comments": ["/tmp/example.cpp:7"],
                },
                "buffer_records": {
                    "pc_sample_host_trap": [
                        {
                            "inst_index": 0,
                            "record": {
                                "dispatch_id": 5,
                                "pc": {
                                    "code_object_id": 1,
                                    "code_object_offset": 8192,
                                },
                            },
                        }
                    ],
                },
            }
        ]
    }


class AnalyzePcSamplingTests(unittest.TestCase):
    def test_parse_code_object_id_zero_is_preserved(self):
        self.assertEqual(analyze.parse_code_object_id({"code_object_id": 0}), 0)
        self.assertIsNone(analyze.parse_code_object_id({}))

    def test_json_sample_preserves_code_object_id_zero(self):
        tool = _stochastic_json(code_object_id=0)["rocprofiler-sdk-tool"][0]
        sample = next(analyze.iter_json_samples(tool, "stochastic"))
        self.assertEqual(sample.code_object_id, 0)

    def test_unmapped_dispatch_ids_warn_in_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out_results.json"
            path.write_text(json.dumps(_host_trap_json()), encoding="utf-8")
            report = analyze.parse_inputs(path)
            text = analyze.render_report(report, top_n=5, kernel_filter=None)

        self.assertEqual(report.unmapped_dispatch_ids, {5})
        self.assertIn("<unknown dispatch 5>", report.kernels)
        self.assertIn("had no kernel trace mapping", text)

    def test_stochastic_report_includes_percentages_and_stall_reasons(self):
        data = _stochastic_json(wave_issued=False)
        data["rocprofiler-sdk-tool"][0]["buffer_records"]["pc_sample_stochastic"].append(
            {
                "inst_index": 0,
                "record": {
                    "dispatch_id": 1,
                    "wave_issued": True,
                    "inst_type": "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_VALU",
                    "snapshot": {},
                    "pc": {"code_object_id": 0, "code_object_offset": 4096},
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out_results.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            report = analyze.parse_inputs(path)
            text = analyze.render_report(report, top_n=5, kernel_filter=None)

        self.assertIn(
            "| Source | Instruction | Count | % Samples | Issued | % Issued |", text
        )
        self.assertIn("/tmp/example.cpp:42", text)
        self.assertIn("WAITCNT", text)
        self.assertIn("50.0%", text)

    def test_host_trap_report_uses_source_first_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out_results.json"
            path.write_text(json.dumps(_host_trap_json()), encoding="utf-8")
            report = analyze.parse_inputs(path)
            text = analyze.render_report(report, top_n=5, kernel_filter=None)

        self.assertEqual(report.method, "host_trap")
        self.assertIn("| Rank | Source | Instruction | Count | % Samples |", text)
        self.assertIn("/tmp/example.cpp:7", text)

    def test_undecodable_samples_are_noted(self):
        data = _stochastic_json()
        data["rocprofiler-sdk-tool"][0]["buffer_records"]["pc_sample_stochastic"][0][
            "inst_index"
        ] = -1
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out_results.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            report = analyze.parse_inputs(path)
            text = analyze.render_report(report, top_n=5, kernel_filter=None)

        self.assertEqual(report.undecodable_samples, 1)
        self.assertIn("could not be decoded", text)
        self.assertIn("unrecognized code object", text)


if __name__ == "__main__":
    unittest.main()
