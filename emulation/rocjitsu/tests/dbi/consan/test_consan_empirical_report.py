#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

import consan_empirical_report as report


class ConSanEmpiricalReportTest(unittest.TestCase):
    def test_metric_cells_require_qualification_or_explicit_allowlist(self):
        campaign = {
            "workload": "kernel",
            "required_accepted_rounds": 10,
            "summary": {
                "profiles": {
                    "sampled": {
                        "metrics": {
                            "cold:process": {
                                "slowdown": {
                                    "count": 7,
                                    "median": 2.0,
                                    "bootstrap_median_95": {"lower": 1.5, "upper": 2.5},
                                },
                                "timing_ms": {"median": 20.0},
                            }
                        }
                    }
                }
            },
        }
        with self.assertRaisesRegex(report.StudyError, "underqualified metric"):
            report.extract_metric_cells(campaign, set())
        cells = report.extract_metric_cells(campaign, {"kernel/sampled/cold:process"})
        self.assertFalse(cells["sampled"]["cold_process"]["qualified"])
        self.assertEqual(cells["sampled"]["cold_process"]["count"], 7)

    def test_detection_rejects_unreached_final_trial(self):
        not_applicable = [
            {
                "profile": profile,
                "accepted": True,
                "disposition": "not-applicable",
                "reason": "clean admission rejected",
            }
            for profile in report.PROFILE_ORDER
            if profile != "sampled"
        ]
        summary = {
            "schema_version": 2,
            "target": "gfx1201",
            "workload": "kernel",
            "fault": "drop",
            "accepted": True,
            "profiles": not_applicable
            + [
                {
                    "profile": "sampled",
                    "accepted": True,
                    "trials": 32,
                    "attempted_trials": 32,
                    "admitted_trials": 32,
                    "reached_trials": 31,
                    "reach_outcomes": {"reviewed-unconditional-final-isa": 31},
                }
            ],
        }
        manifest = {
            "target": "gfx1201",
            "detection_cases": [{"workload": "kernel", "fault": "drop"}],
        }
        with self.assertRaisesRegex(report.StudyError, "not every final trial reached"):
            report.collect_detection(
                manifest, [("artifact", Path("summary.json"), summary)]
            )

    def test_provenance_signature_includes_shared_runtime_and_device(self):
        files = {
            name: {"sha256": character * 64}
            for name, character in {
                "hook": "a",
                "hip-runtime": "b",
                "hsa-runtime": "c",
                "llvm-readelf": "d",
            }.items()
        }
        provenance = {
            "target": "gfx1201",
            "sources": [{"root": "/checkout/rocm-systems", "head": "1" * 40}],
            "files": files,
            "environment_selectors": {
                "ROCR_VISIBLE_DEVICES": {"present": True, "value": "0"}
            },
            "machine": {
                "selected_kfd_nodes": ["1"],
                "kfd_topology": {
                    "1": {
                        "gpu_id": {"available": True, "value": "9"},
                        "name": {"available": True, "value": "gfx1201"},
                        "properties": {
                            "available": True,
                            "value": "gfx_target_version 120001",
                        },
                    }
                },
                "uname": {"node": "host", "release": "kernel"},
                "amdgpu_module_source_version": {"available": True, "value": "driver"},
            },
            "runtime_tools": {
                name: {"output_sha256": character * 64}
                for name, character in {
                    "amdclang++": "e",
                    "llvm-readelf": "f",
                    "python": "0",
                    "rocminfo": "1",
                }.items()
            },
        }
        signature = report.provenance_signature(provenance)
        self.assertEqual(signature["source_head"], "1" * 40)
        self.assertEqual(signature["hook"], "a" * 64)
        self.assertEqual(signature["selected_topology"]["1"]["name"], "gfx1201")

    def test_workload_identity_rejects_changed_executable(self):
        provenance = {
            "workload": "kernel",
            "files": {"executable": {"sha256": "a" * 64}},
            "sources": [
                {
                    "root": "/checkout/corpus",
                    "head": "1" * 40,
                    "dirty": False,
                }
            ],
            "workload_runtime": {
                "loaded_runtime_libraries": {
                    "hip-runtime": {"sha256": "b" * 64},
                    "hsa-runtime": {"sha256": "c" * 64},
                }
            },
        }
        config = {
            "file_labels": ["executable"],
            "required_runtime_libraries": ["hip-runtime", "hsa-runtime"],
            "source_heads": {"corpus": "1" * 40},
        }
        first = report.workload_provenance_signature(provenance, config)
        provenance["files"]["executable"]["sha256"] = "d" * 64
        second = report.workload_provenance_signature(provenance, config)
        self.assertNotEqual(first, second)

    def test_unknown_metric_and_ragged_markdown_fail_closed(self):
        with self.assertRaisesRegex(report.StudyError, "unmapped empirical metric"):
            report._metric_kind("sustained:workload:kernel")
        with self.assertRaisesRegex(report.StudyError, "ragged row"):
            report._markdown_table(["a", "b"], [["only-a"]])

    def test_structural_metrics_sum_only_modified_code_objects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign_path = root / "workload" / "empirical-campaign" / "campaign.json"
            admission_path = (
                campaign_path.parent / "admission" / "sampled" / "result.json"
            )
            round_path = campaign_path.parent / "rounds" / "round-000" / "round.json"
            result_path = (
                campaign_path.parent
                / "rounds"
                / "round-000"
                / "cold"
                / "sampled"
                / "result.json"
            )
            for path in (admission_path, round_path, result_path):
                path.parent.mkdir(parents=True, exist_ok=True)
            admission_path.write_text(
                json.dumps({"accepted": True, "returncodes": [0]}), encoding="utf-8"
            )
            relative_result = result_path.relative_to(root)
            round_path.write_text(
                json.dumps({"row_results": [str(relative_result)]}), encoding="utf-8"
            )
            result_path.write_text(
                json.dumps(
                    {
                        "accepted": True,
                        "profile": "sampled",
                        "returncodes": [0],
                        "structural_metrics_runs": [
                            {
                                "accepted": True,
                                "total_patch_ms": 7.5,
                                "code_objects": [
                                    {
                                        "modified": False,
                                        "original_bytes": 1000,
                                        "patched_bytes": 1000,
                                        "waitcheck_ms": 1.0,
                                        "inventory_ms": 2.0,
                                    },
                                    {
                                        "modified": True,
                                        "original_bytes": 100,
                                        "patched_bytes": 250,
                                        "waitcheck_ms": 3.0,
                                        "inventory_ms": 4.0,
                                        "resources": {
                                            "descriptor_growth": 5,
                                            "spill": 0,
                                            "unsupported": 1,
                                        },
                                    },
                                ],
                                "process_memory": {
                                    "report_peak_live_bytes": 4096,
                                    "transform_peak_reserved_bytes": 8192,
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            campaign = {
                "workload": "workload",
                "timed_profiles": ["sampled"],
                "required_accepted_rounds": 1,
                "admission": {
                    "rows": {
                        "sampled": {"result": str(admission_path.relative_to(root))}
                    }
                },
            }
            values = report.structural_metrics(campaign_path, campaign)["sampled"]
            self.assertEqual(values["original_bytes"], 100)
            self.assertEqual(values["patched_bytes"], 250)
            self.assertEqual(values["growth_ratio"], 2.5)
            self.assertEqual(values["patch_ms"], 7.5)
            self.assertEqual(values["waitcheck_ms"], 4.0)
            self.assertEqual(values["inventory_ms"], 6.0)
            self.assertEqual(values["clean_gate_runs"], 2)
            self.assertEqual(values["clean_gate_rejections"], 0)
            self.assertEqual(values["unexpected_diagnostics"], 0)


if __name__ == "__main__":
    unittest.main()
