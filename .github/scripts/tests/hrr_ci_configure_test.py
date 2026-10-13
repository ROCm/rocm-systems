# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for hrr_ci_configure.py."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import hrr_ci_configure as cfg

MATRIX = {
    "gfx94x": {
        "linux": {
            "family": "gfx94X-dcgpu",
            "test-runs-on": "gpu-94x",
            "test-runs-on-labels": [{"label": "gpu-94x-a", "count": 1}],
            "fetch-gfx-targets": ["gfx942"],
            "tests_on_trigger": ["presubmit", "postsubmit", "nightly"],
        }
    },
    "gfx90a": {
        "linux": {
            "family": "gfx90a",
            "test-runs-on": "gpu-90a",
            "fetch-gfx-targets": ["gfx90a"],
            "tests_on_trigger": ["postsubmit", "nightly"],
        }
    },
    "gfx110x": {
        "linux": {
            "family": "gfx110X-all",
            "test-runs-on": "gpu-110x",
            "fetch-gfx-targets": ["gfx1100", "gfx1101"],
            "tests_on_trigger": ["nightly"],
        }
    },
    "gfx950": {
        "linux": {
            "family": "gfx950-dcgpu",
            "test-runs-on": "gpu-950-1",
            "test-runs-on-multi-gpu": "gpu-950-8",
            "fetch-gfx-targets": ["gfx950"],
            "tests_on_trigger": ["submodule_bump"],
        }
    },
    "gfx1150": {
        "linux": {
            "family": "gfx1150",
            "test-runs-on": "",
            "tests_on_trigger": ["nightly"],
        }
    },
    "gfx125x": {
        "linux": {
            "family": "gfx125X-dcgpu",
            "test-runs-on": "gpu-125x",
            "fetch-gfx-targets": ["gfx1250"],
            "tests_on_trigger": ["nightly"],
        }
    },
}


def pick_first(labels, _context):
    return labels[0]["label"]


def decide(event, paths, labels=(), is_fork=False, requested=()):
    return cfg.decide(
        event, paths, list(labels), is_fork, list(requested), MATRIX, pick_first
    )


def family_keys(decision):
    return [f["family"] for f in decision.families]


class PathClassificationTest(unittest.TestCase):
    def test_playback_change_is_capture(self):
        self.assertTrue(cfg.touches_capture(["projects/hrr/playback/hip_playback.cpp"]))

    def test_capture_source_is_capture(self):
        self.assertTrue(cfg.touches_capture(["projects/clr/hipamd/src/hrr/hip_capture.cpp"]))

    def test_public_header_is_cpu_only(self):
        paths = ["projects/hip/include/hip/hip_runtime_api.h"]
        self.assertTrue(cfg.touches_cpu(paths))
        self.assertFalse(cfg.touches_capture(paths))

    def test_unit_test_is_cpu_only(self):
        paths = ["projects/hrr/tests/unit/hrr_format_test.cc"]
        self.assertTrue(cfg.touches_cpu(paths))
        self.assertFalse(cfg.touches_capture(paths))

    def test_skills_are_excluded(self):
        paths = ["projects/hrr/skills/hrr/SKILL.md", "projects/hrr/README.md"]
        self.assertFalse(cfg.touches_cpu(paths))

    def test_unrelated_change(self):
        self.assertFalse(cfg.touches_cpu(["projects/rocminfo/rocminfo.cc"]))

    def test_prefix_must_match_whole_component(self):
        self.assertFalse(cfg.touches_cpu(["projects/hrrx/foo.cc"]))


class DecideTest(unittest.TestCase):
    def test_unrelated_pull_request_runs_nothing(self):
        d = decide("pull_request", ["projects/rocminfo/rocminfo.cc"])
        self.assertFalse(d.run_cpu)
        self.assertFalse(d.run_gpu)

    def test_cpu_only_pull_request(self):
        d = decide("pull_request", ["projects/hrr/tests/unit/hrr_format_test.cc"])
        self.assertTrue(d.run_cpu)
        self.assertFalse(d.run_gpu)

    def test_capture_pull_request_uses_presubmit_families(self):
        d = decide("pull_request", ["projects/clr/hipamd/src/hrr/hip_capture.cpp"])
        self.assertTrue(d.run_gpu)
        self.assertEqual(family_keys(d), ["gfx94x"])
        self.assertEqual(d.families[0]["runner"], "gpu-94x-a")
        self.assertEqual(d.families[0]["gfx_targets"], "gfx942")

    def test_rename_out_of_capture_counts(self):
        # The files API lists previous_filename alongside filename.
        d = decide("pull_request", ["projects/other/x.cpp", "projects/hrr/playback/x.cpp"])
        self.assertTrue(d.run_gpu)

    def test_label_requests_gpu_on_unrelated_pull_request(self):
        d = decide("pull_request", ["projects/rocminfo/x.cc"], labels=[cfg.CAPTURE_LABEL])
        self.assertTrue(d.run_cpu)
        self.assertTrue(d.run_gpu)

    def test_family_label_adds_family(self):
        d = decide(
            "pull_request",
            ["projects/hrr/playback/x.cpp"],
            labels=["ci:gfx110X-all", "ci:gfx90a"],
        )
        self.assertEqual(family_keys(d), ["gfx90a", "gfx94x", "gfx110x"])

    def test_fork_skips_gpu(self):
        d = decide("pull_request", ["projects/hrr/playback/x.cpp"], is_fork=True)
        self.assertTrue(d.run_cpu)
        self.assertFalse(d.run_gpu)
        self.assertIn("fork", d.reason)

    def test_push_uses_postsubmit_families(self):
        d = decide("push", ["projects/hrr/playback/x.cpp"])
        self.assertEqual(family_keys(d), ["gfx90a", "gfx94x"])

    def test_push_without_capture_change_has_no_gpu(self):
        d = decide("push", ["projects/hrr/tests/unit/x.cc"])
        self.assertTrue(d.run_cpu)
        self.assertFalse(d.run_gpu)

    def test_unlisted_change_set_runs_everything(self):
        d = decide("push", None)
        self.assertTrue(d.run_cpu)
        self.assertTrue(d.run_gpu)

    def test_nightly_skips_on_request_and_runnerless_families(self):
        d = decide("schedule", [])
        self.assertEqual(family_keys(d), ["gfx90a", "gfx94x", "gfx950", "gfx110x"])

    def test_gfx950_uses_multi_gpu_runner(self):
        d = decide("schedule", [])
        runners = {f["family"]: f["runner"] for f in d.families}
        self.assertEqual(runners["gfx950"], "gpu-950-8")

    def test_gfx950_not_on_pull_requests_by_default(self):
        d = decide("pull_request", ["projects/hrr/playback/x.cpp"])
        self.assertNotIn("gfx950", family_keys(d))

    def test_dispatch_with_families(self):
        d = decide("workflow_dispatch", [], requested=["gfx125x", "gfx1150"])
        # gfx1150 has no runner in this matrix.
        self.assertEqual(family_keys(d), ["gfx125x"])

    def test_unknown_event(self):
        d = decide("merge_group", [])
        self.assertFalse(d.run_cpu)
        self.assertFalse(d.run_gpu)


class OutputTest(unittest.TestCase):
    def test_write_outputs(self):
        import tempfile

        d = cfg.Decision(True, True, "why", [{"family": "gfx94x", "runner": "r", "gfx_targets": "gfx942"}])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            cfg.write_outputs(d, str(out))
            text = out.read_text()
        self.assertIn("run_cpu=true\n", text)
        self.assertIn('families=[{"family":"gfx94x","runner":"r","gfx_targets":"gfx942"}]\n', text)
        self.assertIn("reason=why\n", text)


if __name__ == "__main__":
    unittest.main()
