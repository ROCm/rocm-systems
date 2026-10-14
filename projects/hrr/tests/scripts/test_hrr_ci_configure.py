#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for hrr_ci_configure.py."""

import copy
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hrr_ci_configure as cfg  # noqa: E402

MATRIX = {
    "gfx94x": {
        "linux": {
            "family": "gfx94X-dcgpu",
            "test-runs-on": "gpu-94x",
            "test-runs-on-labels": [{"label": "gpu-94x-a", "count": 1}],
            "fetch-gfx-targets": ["gfx942"],
        }
    },
    "gfx90a": {
        "linux": {
            "family": "gfx90a",
            "test-runs-on": "gpu-90a",
            "fetch-gfx-targets": ["gfx90a"],
        }
    },
    "gfx103x": {
        "linux": {
            "family": "gfx103X-all",
            "test-runs-on": "gpu-103x",
            "fetch-gfx-targets": ["gfx1030"],
        }
    },
    "gfx110x": {
        "linux": {
            "family": "gfx110X-all",
            "test-runs-on": "gpu-110x",
            "fetch-gfx-targets": ["gfx1100", "gfx1101"],
        }
    },
    "gfx950": {
        "linux": {
            "family": "gfx950-dcgpu",
            "test-runs-on": "gpu-950-1",
            "test-runs-on-multi-gpu": "gpu-950-8",
            "fetch-gfx-targets": ["gfx950"],
        }
    },
    "gfx1150": {
        "linux": {
            "family": "gfx1150",
            "test-runs-on": "",
        }
    },
    "gfx125x": {
        "linux": {
            "family": "gfx125X-dcgpu",
            "test-runs-on": "gpu-125x",
            "fetch-gfx-targets": ["gfx1250"],
        }
    },
}

REPO_CONFIG = yaml.safe_load(cfg.DEFAULT_CONFIG.read_text(encoding="utf-8"))

# The repository's paths with families that exercise every option. gfx103x is
# in MATRIX but not here.
FAMILIES = {
    "gfx90a": {},
    "gfx94x": None,
    "gfx950": {"multi_gpu": True},
    "gfx110x": {"triggers": ["schedule", "workflow_dispatch"]},
    "gfx1150": {},
    "gfx125x": {"enabled": False},
}


def config_data(**changes):
    data = copy.deepcopy(REPO_CONFIG)
    data["families"] = copy.deepcopy(FAMILIES)
    data.update(changes)
    return data


CONFIG = cfg.parse_config(config_data())
REPO = cfg.load_config(cfg.DEFAULT_CONFIG)


def pick_first(labels, _context):
    return labels[0]["label"]


def decide(event, paths, labels=(), is_fork=False, requested=(), config=CONFIG):
    return cfg.decide(
        event, paths, list(labels), is_fork, list(requested), config, MATRIX, pick_first
    )


def family_keys(decision):
    return [f["family"] for f in decision.families]


class RepoConfigTest(unittest.TestCase):
    def test_every_family_runs_on_every_event_by_default(self):
        for key, family in REPO.families.items():
            self.assertTrue(family.enabled, key)
            self.assertEqual(family.triggers, cfg.EVENTS, key)

    def test_gfx950_takes_the_multi_gpu_runner(self):
        self.assertTrue(REPO.families["gfx950"].multi_gpu)


class PathClassificationTest(unittest.TestCase):
    def test_playback_change_is_capture(self):
        self.assertTrue(cfg.touches_capture(["projects/hrr/playback/hip_playback.cpp"], REPO))

    def test_capture_source_is_capture(self):
        self.assertTrue(
            cfg.touches_capture(["projects/clr/hipamd/src/hrr/hip_capture.cpp"], REPO)
        )

    def test_ci_config_is_capture(self):
        self.assertTrue(cfg.touches_capture(["projects/hrr/tests/config/ci.yaml"], REPO))

    def test_public_header_is_cpu_only(self):
        paths = ["projects/hip/include/hip/hip_runtime_api.h"]
        self.assertTrue(cfg.touches_cpu(paths, REPO))
        self.assertFalse(cfg.touches_capture(paths, REPO))

    def test_unit_test_is_cpu_only(self):
        paths = ["projects/hrr/tests/unit/hrr_format_test.cc"]
        self.assertTrue(cfg.touches_cpu(paths, REPO))
        self.assertFalse(cfg.touches_capture(paths, REPO))

    def test_skills_are_excluded(self):
        paths = ["projects/hrr/skills/hrr/SKILL.md", "projects/hrr/README.md"]
        self.assertFalse(cfg.touches_cpu(paths, REPO))

    def test_unrelated_change(self):
        self.assertFalse(cfg.touches_cpu(["projects/rocminfo/rocminfo.cc"], REPO))

    def test_prefix_must_match_whole_component(self):
        self.assertFalse(cfg.touches_cpu(["projects/hrrx/foo.cc"], REPO))


class ConfigValidationTest(unittest.TestCase):
    def assertRejected(self, data, message):
        with self.assertRaises(cfg.ConfigError) as caught:
            cfg.parse_config(data)
        self.assertIn(message, str(caught.exception))

    def test_unknown_top_level_key(self):
        self.assertRejected(config_data(gpu_families=[]), "unknown key gpu_families")

    def test_missing_key(self):
        data = config_data()
        del data["capture_label"]
        self.assertRejected(data, "missing capture_label")

    def test_wrong_version(self):
        self.assertRejected(config_data(version=2), "version: expected 1")

    def test_unknown_family_option(self):
        families = {"gfx94x": {"runs_on": "x"}}
        self.assertRejected(config_data(families=families), "families.gfx94x: unknown key runs_on")

    def test_unknown_trigger(self):
        families = {"gfx94x": {"triggers": ["nightly"]}}
        self.assertRejected(config_data(families=families), "unknown trigger 'nightly'")

    def test_enabled_must_be_a_bool(self):
        families = {"gfx94x": {"enabled": "no"}}
        self.assertRejected(config_data(families=families), "families.gfx94x.enabled")

    def test_empty_path_list(self):
        data = config_data()
        data["paths"]["capture"] = []
        self.assertRejected(data, "paths.capture: expected a non-empty list")

    def test_no_families(self):
        self.assertRejected(config_data(families={}), "families: expected a mapping")

    def test_family_missing_from_matrix(self):
        config = cfg.parse_config(config_data(families={"gfx94x": {}, "gfx999": {}}))
        with self.assertRaises(cfg.ConfigError) as caught:
            cfg.check_families(config, MATRIX)
        self.assertIn("gfx999 not in TheRock's family matrix", str(caught.exception))

    def test_load_error_names_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ci.yaml"
            path.write_text("version: 1\n")
            with self.assertRaises(cfg.ConfigError) as caught:
                cfg.load_config(path)
        self.assertIn(str(path), str(caught.exception))

    def test_main_reports_a_bad_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ci.yaml"
            path.write_text(yaml.safe_dump(config_data(version=2)))
            out = io.StringIO()
            with redirect_stdout(out):
                status = cfg.main(
                    ["--event", "schedule", "--repo", "r", "--therock-dir", tmp,
                     "--config", str(path)]
                )
        self.assertEqual(status, 1)
        self.assertIn("::error::", out.getvalue())


class DecideTest(unittest.TestCase):
    def test_unrelated_pull_request_runs_nothing(self):
        d = decide("pull_request", ["projects/rocminfo/rocminfo.cc"])
        self.assertFalse(d.run_cpu)
        self.assertFalse(d.run_gpu)

    def test_cpu_only_pull_request(self):
        d = decide("pull_request", ["projects/hrr/tests/unit/hrr_format_test.cc"])
        self.assertTrue(d.run_cpu)
        self.assertFalse(d.run_gpu)

    def test_capture_pull_request_runs_enabled_families(self):
        d = decide("pull_request", ["projects/clr/hipamd/src/hrr/hip_capture.cpp"])
        self.assertTrue(d.run_gpu)
        # gfx110x is not triggered by pull requests, gfx125x is disabled and
        # gfx1150 has no runner.
        self.assertEqual(family_keys(d), ["gfx90a", "gfx94x", "gfx950"])
        self.assertEqual(d.families[1]["runner"], "gpu-94x-a")
        self.assertEqual(d.families[1]["gfx_targets"], "gfx942")
        self.assertIn("no runner for gfx1150", d.reason)

    def test_rename_out_of_capture_counts(self):
        # The files API lists previous_filename alongside filename.
        d = decide("pull_request", ["projects/other/x.cpp", "projects/hrr/playback/x.cpp"])
        self.assertTrue(d.run_gpu)

    def test_label_requests_gpu_on_unrelated_pull_request(self):
        d = decide("pull_request", ["projects/rocminfo/x.cc"], labels=["ci:hrr-capture"])
        self.assertTrue(d.run_cpu)
        self.assertTrue(d.run_gpu)

    def test_family_label_adds_family(self):
        d = decide(
            "pull_request",
            ["projects/hrr/playback/x.cpp"],
            labels=["ci:gfx110X-all", "ci:gfx125x"],
        )
        self.assertEqual(family_keys(d), ["gfx90a", "gfx94x", "gfx950", "gfx110x", "gfx125x"])

    def test_label_for_family_not_in_config_is_ignored(self):
        d = decide("pull_request", ["projects/hrr/playback/x.cpp"], labels=["ci:gfx103x"])
        self.assertNotIn("gfx103x", family_keys(d))

    def test_fork_skips_gpu(self):
        d = decide("pull_request", ["projects/hrr/playback/x.cpp"], is_fork=True)
        self.assertTrue(d.run_cpu)
        self.assertFalse(d.run_gpu)
        self.assertIn("fork", d.reason)

    def test_push_without_capture_change_has_no_gpu(self):
        d = decide("push", ["projects/hrr/tests/unit/x.cc"])
        self.assertTrue(d.run_cpu)
        self.assertFalse(d.run_gpu)

    def test_unlisted_change_set_runs_everything(self):
        d = decide("push", None)
        self.assertTrue(d.run_cpu)
        self.assertTrue(d.run_gpu)

    def test_schedule_follows_triggers(self):
        d = decide("schedule", [])
        self.assertEqual(family_keys(d), ["gfx90a", "gfx94x", "gfx950", "gfx110x"])

    def test_multi_gpu_runner(self):
        d = decide("schedule", [])
        runners = {f["family"]: f["runner"] for f in d.families}
        self.assertEqual(runners["gfx950"], "gpu-950-8")

    def test_runner_override(self):
        config = cfg.parse_config(config_data(families={"gfx94x": {"runner": "my-gpu"}}))
        d = decide("schedule", [], config=config)
        self.assertEqual(d.families[0]["runner"], "my-gpu")
        self.assertEqual(d.families[0]["gfx_targets"], "gfx942")

    def test_dispatch_families_replace_the_selection(self):
        d = decide("workflow_dispatch", [], requested=["gfx125x", "gfx1150"])
        self.assertEqual(family_keys(d), ["gfx125x"])
        self.assertIn("no runner for gfx1150", d.reason)

    def test_dispatch_unknown_family(self):
        with self.assertRaises(cfg.ConfigError) as caught:
            decide("workflow_dispatch", [], requested=["gfx103x"])
        self.assertIn("gfx103x", str(caught.exception))

    def test_unknown_event(self):
        d = decide("merge_group", [])
        self.assertFalse(d.run_cpu)
        self.assertFalse(d.run_gpu)


class OutputTest(unittest.TestCase):
    def test_write_outputs(self):
        d = cfg.Decision(True, True, "why", [{"family": "gfx94x", "runner": "r", "gfx_targets": "gfx942"}])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            with redirect_stdout(io.StringIO()):
                cfg.write_outputs(d, "1.2.3", str(out))
            text = out.read_text()
        self.assertIn("run_cpu=true\n", text)
        self.assertIn('families=[{"family":"gfx94x","runner":"r","gfx_targets":"gfx942"}]\n', text)
        self.assertIn("reason=why\n", text)
        self.assertIn("rocm_sdk_version=1.2.3\n", text)


if __name__ == "__main__":
    unittest.main()
