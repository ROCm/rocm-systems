#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for hrr_ci_configure.py."""

import copy
import io
import os
import sys
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

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
    "gfx950": {"multi_gpu_triggers": ["schedule"]},
    "gfx110x": {"triggers": ["schedule", "workflow_dispatch"]},
    "gfx1150": {},
    "gfx125x": {"enabled": False},
}


def config_data(**changes):
    data = copy.deepcopy(REPO_CONFIG)
    data["families"] = copy.deepcopy(FAMILIES)
    data.update(changes)
    families = data.get("families")
    simulator = data.get("simulator")
    if isinstance(families, dict) and isinstance(simulator, dict):
        present = [
            key for key in simulator.get("default_families", []) if key in families
        ]
        if not present and families:
            present = [next(iter(families))]
        simulator["default_families"] = present
        configs = simulator.setdefault("configs", {})
        if isinstance(configs, dict):
            for key in [key for key in configs if key not in families]:
                del configs[key]
            for key in present:
                configs.setdefault(key, f"emulation/rocjitsu/configs/{key}.json")
    return data


def gpu_data(**changes):
    """The family-selection tests predate the simulator default."""
    data = config_data(**changes)
    data["backend"] = {"default": "gpu", "gpu": {"triggers": []}}
    return data


CONFIG = cfg.parse_config(gpu_data())
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
    def test_only_gfx94x_runs_by_default(self):
        """Every family stays listed, but only gfx94x runs unless asked for.

        The others stay in ci.yaml so the workflow_dispatch families input can
        still request them; disabling is what keeps them off the default run.
        ci:gfx labels are TheRock's and do not select an HRR family.
        """
        self.assertEqual(
            sorted(key for key, family in REPO.families.items() if family.enabled),
            ["gfx94x"],
        )
        for key, family in REPO.families.items():
            self.assertEqual(family.triggers, cfg.EVENTS, key)

    def test_a_gfx_label_does_not_add_a_simulator_family(self):
        """ci:gfx is a TheRock label. It must not widen the rocJITsu matrix."""
        data = copy.deepcopy(REPO_CONFIG)
        data["simulator"]["default_families"] = ["gfx950", "gfx125x"]
        d = decide(
            "pull_request",
            ["projects/hrr/playback/x.cpp"],
            ["ci:gfx110x"],
            config=cfg.parse_config(data),
        )
        self.assertEqual(family_keys(d), ["gfx950", "gfx125x"])
        self.assertTrue(all(row["backend"] == "simulator" for row in d.families))

    def test_multi_gpu_runner_only_nightly(self):
        for key, family in REPO.families.items():
            self.assertIn(family.multi_gpu_triggers, ((), ("schedule",)), key)
        self.assertEqual(REPO.families["gfx950"].multi_gpu_triggers, ("schedule",))


class PathClassificationTest(unittest.TestCase):
    def test_playback_change_is_capture(self):
        self.assertTrue(
            cfg.touches_capture(["projects/hrr/playback/hip_playback.cpp"], REPO)
        )

    def test_capture_source_is_capture(self):
        self.assertTrue(
            cfg.touches_capture(["projects/clr/hipamd/src/hrr/hip_capture.cpp"], REPO)
        )

    def test_ci_config_runs_the_suite_and_is_not_capture(self):
        paths = ["projects/hrr/tests/config/ci.yaml"]
        self.assertFalse(cfg.touches_capture(paths, REPO))
        self.assertTrue(cfg.touches_suite(paths, REPO))
        decision = decide("pull_request", paths, config=REPO)
        self.assertTrue(decision.run_gpu)
        self.assertEqual(family_keys(decision), list(REPO.simulator.default_families))

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
        self.assertRejected(
            config_data(families=families), "families.gfx94x: unknown key runs_on"
        )

    def test_unknown_trigger(self):
        families = {"gfx94x": {"triggers": ["nightly"]}}
        self.assertRejected(config_data(families=families), "unknown trigger 'nightly'")

    def test_unknown_multi_gpu_trigger(self):
        families = {"gfx950": {"multi_gpu_triggers": ["nightly"]}}
        self.assertRejected(
            config_data(families=families),
            "families.gfx950.multi_gpu_triggers: unknown trigger",
        )

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

    def test_family_targets(self):
        self.assertEqual(
            cfg.family_targets(MATRIX["gfx110x"]["linux"]), ["gfx1100", "gfx1101"]
        )
        # gfx1150 lists no fetch targets; its family name is the target.
        self.assertEqual(cfg.family_targets(MATRIX["gfx1150"]["linux"]), ["gfx1150"])
        self.assertEqual(
            cfg.family_targets({"fetch-gfx-targets": ["gfx942", "gfx942:xnack+"]}),
            ["gfx942"],
        )
        self.assertEqual(cfg.family_targets({"family": "gfx101X-dgpu"}), [])

    def test_targets_cover_every_family(self):
        archs = [
            "gfx90a",
            "gfx942",
            "gfx950",
            "gfx1100",
            "gfx1101",
            "gfx1150",
            "gfx1250",
        ]
        cfg.check_targets(CONFIG, MATRIX, archs)
        with self.assertRaises(cfg.ConfigError) as caught:
            cfg.check_targets(
                CONFIG, MATRIX, [a for a in archs if a not in ("gfx1101", "gfx1250")]
            )
        # gfx125x is disabled, but a label can still request it.
        self.assertIn(
            "gfx110x needs gfx1101; gfx125x needs gfx1250", str(caught.exception)
        )

    def test_family_without_targets_is_an_error(self):
        config = cfg.parse_config(config_data(families={"gfx94x": {}}))
        matrix = {"gfx94x": {"linux": {"family": "gfx94X-dcgpu"}}}
        with self.assertRaises(cfg.ConfigError) as caught:
            cfg.check_targets(config, matrix, ["gfx942"])
        self.assertIn("no GPU targets for gfx94x", str(caught.exception))

    def test_repo_integration_config_lists_archs(self):
        self.assertIn("gfx1103", cfg.load_archs(cfg.INTEGRATION_CONFIG))

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
                    [
                        "--event",
                        "schedule",
                        "--repo",
                        "r",
                        "--therock-dir",
                        tmp,
                        "--config",
                        str(path),
                    ]
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
        self.assertEqual(
            d.families[1],
            {
                "family": "gfx94x",
                "runner": "gpu-94x-a",
                "backend": "gpu",
                "arch": "gfx942",
                "config": "",
            },
        )
        self.assertIn("no runner for gfx1150", d.reason)
        self.assertEqual(
            d.skipped,
            [
                {
                    "family": "gfx110x",
                    "reason": "not run on pull_request (ci.yaml triggers)",
                },
                {"family": "gfx1150", "reason": cfg.NO_RUNNER},
                {"family": "gfx125x", "reason": "disabled in ci.yaml"},
            ],
        )

    def test_rename_out_of_capture_counts(self):
        # The files API lists previous_filename alongside filename.
        d = decide(
            "pull_request", ["projects/other/x.cpp", "projects/hrr/playback/x.cpp"]
        )
        self.assertTrue(d.run_gpu)

    def test_label_requests_gpu_on_unrelated_pull_request(self):
        d = decide(
            "pull_request", ["projects/rocminfo/x.cc"], labels=["ci:hrr-capture"]
        )
        self.assertTrue(d.run_cpu)
        self.assertTrue(d.run_gpu)

    def test_label_match_ignores_case(self):
        d = decide(
            "pull_request", ["projects/rocminfo/x.cc"], labels=["CI:HRR-Capture"]
        )
        self.assertTrue(d.run_gpu)

    def test_capture_only_path_also_runs_cpu(self):
        d = decide("pull_request", ["projects/clr/hipamd/src/hip_context.cpp"])
        self.assertTrue(d.run_cpu)
        self.assertTrue(d.run_gpu)

    def test_gfx_label_does_not_add_a_family(self):
        d = decide(
            "pull_request",
            ["projects/hrr/playback/x.cpp"],
            labels=["ci:gfx110X-all", "ci:gfx125x"],
        )
        self.assertEqual(family_keys(d), ["gfx90a", "gfx94x", "gfx950"])

    def test_label_for_family_not_in_config_is_ignored(self):
        d = decide(
            "pull_request", ["projects/hrr/playback/x.cpp"], labels=["ci:gfx103x"]
        )
        self.assertNotIn("gfx103x", family_keys(d))

    def test_fork_still_runs_the_simulator(self):
        d = decide(
            "pull_request",
            ["projects/hrr/playback/x.cpp"],
            is_fork=True,
            config=cfg.parse_config(config_data()),
        )
        self.assertTrue(d.run_cpu)
        self.assertTrue(d.run_gpu)
        self.assertTrue(d.families)
        self.assertTrue(all(row["backend"] == "simulator" for row in d.families))

    def test_fork_skips_the_real_gpu_backend(self):
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

    def test_multi_gpu_runner_on_its_triggers_only(self):
        nightly = {f["family"]: f["runner"] for f in decide("schedule", []).families}
        self.assertEqual(nightly["gfx950"], "gpu-950-8")
        pr = decide("pull_request", ["projects/hrr/playback/x.cpp"])
        self.assertEqual(
            {f["family"]: f["runner"] for f in pr.families}["gfx950"], "gpu-950-1"
        )

    def test_runner_override(self):
        config = cfg.parse_config(gpu_data(families={"gfx94x": {"runner": "my-gpu"}}))
        d = decide("schedule", [], config=config)
        self.assertEqual(d.families[0]["runner"], "my-gpu")

    def test_dispatch_families_replace_the_selection(self):
        d = decide("workflow_dispatch", [], requested=["gfx125x", "gfx1150"])
        self.assertEqual(family_keys(d), ["gfx125x"])
        self.assertIn("no runner for gfx1150", d.reason)
        self.assertIn({"family": "gfx94x", "reason": "not requested"}, d.skipped)

    def test_dispatch_unknown_family(self):
        with self.assertRaises(cfg.ConfigError) as caught:
            decide("workflow_dispatch", [], requested=["gfx103x"])
        self.assertIn("gfx103x", str(caught.exception))

    def test_unknown_event(self):
        d = decide("merge_group", [])
        self.assertFalse(d.run_cpu)
        self.assertFalse(d.run_gpu)


class SimulatorBackendTest(unittest.TestCase):
    def test_default_runs_no_real_gpu_job_on_any_event(self):
        """Pull requests, pushes and the nightly run only rocJitsu jobs, on every
        family that has a rocJitsu config; none queues on a GPU runner."""
        for event in ("pull_request", "push", "schedule", "workflow_dispatch"):
            decision = decide(event, ["projects/hrr/playback/x.cpp"], config=REPO)
            self.assertTrue(decision.run_gpu, event)
            self.assertEqual(
                family_keys(decision),
                [
                    "gfx90a",
                    "gfx94x",
                    "gfx950",
                    "gfx110x",
                    "gfx1151",
                    "gfx120x",
                    "gfx125x",
                ],
                event,
            )
            self.assertEqual(
                {row["backend"] for row in decision.families}, {"simulator"}, event
            )
            self.assertEqual(
                {row["runner"] for row in decision.families}, {"ubuntu-24.04"}, event
            )

    def test_every_default_simulator_family_has_a_config_file(self):
        root = Path(__file__).resolve().parents[4]
        for key in REPO.simulator.default_families:
            self.assertTrue((root / REPO.simulator.configs[key]).is_file(), key)

    def test_the_label_does_not_move_a_run_to_a_real_gpu(self):
        for label, path in (
            ("ci:hrr-capture", "projects/hrr/README.md"),
            ("ci:gfx94x", "projects/hrr/playback/x.cpp"),
        ):
            decision = decide("pull_request", [path], [label], config=REPO)
            self.assertTrue(decision.run_gpu, label)
            self.assertEqual(
                {row["backend"] for row in decision.families}, {"simulator"}, label
            )

    def test_default_event_is_the_rocjitsu_jobs(self):
        decision = decide("pull_request", ["projects/hrr/playback/x.cpp"], config=REPO)
        self.assertIn("gfx950", family_keys(decision))
        self.assertIn("gfx125x", family_keys(decision))
        by_family = {row["family"]: row for row in decision.families}
        self.assertEqual(by_family["gfx950"]["runner"], "ubuntu-24.04")
        self.assertEqual(by_family["gfx950"]["backend"], "simulator")
        self.assertEqual(by_family["gfx950"]["arch"], "gfx950")
        self.assertTrue(
            by_family["gfx950"]["config"].endswith("gfx950_mi355x_kmd_2gpu.json")
        )
        self.assertEqual(by_family["gfx125x"]["runner"], "ubuntu-24.04")
        self.assertEqual(by_family["gfx125x"]["arch"], "gfx1250")
        self.assertTrue(
            by_family["gfx125x"]["config"].endswith("gfx1250_mi455x_kmd_4gpu.json")
        )
        self.assertNotEqual(
            by_family["gfx125x"]["runner"], MATRIX["gfx125x"]["linux"]["test-runs-on"]
        )

    def test_gpu_dispatch_selects_the_therock_runner(self):
        decision = cfg.decide(
            "workflow_dispatch",
            [],
            [],
            False,
            ["gfx94x"],
            REPO,
            MATRIX,
            pick_first,
            backend="gpu",
        )
        self.assertEqual(family_keys(decision), ["gfx94x"])
        self.assertEqual(decision.families[0]["backend"], "gpu")
        self.assertEqual(decision.families[0]["runner"], "gpu-94x-a")
        self.assertNotIn("gfx125x", family_keys(decision))

    def test_pull_request_gpu_triggers_select_real_runners(self):
        data = config_data()
        data["backend"]["gpu"]["triggers"] = ["pull_request"]
        config = cfg.parse_config(data)
        decision = decide(
            "pull_request", ["projects/hrr/playback/x.cpp"], config=config
        )
        self.assertEqual(family_keys(decision), ["gfx90a", "gfx94x", "gfx950"])
        self.assertTrue(all(row["backend"] == "gpu" for row in decision.families))
        self.assertNotIn("gfx125x", family_keys(decision))

    def test_simulator_rejects_a_requested_family_without_a_config(self):
        with self.assertRaises(cfg.ConfigError) as caught:
            decide(
                "workflow_dispatch", [], requested=["gfx103x"], config=REPO
            )
        self.assertIn("no config for gfx103x", str(caught.exception))

    def test_unknown_backend_value(self):
        data = config_data()
        data["backend"]["default"] = "metal"
        with self.assertRaises(cfg.ConfigError) as caught:
            cfg.parse_config(data)
        self.assertIn("backend.default", str(caught.exception))


class LabelEventTest(unittest.TestCase):
    CAPTURE = ["projects/hrr/playback/x.cpp"]
    WORKFLOW = (
        Path(__file__).resolve().parents[4] / ".github" / "workflows" / "hrr-ci.yml"
    )

    def test_which_labels_matter(self):
        for label in ("ci:hrr-capture", "CI:HRR-Capture"):
            self.assertTrue(cfg.label_matters(label, REPO), label)
        for label in ("documentation", "ci:skip", "hrr", "ci:gfx110x", "ci:gfx110X-all"):
            self.assertFalse(cfg.label_matters(label, REPO), label)

    def test_other_label_runs_nothing(self):
        d = cfg.decide(
            "pull_request",
            self.CAPTURE,
            ["ci:hrr-capture", "documentation"],
            False,
            [],
            CONFIG,
            MATRIX,
            pick_first,
            added_label="documentation",
        )
        self.assertFalse(d.run_cpu)
        self.assertFalse(d.run_gpu)
        self.assertIn("documentation", d.reason)

    def test_capture_label_reruns_and_a_gfx_label_does_not(self):
        d = cfg.decide(
            "pull_request",
            self.CAPTURE,
            ["ci:hrr-capture"],
            False,
            [],
            CONFIG,
            MATRIX,
            pick_first,
            added_label="ci:hrr-capture",
        )
        self.assertTrue(d.run_gpu)
        d = cfg.decide(
            "pull_request",
            self.CAPTURE,
            ["ci:gfx110x"],
            False,
            [],
            CONFIG,
            MATRIX,
            pick_first,
            added_label="ci:gfx110x",
        )
        self.assertFalse(d.run_cpu)
        self.assertFalse(d.run_gpu)

    def test_labeled_trigger_and_summary_report_a_verdict(self):
        """Regression for the label that can turn a failing HRR CI green.

        github keeps the most recent check run of a name on a commit, and a
        skipped check satisfies a required-status rule, so a labeled run
        that configures nothing must not let its summary skip on top of the
        run that failed: the summary must run for every labeled event and
        answer with the verdict of the run that tested the commit (the
        label run knows nothing itself), never a blanket pass.
        """
        text = self.WORKFLOW.read_text(encoding="utf-8")
        types = re.search(r"types:\s*\[([^\]]*)\]", text).group(1)
        self.assertIn("labeled", types)

        summary = text[text.index("  summary:") :]
        # The summary also runs when the label run configured nothing.
        self.assertIn("github.event.action == 'labeled'", summary)
        # And it answers with an actual verdict, never a skip of its own.
        self.assertNotIn(
            "needs.configure.result != 'success' || needs.configure.outputs.run_cpu == 'true' ||",
            summary,
        )
        # Only a label run that configured nothing passes the verdict through;
        # every other run reports its own jobs.
        step = summary[summary.index("PASS_THROUGH:") : summary.index("run: >-")]
        self.assertIn("github.event.action == 'labeled'", step)
        self.assertIn("needs.configure.result == 'success'", step)
        self.assertIn("run_cpu == 'false'", step)
        self.assertIn("run_gpu == 'false'", step)
        self.assertIn('--pass-through "${PASS_THROUGH}"', summary)
        # The runs API keys a run by the pull request head, not the merge commit.
        self.assertIn("github.event.pull_request.head.sha", summary)

    def test_concurrency_group_makes_the_same_test(self):
        text = self.WORKFLOW.read_text(encoding="utf-8")
        group = text[text.index("concurrency:") : text.index("cancel-in-progress:")]
        self.assertIn(f"github.event.label.name != '{REPO.capture_label}'", group)
        self.assertNotIn("ci:gfx", group)


class ForkTest(unittest.TestCase):
    REPO = "ROCm/rocm-systems"

    def test_same_repository_is_not_a_fork(self):
        self.assertFalse(cfg.is_fork("pull_request", self.REPO, self.REPO))

    def test_other_repository_is_a_fork(self):
        self.assertTrue(cfg.is_fork("pull_request", "someone/rocm-systems", self.REPO))

    def test_deleted_head_repository_is_a_fork(self):
        self.assertTrue(cfg.is_fork("pull_request", "", self.REPO))

    def test_only_pull_requests_are_forks(self):
        self.assertFalse(cfg.is_fork("push", "", self.REPO))

    def test_main_skips_real_gpu_for_a_deleted_head_repository(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            cfg, "list_pr_files", return_value=["projects/hrr/playback/x.cpp"]
        ), mock.patch.object(
            cfg, "load_matrix", return_value=(MATRIX, pick_first)
        ), mock.patch.dict(
            os.environ, {"GITHUB_OUTPUT": ""}
        ):
            path = Path(tmp) / "ci.yaml"
            path.write_text(yaml.safe_dump(gpu_data()))
            out = io.StringIO()
            with redirect_stdout(out):
                status = cfg.main(
                    [
                        "--event",
                        "pull_request",
                        "--repo",
                        self.REPO,
                        "--pr-number",
                        "1",
                        "--head-repo",
                        "",
                        "--therock-dir",
                        tmp,
                        "--config",
                        str(path),
                    ]
                )
        self.assertEqual(status, 0)
        self.assertIn("run_cpu=true\n", out.getvalue())
        self.assertIn("run_gpu=false\n", out.getvalue())
        self.assertIn("fork", out.getvalue())


class OutputTest(unittest.TestCase):
    def test_write_outputs(self):
        d = cfg.Decision(
            True,
            True,
            "why",
            [{"family": "gfx94x", "runner": "r"}],
            [{"family": "gfx125x", "reason": "disabled in ci.yaml"}],
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            with redirect_stdout(io.StringIO()):
                cfg.write_outputs(d, "1.2.3", str(out))
            text = out.read_text()
        self.assertIn("run_cpu=true\n", text)
        self.assertIn('families=[{"family":"gfx94x","runner":"r"}]\n', text)
        self.assertIn(
            'skipped_families=[{"family":"gfx125x","reason":"disabled in ci.yaml"}]\n',
            text,
        )
        self.assertIn("reason=why\n", text)
        self.assertIn("rocm_sdk_version=1.2.3\n", text)


class WorkflowPinTest(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[4]
    SHA = re.compile(r"(?:@|ref:\s*)([0-9a-f]{40})")

    def therock_shas(self, path: Path) -> list[str]:
        lines = path.read_text(encoding="utf-8").splitlines()
        found = []
        for index, line in enumerate(lines):
            if "ROCm/TheRock" not in line:
                continue
            window = "\n".join(lines[index:index + 6])
            found.extend(self.SHA.findall(window))
        return found

    def test_hrr_workflows_use_the_multi_arch_therock_pin(self):
        pinned = self.therock_shas(
            self.ROOT / ".github/workflows/therock-multi-arch-ci.yml"
        )
        self.assertTrue(pinned)
        self.assertEqual(set(pinned), {pinned[0]})
        for name in ("hrr-ci.yml", "hrr-gpu.yml"):
            path = self.ROOT / ".github/workflows" / name
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("THEROCK_PIN", text, name)
            self.assertNotIn("BUILD_IMAGE", text, name)
            found = self.therock_shas(path)
            self.assertTrue(found, name)
            self.assertEqual(set(found), {pinned[0]}, name)


if __name__ == "__main__":
    unittest.main()
