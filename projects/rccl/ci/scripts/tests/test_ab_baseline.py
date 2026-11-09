"""Tests for the A/B baseline: the previous scheduled build and the two-overlay pair.

Run from projects/rccl/ci/scripts:  python3 -m unittest discover -s tests
"""

import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))
import test_madengine as tm

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

ENV = {"GITHUB_RUN_ID": "500", "GITHUB_EVENT_NAME": "schedule", "GITHUB_SHA": "abc",
       "GITHUB_REPOSITORY": "ROCm/rocm-systems", "GITHUB_TOKEN": "t"}


def run(build: int, created_at: str) -> dict:
    return {"id": build, "created_at": created_at}


# Newest first, as the API returns them; 2026-11-07 is a Saturday.
RUNS = [
    run(500, "2026-11-09T06:34:37Z"),  # Mon: the candidate itself
    run(400, "2026-11-08T06:24:24Z"),  # Sun
    run(300, "2026-11-07T06:23:27Z"),  # Sat
    run(200, "2026-11-06T06:27:28Z"),  # Fri
    run(100, "2026-10-31T06:23:34Z"),  # Sat
]


class FakeGitHub:
    """The two Actions API calls previous_scheduled_build() makes."""

    def __init__(self, runs=RUNS, candidate_created="2026-11-09T06:34:37Z"):
        self.runs, self.candidate_created, self.paths = runs, candidate_created, []

    def __call__(self, path: str, env: dict) -> dict:
        self.paths.append(path)
        if path.startswith("/actions/runs/"):
            return {"event": "schedule", "head_sha": "abc",
                    "created_at": self.candidate_created}
        return {"workflow_runs": self.runs}


class PreviousScheduledBuild(unittest.TestCase):
    def previous(self, schedule: str, missing=(), github=None) -> str | None:
        github = github or FakeGitHub()
        with mock.patch.object(tm, "_github_json", github), \
             mock.patch.object(tm, "rccl_artifact_exists",
                               side_effect=lambda build, *_: build not in missing):
            return tm.previous_scheduled_build("500", schedule, "gfx950", env=ENV)

    def test_nightly_is_the_previous_scheduled_build(self):
        self.assertEqual(self.previous("nightly"), "400")

    def test_weekly_is_the_previous_saturday(self):
        self.assertEqual(self.previous("weekly"), "300")

    def test_a_build_without_rccl_in_the_bucket_is_skipped(self):
        self.assertEqual(self.previous("nightly", missing={"400"}), "300")
        self.assertEqual(self.previous("weekly", missing={"300"}), "100")

    def test_nothing_older(self):
        self.assertIsNone(self.previous("weekly", missing={"300", "100"}))

    def test_only_older_scheduled_develop_runs_are_listed(self):
        github = FakeGitHub()
        self.previous("nightly", github=github)
        listing = github.paths[-1]
        self.assertTrue(listing.startswith(f"/actions/workflows/{tm.CI_WORKFLOW}/runs?"))
        for part in ("event=schedule", "branch=develop", "created=%3C2026-11-09T06%3A34%3A37Z"):
            self.assertIn(part, listing)

    def test_the_bucket_key_is_the_gpu_targets_runtime(self):
        self.assertEqual(
            tm.rccl_artifact_url("300", "gfx950", "ROCm/rocm-systems"),
            "https://therock-ci-artifacts-external.s3.amazonaws.com/"
            "ROCm-rocm-systems/300-linux/rccl_lib_gfx950.tar.zst")


class ResolveBaseline(unittest.TestCase):
    def resolve(self, workload="llama-3.1-70b-training", **env) -> str:
        return tm.resolve_baseline(workload, "gfx950", env={**ENV, **env})

    def test_the_dispatchers_build_wins(self):
        with mock.patch.object(tm, "previous_scheduled_build") as previous:
            self.assertEqual(self.resolve(BASELINE_ARTIFACT_RUN_ID=" 7 "), "7")
        previous.assert_not_called()

    def test_a_dispatched_build_must_be_a_run_id(self):
        with self.assertRaisesRegex(RuntimeError, "must be a run ID"):
            self.resolve(BASELINE_ARTIFACT_RUN_ID="7; true")

    def test_the_workloads_schedule_picks_the_build(self):
        with mock.patch.object(tm, "previous_scheduled_build", return_value="300") as previous:
            self.assertEqual(self.resolve("deepseek-r1-sglang-disagg", ARTIFACT_RUN_ID="500"),
                             "300")
        self.assertEqual(previous.call_args[0][:3], ("500", "weekly", "gfx950"))

    def test_no_build_fails(self):
        with mock.patch.object(tm, "previous_scheduled_build", return_value=None), \
             self.assertRaisesRegex(RuntimeError, "baseline_artifact_run_id"):
            self.resolve()

    def test_an_unreachable_api_fails(self):
        error = urllib.error.URLError("down")
        with mock.patch.object(tm, "_github_json", side_effect=error), \
             self.assertRaisesRegex(RuntimeError, "could not list"):
            self.resolve(ARTIFACT_RUN_ID="500")


class ArtifactBuild(unittest.TestCase):
    def test_a_run_knows_its_own_build_without_the_api(self):
        error = urllib.error.HTTPError("u", 403, "Forbidden", {}, None)
        with mock.patch.object(tm.urllib.request, "urlopen", side_effect=error):
            self.assertEqual(tm.artifact_build("500", env=ENV),
                             {"event": "schedule", "sha": "abc", "created_at": ""})

    def test_another_build_is_looked_up(self):
        body = io.BytesIO(json.dumps({"event": "schedule", "head_sha": "def",
                                      "created_at": "2026-11-07T06:23:27Z"}).encode())
        with mock.patch.object(tm.urllib.request, "urlopen", return_value=body) as urlopen:
            self.assertEqual(tm.artifact_build("9", env=ENV),
                             {"event": "schedule", "sha": "def",
                              "created_at": "2026-11-07T06:23:27Z"})
        request = urlopen.call_args[0][0]
        self.assertEqual(request.full_url,
                         "https://api.github.com/repos/ROCm/rocm-systems/actions/runs/9")
        self.assertEqual(request.get_header("Authorization"), "Bearer t")

    def test_a_failed_lookup_leaves_it_unknown(self):
        error = urllib.error.HTTPError("u", 404, "Not Found", {}, None)
        with mock.patch.object(tm.urllib.request, "urlopen", side_effect=error):
            self.assertEqual(tm.artifact_build("9", env=ENV),
                             {"event": "", "sha": "", "created_at": ""})


class StageBothPhases(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.work = self.root / "work"
        self.work.mkdir()
        self.libs = {}
        for phase in (tm.BASELINE, tm.CANDIDATE):
            lib = self.root / phase / "lib" / "librccl.so"
            lib.parent.mkdir(parents=True)
            lib.write_bytes(f"librccl {phase}".encode())
            self.libs[phase] = lib
        patcher = mock.patch.dict(os.environ, {"GITHUB_RUN_ID": "42"})
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("RCCL_COMMIT_HASH", None)

    def stage(self, phase: str) -> str:
        return tm.stage_rccl_overlay(
            self.libs[phase], "rocm/primus:v26.4", "gfx950", self.work,
            from_ref="rocm/primus:v26.4@sha256:abc", phase=phase,
        )

    def test_each_phase_lays_its_own_build_over_the_same_base(self):
        tags = {phase: self.stage(phase) for phase in (tm.BASELINE, tm.CANDIDATE)}
        self.assertNotEqual(tags[tm.BASELINE], tags[tm.CANDIDATE])
        ctx = self.work / tm.OVERLAY_CTX
        for phase, tag in tags.items():
            self.assertIn(f"-run42-{phase}-", tag)
            dockerfile = (ctx / tm.OVERLAY_DOCKERFILES[phase]).read_text()
            self.assertTrue(dockerfile.startswith("FROM rocm/primus:v26.4@sha256:abc\n"))
            self.assertIn(f"COPY {tm.OVERLAY_LIBS[phase]}/ /tmp/rccl_ci/", dockerfile)
            staged = ctx / tm.OVERLAY_LIBS[phase] / "librccl.so"
            self.assertEqual(staged.read_bytes(), self.libs[phase].read_bytes())


@unittest.skipIf(yaml is None, "PyYAML not installed")
class Workflow(unittest.TestCase):
    def setUp(self):
        path = (Path(__file__).resolve().parents[5] / ".github" / "workflows"
                / "therock-rccl-test-madengine.yml")
        self.workflow = yaml.safe_load(path.read_text())

    def test_the_baseline_build_is_optional_on_both_triggers(self):
        for trigger in ("workflow_call", "workflow_dispatch"):
            inputs = self.workflow[True][trigger]["inputs"]
            self.assertEqual(inputs["baseline_artifact_run_id"].get("default"), "", trigger)
            self.assertFalse(inputs["baseline_artifact_run_id"].get("required", False))

    def test_the_resolved_build_is_fetched_and_tested(self):
        steps = {s["name"]: s for s in self.workflow["jobs"]["test_madengine"]["steps"]}
        names = list(steps)
        resolve = steps["Resolve the A/B baseline build"]
        self.assertIn("--resolve-baseline", resolve["run"])
        self.assertLess(names.index("Resolve the A/B baseline build"),
                        names.index("Run MADEngine workload"))
        resolved = "${{ steps.baseline.outputs.artifact_run_id }}"
        fetch = steps["Fetch baseline RCCL artifacts"]
        self.assertEqual(fetch["env"]["BASELINE_ARTIFACT_RUN_ID"], resolved)
        self.assertIn('--output-dir="${BASELINE_ARTIFACTS_DIR}"', fetch["run"])
        run = steps["Run MADEngine workload"]
        self.assertEqual(run["env"]["BASELINE_ARTIFACT_RUN_ID"], resolved)
        self.assertIn('--baseline-artifact-dir "${BASELINE_ARTIFACTS_DIR}"', run["run"])


if __name__ == "__main__":
    unittest.main()
