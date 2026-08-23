#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for rccl_perf_gate.py.

Runs standalone (no pip install needed) and under pytest via conftest.py.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rccl_perf_gate as gate  # noqa: E402

COMMAND = "/perf-regression"
REPOSITORY = "ROCm/rocm-systems"


class FakeApi:
    """GitHubApi stand-in serving canned responses and recording writes."""

    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses
        self.posts: list[tuple[str, dict]] = []

    def get(self, path: str) -> object:
        value = self.responses[path]
        if isinstance(value, Exception):
            raise value
        return value

    def create(self, path: str, payload: dict) -> None:
        self.posts.append((path, payload))


def comment_event(body: str, login: str = "octocat", number: int = 9950) -> dict:
    return {
        "issue": {"number": number, "pull_request": {"url": "https://example/1"}},
        "comment": {"body": body, "user": {"login": login}},
    }


def permission_path(login: str = "octocat") -> str:
    return f"/repos/{REPOSITORY}/collaborators/{login}/permission"


class GateCommandTest(unittest.TestCase):
    def test_exact_command_is_accepted_with_surrounding_whitespace(self) -> None:
        for body in (COMMAND, f"  {COMMAND}  ", f"{COMMAND}\r\n", f"\n{COMMAND}\n"):
            with self.subTest(body=body):
                self.assertTrue(gate.is_gate_command(body, COMMAND))

    def test_substrings_and_quotations_are_rejected(self) -> None:
        near_misses = [
            f"{COMMAND} please",
            f"`{COMMAND}`",
            f"> {COMMAND}",
            f"see {COMMAND} docs",
            f"{COMMAND}\nrm -rf /",
            "/perf-regressions",
            "",
        ]
        for body in near_misses:
            with self.subTest(body=body):
                self.assertFalse(gate.is_gate_command(body, COMMAND))


@unittest.skipUnless(shutil.which("git"), "git is required for check-ref-format")
class ValidateRefTest(unittest.TestCase):
    def test_accepts_ordinary_branch_names(self) -> None:
        for ref in ("develop", "users/pvallem/some-branch", "release/rocm-7.0"):
            with self.subTest(ref=ref):
                self.assertEqual(gate.validate_ref(ref), ref)

    def test_rejects_names_git_would_reject(self) -> None:
        for ref in ("", "..", "bad branch", "a:b", "a^b", "tail.lock", "back\\slash"):
            with self.subTest(ref=ref):
                with self.assertRaises(ValueError):
                    gate.validate_ref(ref)


@unittest.skipUnless(shutil.which("git"), "git is required for check-ref-format")
class ResolveRequestTest(unittest.TestCase):
    def _resolve(self, api: FakeApi, event: dict) -> gate.GateRequest:
        return gate.resolve_request(
            "issue_comment", event, api, REPOSITORY, COMMAND, {}
        )

    def _api(self, permission: object, base_ref: str = "develop") -> FakeApi:
        return FakeApi(
            {
                permission_path(): permission,
                f"/repos/{REPOSITORY}/pulls/9950": {
                    "head": {"sha": "a" * 40, "repo": {"full_name": REPOSITORY}},
                    "base": {"ref": base_ref},
                },
            }
        )

    def test_write_permission_authorizes_and_resolves_context(self) -> None:
        request = self._resolve(
            self._api({"permission": "write"}), comment_event(COMMAND)
        )
        self.assertTrue(request.authorized)
        self.assertEqual(request.head_sha, "a" * 40)
        self.assertEqual(request.base_ref, "develop")
        self.assertEqual(request.pr_number, "9950")
        self.assertEqual(request.requester, "octocat")
        self.assertEqual(request.head_repo, REPOSITORY)

    def test_admin_and_maintain_are_authorized(self) -> None:
        for permission in ("admin", "maintain"):
            with self.subTest(permission=permission):
                request = self._resolve(
                    self._api({"permission": permission}), comment_event(COMMAND)
                )
                self.assertTrue(request.authorized)

    def test_read_and_triage_are_denied_with_a_warning(self) -> None:
        for permission in ("read", "triage", "none"):
            with self.subTest(permission=permission):
                request = self._resolve(
                    self._api({"permission": permission}), comment_event(COMMAND)
                )
                self.assertFalse(request.authorized)
                self.assertEqual(request.level, "warning")

    def test_permission_read_failure_fails_closed_and_is_loud(self) -> None:
        error = urllib.error.HTTPError(
            "https://api.github.com", 403, "Forbidden", {}, None
        )
        request = self._resolve(self._api(error), comment_event(COMMAND))
        self.assertFalse(request.authorized)
        self.assertEqual(request.level, "error")

    def test_wrong_command_never_reaches_the_permission_api(self) -> None:
        api = self._api({"permission": "admin"})
        api.responses[permission_path()] = AssertionError("must not be called")
        request = self._resolve(api, comment_event(f"{COMMAND} now"))
        self.assertFalse(request.authorized)

    def test_comment_on_a_plain_issue_is_ignored(self) -> None:
        event = comment_event(COMMAND)
        del event["issue"]["pull_request"]
        request = self._resolve(self._api({"permission": "admin"}), event)
        self.assertFalse(request.authorized)

    def test_fork_head_repo_is_reported_verbatim(self) -> None:
        api = self._api({"permission": "admin"})
        api.responses[f"/repos/{REPOSITORY}/pulls/9950"]["head"]["repo"] = {
            "full_name": "someone/rocm-systems"
        }
        request = self._resolve(api, comment_event(COMMAND))
        self.assertTrue(request.authorized)
        self.assertNotEqual(request.head_repo, REPOSITORY)

    def test_deleted_fork_yields_an_empty_head_repo(self) -> None:
        api = self._api({"permission": "admin"})
        api.responses[f"/repos/{REPOSITORY}/pulls/9950"]["head"]["repo"] = None
        request = self._resolve(api, comment_event(COMMAND))
        self.assertEqual(request.head_repo, "")

    def test_malicious_base_ref_is_rejected(self) -> None:
        api = self._api({"permission": "admin"}, base_ref="develop --upload-pack=evil")
        with self.assertRaises(ValueError):
            self._resolve(api, comment_event(COMMAND))

    def test_workflow_dispatch_is_authorized_without_the_api(self) -> None:
        request = gate.resolve_request(
            "workflow_dispatch",
            {},
            FakeApi({}),
            REPOSITORY,
            COMMAND,
            {
                "GITHUB_SHA": "b" * 40,
                "GITHUB_REF_NAME": "users/pvallem/demo",
                "GITHUB_ACTOR": "pvallem",
            },
        )
        self.assertTrue(request.authorized)
        self.assertEqual(request.head_sha, "b" * 40)
        self.assertEqual(request.base_ref, "users/pvallem/demo")
        self.assertEqual(request.pr_number, "")
        self.assertEqual(request.head_repo, REPOSITORY)


class VerdictTest(unittest.TestCase):
    def test_clean_detector_run_passes(self) -> None:
        verdict = gate.compute_verdict("success", "success", "0")
        self.assertEqual((verdict.publish, verdict.state), (True, "success"))

    def test_detected_regression_stays_advisory(self) -> None:
        verdict = gate.compute_verdict("success", "failure", "1")
        self.assertEqual(verdict.state, "success")

    def test_detector_infra_error_fails(self) -> None:
        verdict = gate.compute_verdict("success", "failure", "2")
        self.assertEqual(verdict.state, "failure")
        self.assertIn("exit 2", verdict.description)

    def test_missing_detector_exit_code_fails(self) -> None:
        verdict = gate.compute_verdict("success", "failure", "")
        self.assertEqual(verdict.state, "failure")
        self.assertIn("unknown", verdict.description)

    def test_build_failure_fails(self) -> None:
        verdict = gate.compute_verdict("failure", "skipped", "")
        self.assertEqual(verdict.state, "failure")

    def test_skipped_or_cancelled_detector_is_an_error(self) -> None:
        for result in ("cancelled", "skipped"):
            with self.subTest(result=result):
                verdict = gate.compute_verdict("success", result, "")
                self.assertEqual(verdict.state, "error")

    def test_superseded_run_does_not_publish(self) -> None:
        verdict = gate.compute_verdict("cancelled", "skipped", "")
        self.assertFalse(verdict.publish)
        self.assertEqual(verdict.outputs()["publish"], "false")


class ChangedPathsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original = gate.git_output
        self.addCleanup(setattr, gate, "git_output", self.original)

    def _stub_diff(self, names: str) -> None:
        def fake(*arguments: str) -> str:
            return "merge-base-sha" if arguments[0] == "merge-base" else names

        gate.git_output = fake

    def test_detects_an_rccl_source_change(self) -> None:
        self._stub_diff("docs/readme.md\nprojects/rccl/src/init.cc\n")
        self.assertTrue(gate.rccl_paths_changed("develop"))

    def test_ignores_changes_outside_rccl(self) -> None:
        self._stub_diff(".github/workflows/rccl_perf_regression.yml\ndocs/x.md\n")
        self.assertFalse(gate.rccl_paths_changed("develop"))

    def test_similarly_named_directories_do_not_match(self) -> None:
        self._stub_diff("projects/rccl-tests/foo.cc\nprojects/rcclx/bar.cc\n")
        self.assertFalse(gate.rccl_paths_changed("develop"))

    def test_empty_diff_is_not_a_change(self) -> None:
        self._stub_diff("")
        self.assertFalse(gate.rccl_paths_changed("develop"))


class BuildDecisionTest(unittest.TestCase):
    def test_comment_run_builds_only_when_rccl_changed(self) -> None:
        self.assertTrue(gate.should_build_rccl("issue_comment", "1", "0"))
        self.assertFalse(gate.should_build_rccl("issue_comment", "0", "1"))

    def test_dispatch_run_honours_the_input(self) -> None:
        self.assertTrue(gate.should_build_rccl("workflow_dispatch", "0", "1"))
        self.assertFalse(gate.should_build_rccl("workflow_dispatch", "1", "0"))


class CleanupConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        (Path(self.root) / "configs").mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _touch(self, name: str) -> Path:
        path = Path(self.root) / "configs" / name
        path.write_text("{}", encoding="utf-8")
        return path

    def test_removes_a_per_run_copy(self) -> None:
        path = self._touch("ci_detect_run_42.json")
        self.assertTrue(gate.cleanup_run_config(str(path), self.root))
        self.assertFalse(path.exists())

    def test_refuses_the_shared_production_config(self) -> None:
        path = self._touch("ci_detect_prod.json")
        self.assertFalse(gate.cleanup_run_config(str(path), self.root))
        self.assertTrue(path.exists())

    def test_refuses_paths_outside_the_configs_directory(self) -> None:
        outside = Path(self.root) / "ci_detect_run_42.json"
        outside.write_text("{}", encoding="utf-8")
        self.assertFalse(gate.cleanup_run_config(str(outside), self.root))
        self.assertTrue(outside.exists())

    def test_empty_path_is_a_no_op(self) -> None:
        self.assertFalse(gate.cleanup_run_config("", self.root))


class OutputTest(unittest.TestCase):
    def test_values_are_flattened_to_a_single_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            gate.write_outputs(output, {"desc": "Build job\nfailure", "state": "error"})
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "desc=Build job failure\nstate=error\n",
            )

    def test_comment_includes_the_report_body(self) -> None:
        verdict = gate.Verdict(True, "success", "Perf gate passed")
        body = gate.render_comment(
            "c" * 40, verdict, "https://example/run", "| a | b |"
        )
        self.assertIn("### RCCL Perf Regression Gate", body)
        self.assertIn("c" * 40, body)
        self.assertIn("| a | b |", body)

    def test_long_status_descriptions_are_truncated(self) -> None:
        api = FakeApi({})
        gate.set_commit_status(
            api, REPOSITORY, "d" * 40, "ctx", "failure", "x" * 300, "https://example"
        )
        _, payload = api.posts[0]
        self.assertEqual(len(payload["description"]), gate.MAXIMUM_DESCRIPTION_LENGTH)


if __name__ == "__main__":
    unittest.main()
