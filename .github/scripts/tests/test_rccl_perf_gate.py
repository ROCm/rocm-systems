#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for rccl_perf_gate.py.

Runs standalone (no pip install needed) and under pytest via conftest.py.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rccl_perf_gate as gate  # noqa: E402

COMMAND = "/perf-regression"
REPOSITORY = "ROCm/rocm-systems"
FORK = "outsider/rocm-systems"
# A real collision pair: both abbreviate to 7065f5e, found in 25 ms. Mixed
# characters also stop a prefix check passing for a substring check.
HEAD_SHA = "7065f5e55eb81912b4144bdba1da53b8ff9de059"
COLLIDING_SHA = "7065f5eab40bc198b0a4d2811c105e1c4780f54d"
SHARED_PREFIX = "7065f5e"


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
    def test_bare_command_is_accepted_with_surrounding_whitespace(self) -> None:
        for body in (COMMAND, f"  {COMMAND}  ", f"{COMMAND}\r\n", f"\n{COMMAND}\n"):
            with self.subTest(body=body):
                self.assertEqual(gate.parse_gate_command(body, COMMAND), "")

    def test_a_vouched_sha_is_returned_lowercased(self) -> None:
        for suffix, expected in (
            ("abc1234", "abc1234"),
            ("ABC1234", "abc1234"),
            ("a" * 40, "a" * 40),
            ("  DEADBEEF  ", "deadbeef"),
        ):
            with self.subTest(suffix=suffix):
                body = f"{COMMAND} {suffix}"
                self.assertEqual(gate.parse_gate_command(body, COMMAND), expected)

    def test_substrings_and_quotations_are_rejected(self) -> None:
        near_misses = [
            f"{COMMAND} please",
            f"`{COMMAND}`",
            f"> {COMMAND}",
            f"see {COMMAND} docs",
            f"{COMMAND}\nrm -rf /",
            # Would parse as a valid vouch without the multi-line guard.
            f"{COMMAND}\n{HEAD_SHA}",
            f"lgtm\n{COMMAND} {HEAD_SHA}",
            "/perf-regressions",
            "",
        ]
        for body in near_misses:
            with self.subTest(body=body):
                self.assertIsNone(gate.parse_gate_command(body, COMMAND))

    def test_malformed_sha_arguments_are_rejected(self) -> None:
        near_misses = [
            f"{COMMAND} abc123",
            f"{COMMAND} {'a' * 41}",
            f"{COMMAND} abcdefg",
            f"{COMMAND} abc1234 extra",
            f"{COMMAND} abc1234; rm -rf /",
            f"{COMMAND} --dry-run",
        ]
        for body in near_misses:
            with self.subTest(body=body):
                self.assertIsNone(gate.parse_gate_command(body, COMMAND))

    def test_a_bare_command_is_distinguishable_from_a_refusal(self) -> None:
        self.assertIsNotNone(gate.parse_gate_command(COMMAND, COMMAND))
        self.assertIsNone(gate.parse_gate_command("hello", COMMAND))


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

    def test_rejects_shell_metacharacters_git_would_accept(self) -> None:
        # These reach a shell in ROCm/cvs through `sbatch --export`.
        for ref in (
            "develop,BASH_ENV=/tmp/x",
            "feature,x",
            "a=b",
            "develop`id`",
            "develop$(id)",
            "develop|id",
            "develop&id",
            "develop'x",
        ):
            with self.subTest(ref=ref):
                self.assertEqual(
                    gate.git_output("check-ref-format", f"refs/heads/{ref}"), ""
                )
                with self.assertRaises(ValueError):
                    gate.validate_ref(ref)


class ResolveRequestTest(unittest.TestCase):
    def _resolve(self, api: FakeApi, event: dict) -> gate.GateRequest:
        return gate.resolve_request(
            "issue_comment", event, api, REPOSITORY, COMMAND, {}
        )

    def _api(
        self,
        permission: object,
        base_ref: str = "develop",
        head_repo: object = REPOSITORY,
        author: str = "octocat",
    ) -> FakeApi:
        return FakeApi(
            {
                permission_path(): permission,
                f"/repos/{REPOSITORY}/pulls/9950": {
                    "head": {
                        "sha": HEAD_SHA,
                        "repo": None if head_repo is None else {"full_name": head_repo},
                    },
                    "base": {"ref": base_ref},
                    "user": {"login": author},
                },
            }
        )

    def test_write_permission_authorizes_and_resolves_context(self) -> None:
        request = self._resolve(
            self._api({"permission": "write"}), comment_event(COMMAND)
        )
        self.assertTrue(request.authorized)
        self.assertEqual(request.head_sha, HEAD_SHA)
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
                self.assertEqual(request.deny_reason, "not_writer")

    def test_permission_read_failure_fails_closed_and_is_loud(self) -> None:
        error = urllib.error.HTTPError(
            "https://api.github.com", 403, "Forbidden", {}, None
        )
        request = self._resolve(self._api(error), comment_event(COMMAND))
        self.assertFalse(request.authorized)
        self.assertEqual(request.level, "error")
        self.assertEqual(request.deny_reason, "perm_read_error")

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

    def test_same_repo_author_may_run_the_gate_on_their_own_branch(self) -> None:
        api = self._api({"permission": "write"}, author="octocat")
        request = self._resolve(api, comment_event(COMMAND))
        self.assertTrue(request.authorized)
        self.assertEqual(request.vouched_sha, "")

    def test_same_repo_sha_is_still_checked_when_supplied(self) -> None:
        api = self._api({"permission": "write"})
        request = self._resolve(api, comment_event(f"{COMMAND} bbbbbbb"))
        self.assertFalse(request.authorized)
        self.assertEqual(request.deny_reason, "sha_stale")

    def test_fork_author_cannot_vouch_for_themselves(self) -> None:
        api = self._api({"permission": "write"}, head_repo=FORK, author="octocat")
        request = self._resolve(api, comment_event(f"{COMMAND} {HEAD_SHA}"))
        self.assertFalse(request.authorized)
        self.assertEqual(request.deny_reason, "fork_author_self")

    def test_fork_without_a_sha_is_told_the_syntax(self) -> None:
        api = self._api({"permission": "write"}, head_repo=FORK, author="outsider")
        request = self._resolve(api, comment_event(COMMAND))
        self.assertFalse(request.authorized)
        self.assertEqual(request.deny_reason, "fork_needs_vouch")
        self.assertIn(f"{COMMAND} {HEAD_SHA}", request.reason)

    def test_fork_with_a_stale_sha_is_told_the_current_head(self) -> None:
        api = self._api({"permission": "write"}, head_repo=FORK, author="outsider")
        request = self._resolve(api, comment_event(f"{COMMAND} {COLLIDING_SHA}"))
        self.assertFalse(request.authorized)
        self.assertEqual(request.deny_reason, "sha_stale")
        self.assertIn(HEAD_SHA, request.reason)

    def test_an_abbreviated_vouch_does_not_authorize_a_fork(self) -> None:
        """A fork author can author two commits sharing a 7-char prefix."""
        for vouched in (SHARED_PREFIX, HEAD_SHA[:12], HEAD_SHA[:39]):
            with self.subTest(vouched=vouched):
                api = self._api(
                    {"permission": "write"}, head_repo=FORK, author="outsider"
                )
                request = self._resolve(api, comment_event(f"{COMMAND} {vouched}"))
                self.assertFalse(request.authorized)
                self.assertEqual(request.deny_reason, "sha_stale")
                self.assertIn(HEAD_SHA, request.reason)

    def test_a_vouch_must_be_the_whole_sha_not_a_substring_of_it(self) -> None:
        api = self._api({"permission": "write"}, head_repo=FORK, author="outsider")
        request = self._resolve(api, comment_event(f"{COMMAND} {HEAD_SHA[3:]}"))
        self.assertFalse(request.authorized)
        self.assertEqual(request.deny_reason, "sha_stale")

    def test_fork_with_a_matching_sha_runs(self) -> None:
        for vouched in (HEAD_SHA, HEAD_SHA.upper()):
            with self.subTest(vouched=vouched):
                api = self._api(
                    {"permission": "write"}, head_repo=FORK, author="outsider"
                )
                request = self._resolve(api, comment_event(f"{COMMAND} {vouched}"))
                self.assertTrue(request.authorized)
                self.assertEqual(request.head_repo, FORK)
                self.assertEqual(request.vouched_sha, HEAD_SHA)

    def test_fork_requester_without_write_access_is_denied_first(self) -> None:
        api = self._api({"permission": "read"}, head_repo=FORK, author="outsider")
        request = self._resolve(api, comment_event(f"{COMMAND} {HEAD_SHA}"))
        self.assertFalse(request.authorized)
        self.assertEqual(request.deny_reason, "not_writer")

    def test_deleted_fork_is_refused_outright(self) -> None:
        api = self._api({"permission": "admin"}, head_repo=None)
        request = self._resolve(api, comment_event(f"{COMMAND} {HEAD_SHA}"))
        self.assertFalse(request.authorized)
        self.assertEqual(request.deny_reason, "head_repo_gone")

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
        self.assertEqual(verdict.state, "success")

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

    def test_skipped_detector_is_an_error(self) -> None:
        verdict = gate.compute_verdict("success", "skipped", "")
        self.assertEqual(verdict.state, "error")

    def test_detector_evicted_by_another_pr_stays_pending(self) -> None:
        verdict = gate.compute_verdict("success", "cancelled", "")
        self.assertEqual(verdict.state, "pending")

    def test_cancelled_build_stays_pending_instead_of_going_silent(self) -> None:
        # Publishing nothing would leave resolve's "queued" pending status on
        # the PR forever with no explanation.
        verdict = gate.compute_verdict("cancelled", "skipped", "")
        self.assertEqual(verdict.state, "pending")
        self.assertIn("re-request", verdict.description)

    def test_a_cancelled_build_never_reports_failure(self) -> None:
        # Nothing failed and nothing was measured, so the gate must not blame
        # the PR for a cancellation it did not cause.
        for detect_result in ("skipped", "cancelled", "success"):
            with self.subTest(detect_result=detect_result):
                verdict = gate.compute_verdict("cancelled", detect_result, "")
                self.assertNotIn(verdict.state, ("failure", "error"))


class GateRequestOutputTest(unittest.TestCase):
    def test_authorized_is_only_true_for_an_authorized_request(self) -> None:
        allowed = gate.GateRequest(True, "ok", head_repo=REPOSITORY)
        self.assertEqual(allowed.outputs()["authorized"], "true")
        for denied in (
            gate.GateRequest(False, "not a writer", deny_reason="not_writer"),
            gate.GateRequest(False, "no vouch", deny_reason="fork_needs_vouch"),
            gate.GateRequest(False, "boom", level="error"),
        ):
            with self.subTest(reason=denied.reason):
                self.assertEqual(denied.outputs()["authorized"], "false")

    def test_vouched_sha_and_deny_reason_are_exported(self) -> None:
        request = gate.GateRequest(
            True, "ok", head_repo=FORK, vouched_sha=HEAD_SHA, deny_reason=""
        )
        self.assertEqual(request.outputs()["vouched_sha"], HEAD_SHA)
        self.assertEqual(request.outputs()["deny_reason"], "")

    def test_a_denied_request_never_leaks_a_vouched_sha(self) -> None:
        denied = gate.GateRequest(False, "stale", deny_reason="sha_stale")
        self.assertEqual(denied.outputs()["vouched_sha"], "")


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


class FetchBaseTest(unittest.TestCase):
    """A fork checkout leaves `origin` pointing at the fork, not upstream."""

    def _argv(self, base_ref: str, remote: str) -> list[str]:
        with mock.patch.object(gate.subprocess, "run") as run:
            gate.fetch_base(base_ref, remote)
        return list(run.call_args.args[0])

    def test_fetches_the_upstream_url_not_the_origin_remote(self) -> None:
        argv = self._argv("develop", "https://github.com/ROCm/rocm-systems")
        self.assertEqual(argv[:2], ["git", "fetch"])
        self.assertIn("https://github.com/ROCm/rocm-systems", argv)
        self.assertNotIn("origin", argv)

    def test_force_overwrites_the_forks_stale_remote_tracking_ref(self) -> None:
        argv = self._argv("develop", "https://example/upstream")
        self.assertIn("+refs/heads/develop:refs/remotes/origin/develop", argv)

    def test_a_failed_fetch_is_not_swallowed(self) -> None:
        with mock.patch.object(gate.subprocess, "run") as run:
            gate.fetch_base("develop", "https://example/upstream")
        self.assertTrue(run.call_args.kwargs["check"])


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

    def test_removes_this_runs_own_copy(self) -> None:
        path = self._touch("ci_detect_run_42.json")
        self.assertTrue(gate.cleanup_run_config(str(path), self.root, "42"))
        self.assertFalse(path.exists())

    def test_refuses_the_shared_production_config(self) -> None:
        path = self._touch("ci_detect_prod.json")
        self.assertFalse(gate.cleanup_run_config(str(path), self.root, "42"))
        self.assertTrue(path.exists())

    def test_refuses_another_runs_live_copy(self) -> None:
        path = self._touch("ci_detect_run_999.json")
        self.assertFalse(gate.cleanup_run_config(str(path), self.root, "42"))
        self.assertTrue(path.exists())

    def test_refuses_paths_outside_the_configs_directory(self) -> None:
        outside = Path(self.root) / "ci_detect_run_42.json"
        outside.write_text("{}", encoding="utf-8")
        self.assertFalse(gate.cleanup_run_config(str(outside), self.root, "42"))
        self.assertTrue(outside.exists())

    def test_refuses_a_traversal_dressed_up_as_this_runs_name(self) -> None:
        traversal = f"{self.root}/configs/../../ci_detect_run_42.json"
        self.assertFalse(gate.cleanup_run_config(traversal, self.root, "42"))

    def test_empty_path_is_a_no_op(self) -> None:
        self.assertFalse(gate.cleanup_run_config("", self.root, "42"))


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
        verdict = gate.Verdict("success", "Perf gate passed")
        body = gate.render_comment(
            "c" * 40, verdict, "https://example/run", "| a | b |"
        )
        self.assertIn("### RCCL Perf Regression Gate", body)
        self.assertIn("c" * 40, body)
        self.assertIn("| a | b |", body)

    def test_a_missing_report_is_called_out_instead_of_shown_as_clean(self) -> None:
        verdict = gate.Verdict("success", "Perf gate passed")
        body = gate.render_comment("c" * 40, verdict, "https://example/run")
        self.assertIn("No perf table", body)

    def test_an_oversized_report_is_truncated_below_the_comment_limit(self) -> None:
        verdict = gate.Verdict("success", "Perf gate passed")
        body = gate.render_comment(
            "c" * 40, verdict, "https://example/run", "x" * 100000
        )
        self.assertLess(len(body), 65536)
        self.assertIn("Report truncated", body)

    def test_long_status_descriptions_are_truncated(self) -> None:
        api = FakeApi({})
        gate.set_commit_status(
            api, REPOSITORY, "d" * 40, "ctx", "failure", "x" * 300, "https://example"
        )
        _, payload = api.posts[0]
        self.assertEqual(len(payload["description"]), gate.MAXIMUM_DESCRIPTION_LENGTH)


class FakeResponse:
    """Minimal stand-in for the context manager urlopen returns."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exception: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


class GitHubApiContractTest(unittest.TestCase):
    """Drives the real GitHubApi. FakeApi cannot catch a wrong URL or verb."""

    def setUp(self) -> None:
        self.api = gate.GitHubApi(REPOSITORY, "tok3n", "https://api.github.com")
        patcher = mock.patch.object(gate.urllib.request, "urlopen")
        self.urlopen = patcher.start()
        self.addCleanup(patcher.stop)
        self.urlopen.return_value = FakeResponse(b"{}")

    def _request(self) -> object:
        return self.urlopen.call_args.args[0]

    def _header(self, name: str) -> str:
        headers = {k.lower(): v for k, v in self._request().headers.items()}
        return headers[name.lower()]

    def _post_a_status(self) -> None:
        gate.set_commit_status(
            self.api, REPOSITORY, "d" * 40, "ctx", "success", "ok", "https://example"
        )

    def test_status_post_targets_the_statuses_endpoint(self) -> None:
        self._post_a_status()
        self.assertEqual(
            self._request().full_url,
            f"https://api.github.com/repos/{REPOSITORY}/statuses/{'d' * 40}",
        )
        self.assertEqual(self._request().method, "POST")

    def test_status_post_sends_exactly_the_documented_payload_keys(self) -> None:
        self._post_a_status()
        payload = json.loads(self._request().data)
        self.assertEqual(
            set(payload), {"state", "context", "target_url", "description"}
        )
        self.assertEqual(payload["state"], "success")
        self.assertEqual(payload["context"], "ctx")
        self.assertEqual(payload["description"], "ok")

    def test_permission_lookup_targets_the_collaborator_endpoint(self) -> None:
        self.urlopen.return_value = FakeResponse(b'{"permission": "write"}')
        permission = gate.collaborator_permission(self.api, REPOSITORY, "oct/cat")
        self.assertEqual(permission, "write")
        self.assertEqual(
            self._request().full_url,
            f"https://api.github.com/repos/{REPOSITORY}"
            "/collaborators/oct%2Fcat/permission",
        )
        self.assertEqual(self._request().method, "GET")
        self.assertIsNone(self._request().data)

    def test_every_request_is_authenticated_and_version_pinned(self) -> None:
        self.api.get("/rate_limit")
        self.assertEqual(self._header("Authorization"), "Bearer tok3n")
        self.assertEqual(self._header("Accept"), "application/vnd.github+json")
        self.assertEqual(self._header("X-GitHub-Api-Version"), "2022-11-28")

    def test_requests_cannot_hang_a_job_forever(self) -> None:
        self.api.get("/rate_limit")
        self.assertEqual(
            self.urlopen.call_args.kwargs["timeout"], gate.API_TIMEOUT_SECONDS
        )

    def test_a_trailing_slash_on_the_api_url_does_not_double_up(self) -> None:
        gate.GitHubApi(REPOSITORY, "tok3n", "https://api.github.com/").get("/x")
        self.assertEqual(self._request().full_url, "https://api.github.com/x")

    def test_the_constructor_refuses_unusable_configuration(self) -> None:
        unusable = [
            ("not-a-repo", "tok3n", "https://api.github.com"),
            (REPOSITORY, "", "https://api.github.com"),
            (REPOSITORY, "tok3n", "http://api.github.com"),
        ]
        for repository, token, url in unusable:
            with self.subTest(repository=repository, token=token, url=url):
                with self.assertRaises(ValueError):
                    gate.GitHubApi(repository, token, url)


class MainCommandTest(unittest.TestCase):
    """Drives main(): argv, environment, GITHUB_OUTPUT contents and exit code."""

    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, True)
        self.output = self.directory / "github_output"
        self.comment = self.directory / "comment.md"
        self.api = FakeApi({})
        patcher = mock.patch.object(gate, "_api", lambda: self.api)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _main(self, argv: list[str], **environment: str) -> int:
        base = {"GITHUB_REPOSITORY": REPOSITORY, "STATUS_CONTEXT": "ctx"}
        with mock.patch.dict(gate.os.environ, {**base, **environment}):
            return gate.main(argv)

    def _outputs(self) -> dict[str, str]:
        lines = self.output.read_text(encoding="utf-8").splitlines()
        return dict(line.split("=", 1) for line in lines)

    def _verdict(self, **environment: str) -> int:
        base = {
            "HEAD_SHA": "e" * 40,
            "BUILD_RESULT": "success",
            "DETECT_RESULT": "success",
            "DETECT_CODE": "0",
        }
        return self._main(
            ["verdict", "--comment", str(self.comment), "--output", str(self.output)],
            **{**base, **environment},
        )

    def test_verdict_stamps_the_status_for_a_pull_request_run(self) -> None:
        self.assertEqual(self._verdict(PR_NUMBER="9950"), 0)
        path, payload = self.api.posts[0]
        self.assertEqual(path, f"/repos/{REPOSITORY}/statuses/{'e' * 40}")
        self.assertEqual(payload["state"], "success")
        self.assertEqual(self._outputs()["state"], "success")

    def test_verdict_never_stamps_a_status_for_a_dispatch_run(self) -> None:
        self.assertEqual(self._verdict(PR_NUMBER=""), 0)
        self.assertEqual(self.api.posts, [])

    def test_verdict_of_a_cancelled_build_stamps_pending_on_the_pr(self) -> None:
        self.assertEqual(self._verdict(PR_NUMBER="9950", BUILD_RESULT="cancelled"), 0)
        path, payload = self.api.posts[0]
        self.assertEqual(path, f"/repos/{REPOSITORY}/statuses/{'e' * 40}")
        self.assertEqual(payload["state"], "pending")

    def test_enforce_fails_only_on_a_failure_or_error_state(self) -> None:
        for state, expected in (
            ("success", 0),
            ("pending", 0),
            ("", 0),
            ("failure", 1),
            ("error", 1),
        ):
            with self.subTest(state=state):
                self.assertEqual(self._main(["enforce"], STATE=state), expected)

    def _resolve(self, event: dict, **environment: str) -> int:
        event_path = self.directory / "event.json"
        event_path.write_text(json.dumps(event), encoding="utf-8")
        return self._main(
            [
                "resolve",
                "--event",
                str(event_path),
                "--comment",
                str(self.comment),
                "--output",
                str(self.output),
            ],
            GITHUB_EVENT_NAME="issue_comment",
            PERF_COMMAND=COMMAND,
            **environment,
        )

    def test_resolve_denies_a_non_writer_and_writes_the_explanation(self) -> None:
        self.api.responses[permission_path()] = {"permission": "read"}
        self.assertEqual(self._resolve(comment_event(COMMAND)), 0)
        self.assertEqual(self._outputs()["authorized"], "false")
        self.assertEqual(self._outputs()["deny_reason"], "not_writer")
        self.assertIn("write access", self.comment.read_text(encoding="utf-8"))

    def test_resolve_exits_non_zero_when_it_cannot_read_the_permission(self) -> None:
        self.api.responses[permission_path()] = urllib.error.HTTPError(
            "https://api.github.com", 403, "Forbidden", {}, None
        )
        self.assertEqual(self._resolve(comment_event(COMMAND)), 1)
        self.assertEqual(self._outputs()["authorized"], "false")
        self.assertEqual(self._outputs()["deny_reason"], "perm_read_error")

    def test_resolve_stays_quiet_when_the_comment_is_not_the_command(self) -> None:
        self.assertEqual(self._resolve(comment_event("looks good to me")), 0)
        self.assertEqual(self._outputs()["deny_reason"], "")
        self.assertFalse(self.comment.exists())

    def test_resolve_still_explains_itself_when_the_pr_lookup_blows_up(self) -> None:
        # Without containment the PR gets a red run and no explanation at all.
        self.api.responses[permission_path()] = {"permission": "admin"}
        self.api.responses[f"/repos/{REPOSITORY}/pulls/9950"] = urllib.error.HTTPError(
            "https://api.github.com", 502, "Bad Gateway", {}, None
        )
        self.assertEqual(self._resolve(comment_event(COMMAND)), 1)
        self.assertEqual(self._outputs()["authorized"], "false")
        self.assertEqual(self._outputs()["deny_reason"], "resolve_error")
        self.assertIn("could not resolve", self.comment.read_text(encoding="utf-8"))

    def test_resolve_contains_an_error_type_main_does_not_catch(self) -> None:
        self.api.responses[permission_path()] = {"permission": "admin"}
        event = comment_event(COMMAND)
        event["issue"]["number"] = None
        self.assertEqual(self._resolve(event), 1)
        self.assertEqual(self._outputs()["deny_reason"], "resolve_error")

    def test_build_emits_the_config_path_before_it_copies(self) -> None:
        # A copy that dies part-way still has to leave a cleanable path behind.
        configs = self.directory / "configs"
        configs.mkdir()
        code = self._main(
            ["build", "--output", str(self.output)],
            GITHUB_EVENT_NAME="issue_comment",
            RCCL_CHANGED="1",
            RCCL_CI_ROOT=str(self.directory),
            GITHUB_RUN_ID="42",
            SOURCE_CONFIG_JSON=str(self.directory / "absent.json"),
        )
        self.assertEqual(code, 1)
        self.assertEqual(
            self._outputs()["config_json"], str(configs / "ci_detect_run_42.json")
        )

    def test_build_without_rccl_changes_passes_the_shared_config_through(self) -> None:
        shared = "/it-share/rccl-ci/configs/ci_detect_prod.json"
        code = self._main(
            ["build", "--output", str(self.output)],
            GITHUB_EVENT_NAME="issue_comment",
            RCCL_CHANGED="0",
            RCCL_CI_ROOT=str(self.directory),
            SOURCE_CONFIG_JSON=shared,
        )
        self.assertEqual(code, 0)
        self.assertEqual(self._outputs()["config_json"], shared)

    def test_cleanup_config_refuses_a_path_this_run_did_not_create(self) -> None:
        configs = self.directory / "configs"
        configs.mkdir()
        victim = configs / "ci_detect_prod.json"
        victim.write_text("{}", encoding="utf-8")
        code = self._main(
            ["cleanup-config"],
            CONFIG_JSON=str(victim),
            RCCL_CI_ROOT=str(self.directory),
            GITHUB_RUN_ID="42",
        )
        self.assertEqual(code, 0)
        self.assertTrue(victim.exists())


if __name__ == "__main__":
    # Skipping instead would hide the whole authorization surface behind a tick.
    if not shutil.which("git"):
        sys.exit("git is required: it backs validate_ref in every resolve test")
    unittest.main()
