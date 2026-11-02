# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for rccl_resolve_coco_run.py."""

import contextlib
import http.client
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import tempfile
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import rccl_resolve_coco_run as resolve

SCHEDULED_WORKFLOW = (
    Path(__file__).resolve().parents[2] / "workflows" / "rccl-coco-scheduled.yml"
)


class CronConsistencyTest(unittest.TestCase):
    def test_workflow_crons_are_all_recognised(self):
        """schedule_to_mode raises on an unknown cron, failing every scheduled run."""
        text = SCHEDULED_WORKFLOW.read_text(encoding="utf-8")
        crons = set(re.findall(r"^\s*-\s*cron:\s*'([^']+)'", text, re.MULTILINE))
        self.assertTrue(crons, f"no cron entries found in {SCHEDULED_WORKFLOW}")
        self.assertEqual(crons, {resolve.WEEKDAY_NIGHTLY_CRON, resolve.SATURDAY_CRON})


class BaselinePinWiringTest(unittest.TestCase):
    """The PR workflow must reach the action's ``require_baseline_sha`` input.

    GitHub only warns about an input the action does not declare, so a
    misspelt name would silently turn the requirement off.
    """

    WORKFLOWS = Path(__file__).resolve().parents[2] / "workflows"
    ACTION = (
        Path(__file__).resolve().parents[2]
        / "actions"
        / "rccl-resolve-coco-run"
        / "action.yml"
    )

    def test_the_action_declares_the_input_and_defaults_it_off(self):
        text = self.ACTION.read_text(encoding="utf-8")
        self.assertRegex(
            text,
            r"(?m)^  require_baseline_sha:\n(?:    .*\n)*?    default: 'false'\n",
        )
        self.assertIn("REQUIRE_BASELINE_SHA: ${{ inputs.require_baseline_sha }}", text)

    def test_the_pr_workflow_requires_the_pin(self):
        """Only PR runs pin the baseline, so only this workflow needs to require it."""
        text = (self.WORKFLOWS / "rccl-coco-pr.yml").read_text(encoding="utf-8")
        self.assertRegex(text, r"(?m)^\s+require_baseline_sha: 'true'$")


class ScheduleToModeTest(unittest.TestCase):
    def test_weekday_cron_is_nightly(self):
        self.assertEqual(
            resolve.schedule_to_mode("47 3 * * 1-5", day_of_month=10),
            "nightly",
        )

    def test_saturday_before_third_week_is_weekly(self):
        self.assertEqual(
            resolve.schedule_to_mode("47 3 * * 6", day_of_month=14),
            "weekly",
        )

    def test_saturday_third_week_is_monthly(self):
        self.assertEqual(
            resolve.schedule_to_mode("47 3 * * 6", day_of_month=18),
            "monthly",
        )

    def test_unknown_schedule_raises(self):
        with self.assertRaises(ValueError):
            resolve.schedule_to_mode("0 0 * * *", day_of_month=1)


class ResolveModeAndTriggerTest(unittest.TestCase):
    def test_pr_number_wins(self):
        mode, kind = resolve.resolve_mode_and_trigger(
            pr_number="42",
            event_name="schedule",
            schedule="47 3 * * 1-5",
            periodic_mode="nightly",
        )
        self.assertEqual((mode, kind), ("pr", "ci"))

    def test_schedule_event(self):
        mode, kind = resolve.resolve_mode_and_trigger(
            pr_number="",
            event_name="schedule",
            schedule="47 3 * * 1-5",
            periodic_mode="nightly",
            day_of_month=3,
        )
        self.assertEqual((mode, kind), ("nightly", "scheduled"))

    def test_dispatch_uses_periodic_mode(self):
        mode, kind = resolve.resolve_mode_and_trigger(
            pr_number="",
            event_name="workflow_dispatch",
            schedule="",
            periodic_mode="weekly",
        )
        self.assertEqual((mode, kind), ("weekly", "manual"))


class BuildCocoArgsTest(unittest.TestCase):
    def test_pr_skips_baselines_not_prune(self):
        args = resolve.build_coco_args(mode="pr", dry_run=False)
        self.assertIn("--skip-baselines", args)
        self.assertNotIn("--prune-build", args)

    def test_nightly_prunes(self):
        args = resolve.build_coco_args(mode="nightly", dry_run=False)
        self.assertIn("--prune-build", args)
        self.assertNotIn("--skip-baselines", args)


class ResolveRunTest(unittest.TestCase):
    def test_rejects_unknown_manual_mode(self):
        with self.assertRaises(ValueError):
            resolve.resolve_run(
                pr_number="",
                pr_sha="",
                pr_repo="",
                event_name="workflow_dispatch",
                schedule="",
                periodic_mode="not-a-mode",
                dry_run=False,
                coco_sha="abc123",
            )


HEAD = "a" * 40
MERGE_BASE = "b" * 40
API = "https://api.github.com/repos/ROCm/rocm-systems"


def fake_github(merge_base=MERGE_BASE, base_ref="develop", head=HEAD):
    """A GitHub REST stub serving one PR and its compare against the base branch."""
    compare = f"{API}/compare/{urllib.parse.quote(base_ref, safe='/')}...{head}"
    responses = {
        f"{API}/pulls/42": {"base": {"ref": base_ref}},
        compare: {"merge_base_commit": {"sha": merge_base}},
    }
    return responses.__getitem__


def merge_base_of(**kwargs):
    return resolve.resolve_merge_base(
        pr_repo="ROCm/rocm-systems",
        pr_number="42",
        pr_sha=kwargs.pop("pr_sha", HEAD),
        get=kwargs.pop("get", fake_github()),
    )


class BaselineShaArgsTest(unittest.TestCase):
    def test_pr_args_pin_the_baseline_to_the_given_commit(self):
        args = resolve.build_coco_args(
            mode="pr", dry_run=False, baseline_sha=MERGE_BASE
        )
        self.assertIn(f"--baseline-sha {MERGE_BASE}", args)

    def test_pr_args_without_a_commit_keep_the_configured_baseline(self):
        self.assertNotIn(
            "--baseline-sha", resolve.build_coco_args(mode="pr", dry_run=False)
        )

    def test_periodic_args_never_pin(self):
        args = resolve.build_coco_args(
            mode="nightly", dry_run=False, baseline_sha=MERGE_BASE
        )
        self.assertNotIn("--baseline-sha", args)


class ResolveMergeBaseTest(unittest.TestCase):
    def test_returns_the_merge_base_with_the_prs_base_branch(self):
        """The baseline arm is built here, so a PR is measured only against code it contains."""
        self.assertEqual(merge_base_of(), MERGE_BASE)

    def test_uses_the_prs_own_base_branch(self):
        self.assertEqual(
            merge_base_of(get=fake_github(base_ref="rccl-develop")), MERGE_BASE
        )

    def test_a_base_branch_with_reserved_characters_is_quoted(self):
        """An unquoted ``#`` would start a URL fragment and swallow the head SHA."""
        requested = []
        serve = fake_github(base_ref="release/foo#1")

        def get(url):
            requested.append(url)
            return serve(url)

        self.assertEqual(merge_base_of(get=get), MERGE_BASE)
        self.assertIn("/compare/release/foo%231...", requested[-1])

    def test_a_head_that_is_its_own_merge_base_is_refused(self):
        """Both arms on one commit would compare the PR with itself and always pass."""
        with self.assertRaisesRegex(ValueError, "no change to measure"):
            merge_base_of(get=fake_github(merge_base=HEAD))

    def test_an_uppercase_head_is_compared_in_lowercase(self):
        """The workflow accepts an uppercase head SHA; GitHub returns lowercase ones."""
        with self.assertRaisesRegex(ValueError, "no change to measure"):
            merge_base_of(pr_sha=HEAD.upper(), get=fake_github(merge_base=HEAD))

    def test_a_malformed_merge_base_is_refused(self):
        with self.assertRaises(ValueError):
            merge_base_of(get=fake_github(merge_base="deadbeef"))

    def test_an_unexpected_response_names_the_pr_and_the_missing_field(self):
        with self.assertRaisesRegex(ValueError, r"PR #42 lacks 'base'"):
            merge_base_of(get=lambda url: {})


def http_error(code):
    return urllib.error.HTTPError(
        f"{API}/pulls/42", code, "reason", {}, io.BytesIO(b"")
    )


def direct_urlopen(request, timeout):
    """``urlopen`` through a fresh opener with no proxy.

    ``urllib.request.urlopen`` builds its default opener once, from whatever
    proxy variables the environment had then, so a test that talks to a local
    socket must not depend on it.
    """
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(request, timeout=timeout)


def response(payload):
    """A urlopen result: a context manager whose body is *payload* as JSON."""
    body = mock.MagicMock()
    body.__enter__.return_value = io.BytesIO(json.dumps(payload).encode())
    return body


@mock.patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"})
@mock.patch("time.sleep")
class GithubGetTest(unittest.TestCase):
    URL = f"{API}/pulls/42"

    def get(self, side_effect):
        with mock.patch("urllib.request.urlopen", side_effect=side_effect) as urlopen:
            try:
                return resolve.github_get(self.URL), urlopen
            except resolve.GitHubApiError as exc:
                return exc, urlopen

    def test_a_missing_pr_fails_at_once_naming_the_request(self, _sleep):
        """A 404 cannot recover: no retries, and the message says what was requested."""
        error, urlopen = self.get([http_error(404)])
        self.assertIsInstance(error, resolve.GitHubApiError)
        self.assertNotIsInstance(error, resolve.TransientGitHubError)
        self.assertEqual(urlopen.call_count, 1)
        self.assertIn(f"GET {self.URL}", str(error))
        self.assertIn("HTTP 404", str(error))

    def test_a_server_error_is_retried_until_it_succeeds(self, _sleep):
        payload, urlopen = self.get([http_error(502), response({"ok": True})])
        self.assertEqual(payload, {"ok": True})
        self.assertEqual(urlopen.call_count, 2)

    def test_a_persistent_server_error_gives_up_with_the_request_named(self, _sleep):
        error, urlopen = self.get([http_error(500)] * 3)
        self.assertEqual(urlopen.call_count, 3)
        self.assertIn(f"GET {self.URL}", str(error))
        self.assertIn("HTTP 500", str(error))

    def test_a_rate_limit_is_retried(self, _sleep):
        payload, urlopen = self.get([http_error(429), response({"ok": True})])
        self.assertEqual(payload, {"ok": True})
        self.assertEqual(urlopen.call_count, 2)

    def test_a_timeout_is_retried_and_reported_as_an_api_error(self, _sleep):
        """``TimeoutError`` is not a ``URLError``, so it once escaped as a traceback."""
        error, urlopen = self.get([TimeoutError("timed out")] * 3)
        self.assertEqual(urlopen.call_count, 3)
        self.assertIsInstance(error, resolve.GitHubApiError)
        self.assertIn("timed out", str(error))

    def test_a_connection_reset_is_retried_until_it_succeeds(self, _sleep):
        """urllib only converts errors raised while sending the request.

        A server that resets the connection after the request was sent raises
        ``RemoteDisconnected`` from ``getresponse``, which is neither a
        ``URLError`` nor a ``TimeoutError``; it once escaped as a traceback.
        """
        payload, urlopen = self.get(
            [http.client.RemoteDisconnected("closed"), response({"ok": True})]
        )
        self.assertEqual(payload, {"ok": True})
        self.assertEqual(urlopen.call_count, 2)

    def test_a_persistent_reset_is_an_api_error_naming_the_request(self, _sleep):
        error, urlopen = self.get([ConnectionResetError("reset")] * 3)
        self.assertEqual(urlopen.call_count, 3)
        self.assertIsInstance(error, resolve.GitHubApiError)
        self.assertIn(f"GET {self.URL}", str(error))
        self.assertIn("ConnectionResetError", str(error))

    def test_a_truncated_body_is_retried(self, _sleep):
        """``IncompleteRead`` is raised while reading the body, after the status."""
        truncated = mock.MagicMock()
        truncated.__enter__.return_value.read.side_effect = http.client.IncompleteRead(
            b"{", 100
        )
        payload, urlopen = self.get([truncated, response({"ok": True})])
        self.assertEqual(payload, {"ok": True})
        self.assertEqual(urlopen.call_count, 2)

    def test_a_server_that_resets_after_the_request_is_an_api_error(self, _sleep):
        """The real path: urllib's own handlers talk to a real socket.

        The default opener would route through any proxy in the environment
        instead of the local server, so this goes through a direct one.
        """
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(5)
        self.addCleanup(listener.close)

        def reset_every_request():
            while True:
                try:
                    connection, _ = listener.accept()
                except OSError:
                    return
                connection.recv(65536)
                connection.close()

        threading.Thread(target=reset_every_request, daemon=True).start()
        url = f"http://127.0.0.1:{listener.getsockname()[1]}/pulls/42"

        with mock.patch("urllib.request.urlopen", side_effect=direct_urlopen):
            with self.assertRaises(resolve.GitHubApiError) as raised:
                resolve.github_get(url)

        self.assertIn(f"GET {url}", str(raised.exception))
        self.assertIn("RemoteDisconnected", str(raised.exception))

    def test_a_malformed_url_fails_at_once_without_retries(self, sleep):
        """``InvalidURL`` is an ``HTTPException``, but retrying cannot fix an input.

        A ``pr_number`` of ``14 105`` once slept 2 s and 4 s before failing as
        a transient error, which reads like a flaky network.
        """
        url = f"{API}/pulls/14 105"

        with mock.patch("urllib.request.urlopen", side_effect=direct_urlopen):
            with self.assertRaises(resolve.GitHubApiError) as raised:
                resolve.github_get(url)

        self.assertNotIsInstance(raised.exception, resolve.TransientGitHubError)
        self.assertIn(f"GET {url} is not a valid URL", str(raised.exception))
        sleep.assert_not_called()

    def test_an_invalid_url_from_the_transport_is_not_retried(self, _sleep):
        error, urlopen = self.get([http.client.InvalidURL("bad")])
        self.assertNotIsInstance(error, resolve.TransientGitHubError)
        self.assertEqual(urlopen.call_count, 1)

    def test_an_unreadable_body_is_an_api_error(self, _sleep):
        body = mock.MagicMock()
        body.__enter__.return_value = io.BytesIO(b"<html>")
        error, _ = self.get([body])
        self.assertIsInstance(error, resolve.GitHubApiError)
        self.assertIn("invalid JSON", str(error))

    def test_a_missing_token_fails_before_any_request(self, _sleep):
        with mock.patch.dict(os.environ, {"GITHUB_TOKEN": ""}):
            error, urlopen = self.get([response({})])
        self.assertIn("GITHUB_TOKEN is not set", str(error))
        urlopen.assert_not_called()


class HarnessProbeTest(unittest.TestCase):
    def harness(self, **sources):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        for relative, text in sources.items():
            path = root / "coco" / "cli" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return root

    def test_a_harness_declaring_the_option_supports_it(self):
        harness = self.harness(
            **{"commands/pr.py": '@click.option(\n    "--baseline-sha",'}
        )
        self.assertTrue(resolve.harness_supports_baseline_sha(harness))

    def test_single_quotes_and_a_multi_line_declaration_are_found(self):
        harness = self.harness(
            **{
                "commands/pr.py": "@click.option(\n    '--baseline-sha',\n    default='',\n)"
            }
        )
        self.assertTrue(resolve.harness_supports_baseline_sha(harness))

    def test_an_option_moved_to_a_shared_module_is_found(self):
        harness = self.harness(
            **{"commands/pr.py": "", "options.py": 'BASELINE = "--baseline-sha"'}
        )
        self.assertTrue(resolve.harness_supports_baseline_sha(harness))

    def test_a_harness_with_only_baseline_stack_does_not_support_it(self):
        """What ``coco`` 1.3.x declares: a neighbouring option must not match."""
        harness = self.harness(
            **{"commands/pr.py": '"--baseline-stack", "--skip-baselines"'}
        )
        self.assertFalse(resolve.harness_supports_baseline_sha(harness))

    def test_a_directory_without_a_cli_does_not_support_it(self):
        self.assertFalse(resolve.harness_supports_baseline_sha(self.harness()))


class ResolveBaselineShaTest(unittest.TestCase):
    def _harness(self, pr_command):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        path = root / "coco" / "cli" / "commands" / "pr.py"
        path.parent.mkdir(parents=True)
        path.write_text(pr_command, encoding="utf-8")
        return root

    def resolve(self, pr_number="42", pr_command='"--baseline-sha"', require=False):
        return resolve.resolve_baseline_sha(
            pr_number=pr_number,
            pr_sha=HEAD,
            pr_repo="ROCm/rocm-systems",
            harness_dir=self._harness(pr_command),
            require=require,
            get=fake_github(),
        )

    def test_periodic_runs_keep_the_configured_baseline(self):
        self.assertEqual(self.resolve(pr_number=""), "")

    def test_a_pr_run_pins_to_the_merge_base_when_the_harness_can(self):
        self.assertEqual(self.resolve(), MERGE_BASE)

    def test_an_older_harness_is_not_passed_a_flag_it_would_reject(self):
        """The harness ref (1.3.x) may predate --baseline-sha; passing it would fail every gate."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            sha = self.resolve(pr_command='"--skip-baselines"')
        self.assertEqual(sha, "")
        self.assertIn("::warning::", out.getvalue())

    def test_a_required_pin_fails_on_a_harness_without_the_flag(self):
        """Once the gate runs a harness that has the flag, losing it must not unpin quietly."""
        with self.assertRaisesRegex(ValueError, "requires the PR baseline"):
            self.resolve(pr_command='"--skip-baselines"', require=True)

    def test_a_required_pin_still_pins_when_the_harness_can(self):
        self.assertEqual(self.resolve(require=True), MERGE_BASE)


class MainTest(unittest.TestCase):
    """``main`` reads its inputs from the environment, as the composite action sets them."""

    def run_main(self, **env):
        outputs = {}
        completed = subprocess.CompletedProcess([], 0, stdout="cafe\n")
        out = io.StringIO()
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(resolve, "set_github_output", outputs.update),
            mock.patch.object(resolve.subprocess, "run", return_value=completed),
            contextlib.redirect_stdout(out),
        ):
            try:
                resolve.main()
                code = 0
            except SystemExit as exc:
                code = exc.code
        return code, outputs, out.getvalue()

    def harness(self, pr_command):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        path = root / "coco" / "cli" / "commands" / "pr.py"
        path.parent.mkdir(parents=True)
        path.write_text(pr_command, encoding="utf-8")
        return os.fspath(root)

    def test_a_missing_harness_dir_fails_instead_of_silently_unpinning(self):
        """The probe once read ``Path.cwd()``; a dropped working-directory unpinned every run."""
        code, outputs, out = self.run_main(PR_NUMBER="42", PR_SHA=HEAD)
        self.assertEqual(code, 1)
        self.assertEqual(outputs, {})
        self.assertIn("::error::HARNESS_DIR is not set", out)

    def test_a_periodic_run_needs_no_api_call(self):
        code, outputs, _ = self.run_main(
            HARNESS_DIR=self.harness(""),
            EVENT_NAME="workflow_dispatch",
            PERIODIC_MODE="nightly",
        )
        self.assertEqual(code, 0)
        self.assertEqual(outputs["mode"], "nightly")
        self.assertNotIn("--baseline-sha", outputs["coco_args"])

    def test_a_required_pin_on_an_old_harness_fails_the_setup(self):
        code, outputs, out = self.run_main(
            HARNESS_DIR=self.harness('"--skip-baselines"'),
            PR_NUMBER="42",
            PR_SHA=HEAD,
            PR_REPO="ROCm/rocm-systems",
            REQUIRE_BASELINE_SHA="true",
        )
        self.assertEqual(code, 1)
        self.assertEqual(outputs, {})
        self.assertIn("::error::This coco harness has no --baseline-sha", out)

    def test_an_api_failure_fails_the_setup_with_the_request_named(self):
        """End to end: a missing PR ends setup with the request in the annotation."""
        with mock.patch("urllib.request.urlopen", side_effect=[http_error(404)]):
            code, outputs, out = self.run_main(
                HARNESS_DIR=self.harness('"--baseline-sha"'),
                PR_NUMBER="42",
                PR_SHA=HEAD,
                PR_REPO="ROCm/rocm-systems",
                GITHUB_TOKEN="test-token",
            )
        self.assertEqual(code, 1)
        self.assertEqual(outputs, {})
        self.assertIn(f"::error::GET {API}/pulls/42 failed: HTTP 404", out)


if __name__ == "__main__":
    unittest.main()
