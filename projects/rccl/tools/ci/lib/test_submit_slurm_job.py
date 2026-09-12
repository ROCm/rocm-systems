#!/usr/bin/env python3
"""Unit tests for submit_slurm_job.py helpers."""

import tempfile
import unittest
from unittest import mock

import submit_slurm_job
from submit_slurm_job import JobResult, parse_parsable_job_id, wait_for_job


class ParseParsableJobIdTest(unittest.TestCase):
    def test_bare_id(self) -> None:
        self.assertEqual(parse_parsable_job_id("19010\n"), "19010")

    def test_id_and_cluster(self) -> None:
        self.assertEqual(parse_parsable_job_id("19010;tensorwave\n"), "19010")

    def test_banner_before_id(self) -> None:
        # The reason .splitlines()[-1] exists: sbatch can print site banner
        # text (e.g. a reservation notice) on stdout ahead of the id.
        self.assertEqual(
            parse_parsable_job_id("sbatch: note: using reservation\n19010\n"), "19010"
        )

    def test_empty(self) -> None:
        self.assertEqual(parse_parsable_job_id(""), "")
        self.assertEqual(parse_parsable_job_id("   \n"), "")

    def test_banner_after_id_is_rejected(self) -> None:
        # The mirror image of test_banner_before_id: `--parsable` only ever
        # prints digits, so trailing non-numeric text on the last line means
        # this is not a real job id and must not be handed to sacct/scancel.
        self.assertEqual(parse_parsable_job_id("19010\nsome trailing note\n"), "")


class WaitForJobTest(unittest.TestCase):
    """wait_for_job is the load-bearing replacement for `sbatch --wait`."""

    def _run(self, states, cancelled=None, **kwargs):
        """Drive wait_for_job over a canned sacct state sequence.

        poll_interval=0 keeps this off the clock: no Slurm, no sleeping.
        """
        remaining = list(states)
        scancelled = []

        def fake_query_job(job_id, retries, interval):
            return JobResult(state=remaining.pop(0), exit_code="0:0")

        with mock.patch.object(submit_slurm_job, "query_job", fake_query_job):
            with mock.patch.object(submit_slurm_job, "scancel_job", scancelled.append):
                rc, result = wait_for_job(
                    "19010", 0.0, cancelled or (lambda: False), **kwargs
                )
        return rc, result, scancelled, remaining

    def test_waits_through_non_terminal_states(self) -> None:
        rc, result, scancelled, remaining = self._run(
            ["PENDING", "RUNNING", "COMPLETED"]
        )
        self.assertEqual(rc, 0)
        self.assertEqual(result.state, "COMPLETED")
        self.assertEqual(scancelled, [])
        self.assertEqual(remaining, [])

    def test_held_states_are_not_terminal(self) -> None:
        # These used to fall through as "finished" while the job was still
        # alive and still holding the reservation, with nothing scancelling it.
        for state in ("SUSPENDED", "RESV_DEL_HOLD", "REQUEUE_HOLD", "SPECIAL_EXIT"):
            with self.subTest(state=state):
                rc, result, _, remaining = self._run([state, "COMPLETED"])
                self.assertEqual(rc, 0)
                self.assertEqual(result.state, "COMPLETED")
                self.assertEqual(remaining, [])

    def test_non_terminal_states_is_exactly_expected(self) -> None:
        # A literal expected set, not a spot-check subset: iterating
        # NON_TERMINAL_STATES itself would pass trivially even after a
        # deletion, since that just shortens what the loop checks against.
        self.assertEqual(
            submit_slurm_job.NON_TERMINAL_STATES,
            frozenset(
                {
                    "",
                    "COMPLETING",
                    "CONFIGURING",
                    "PENDING",
                    "REQUEUED",
                    "REQUEUE_FED",
                    "REQUEUE_HOLD",
                    "RESIZING",
                    "RESV_DEL_HOLD",
                    "REVOKED",
                    "RUNNING",
                    "SIGNALING",
                    "SPECIAL_EXIT",
                    "STAGE_OUT",
                    "STOPPED",
                    "SUSPENDED",
                }
            ),
        )

    def test_terminal_failure_still_returns_zero(self) -> None:
        # wait_for_job only reports "did we cancel it"; sacct is the success
        # oracle, and evaluate() turns FAILED into a non-zero exit. The
        # FAILED result itself must still come back to the caller: a second,
        # independent sacct query right after the job ends can catch the
        # accounting DB with no row yet and misread this as success.
        rc, result, scancelled, _ = self._run(["FAILED"])
        self.assertEqual(rc, 0)
        self.assertEqual(result.state, "FAILED")
        self.assertEqual(scancelled, [])

    def test_cancel_flag_scancels(self) -> None:
        rc, result, scancelled, _ = self._run(["RUNNING"], cancelled=lambda: True)
        self.assertEqual(rc, 1)
        self.assertIsNone(result)
        self.assertEqual(scancelled, ["19010"])

    def test_missing_sacct_row_is_bounded(self) -> None:
        # "" is a non-terminal state, so an sacct that never produces a row
        # would otherwise spin until the GitHub job timeout with the
        # allocation still held.
        rc, result, scancelled, _ = self._run(["", ""], missing_row_timeout=0.0)
        self.assertEqual(rc, 1)
        self.assertIsNone(result)
        self.assertEqual(scancelled, ["19010"])

    def test_transient_empty_state_does_not_give_up(self) -> None:
        # query_job maps a CalledProcessError from sacct to state="". That must
        # not cancel a job that is otherwise reporting state.
        rc, result, scancelled, remaining = self._run(
            ["RUNNING", "", "COMPLETED"], missing_row_timeout=30.0
        )
        self.assertEqual(rc, 0)
        self.assertEqual(result.state, "COMPLETED")
        self.assertEqual(scancelled, [])
        self.assertEqual(remaining, [])


class SubmitCommandTest(unittest.TestCase):
    def test_submit_does_not_use_sbatch_wait(self) -> None:
        """`sbatch --wait` is what leaked nodes on cancel; pin its absence."""
        seen = {}

        def fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            return mock.Mock(returncode=0, stdout="19010\n", stderr="")

        with mock.patch.object(submit_slurm_job.subprocess, "run", fake_run):
            with mock.patch.object(
                submit_slurm_job, "wait_for_job", lambda *a, **k: (0, None)
            ):
                rc, job_id, result = submit_slurm_job.submit_and_wait(
                    submit_slurm_job.Path("job.sbatch"), "ALL", None, None
                )

        self.assertEqual((rc, job_id, result), (0, "19010", None))
        self.assertNotIn("--wait", seen["cmd"])
        self.assertIn("--parsable", seen["cmd"])

    def test_submit_writes_job_id_file_when_chdir_given(self) -> None:
        """Argus flagged that no test ever exercised the slurm-job-id write.

        That file is what the `if: cancelled()` backup step reads to scancel a
        SIGKILL'd process, so a real chdir must actually end up on disk with
        the job id in it. This also pins that sbatch itself runs from that
        directory (`cwd=`), since a relative `#SBATCH --output=%x-%j.out`
        depends on it.
        """
        seen_kwargs = {}

        def fake_run(cmd, **kwargs):
            seen_kwargs.update(kwargs)
            return mock.Mock(returncode=0, stdout="19010\n", stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            chdir = submit_slurm_job.Path(tmpdir)
            with mock.patch.object(submit_slurm_job.subprocess, "run", fake_run):
                with mock.patch.object(
                    submit_slurm_job, "wait_for_job", lambda *a, **k: (0, None)
                ):
                    rc, job_id, result = submit_slurm_job.submit_and_wait(
                        submit_slurm_job.Path("job.sbatch"), "ALL", chdir, None
                    )

            self.assertEqual((rc, job_id, result), (0, "19010", None))
            self.assertEqual((chdir / "slurm-job-id").read_text(), "19010\n")
        self.assertEqual(seen_kwargs["cwd"], str(chdir))

    def test_empty_job_id_is_hard_failure(self) -> None:
        """sbatch rc=0 with no parsable id must not be read as success.

        Nothing was submitted to wait on or scancel, so trusting rc=0 here
        would report a queued-but-unknown job as a pass. wait_for_job is
        mocked and asserted never called: without this, a regression in the
        early-return guard would fall through into a real wait loop instead
        of failing this test, since there is no job id for it to fail on.
        """
        wait_for_job_calls = []

        def fake_run(cmd, **kwargs):
            return mock.Mock(returncode=0, stdout="\n", stderr="")

        def fake_wait_for_job(*a, **k):
            wait_for_job_calls.append((a, k))
            return (0, JobResult(state="COMPLETED", exit_code="0:0"))

        with mock.patch.object(submit_slurm_job.subprocess, "run", fake_run):
            with mock.patch.object(submit_slurm_job, "wait_for_job", fake_wait_for_job):
                rc, job_id, result = submit_slurm_job.submit_and_wait(
                    submit_slurm_job.Path("job.sbatch"), "ALL", None, None
                )

        self.assertEqual((rc, job_id, result), (1, "", None))
        self.assertEqual(wait_for_job_calls, [])

    def test_unexpected_exception_scancels_before_reraising(self) -> None:
        """An unexpected crash after a job id exists must not leak the node.

        The `if: cancelled()` backup step only fires on an actual GitHub
        cancellation, not on an ordinary exception, so submit_and_wait itself
        is the last chance to release the allocation.
        """
        scancelled = []

        def fake_run(cmd, **kwargs):
            return mock.Mock(returncode=0, stdout="19010\n", stderr="")

        def fake_wait_for_job(*a, **k):
            raise RuntimeError("sacct vanished from PATH")

        with mock.patch.object(submit_slurm_job.subprocess, "run", fake_run):
            with mock.patch.object(submit_slurm_job, "wait_for_job", fake_wait_for_job):
                with mock.patch.object(
                    submit_slurm_job, "scancel_job", scancelled.append
                ):
                    with self.assertRaises(RuntimeError):
                        submit_slurm_job.submit_and_wait(
                            submit_slurm_job.Path("job.sbatch"), "ALL", None, None
                        )

        self.assertEqual(scancelled, ["19010"])

    def test_cancel_during_submission_does_not_report_success(self) -> None:
        """A signal arriving while sbatch itself is still running must not win.

        cancel_requested can flip true before job_id is known; submit_and_wait
        must still refuse to return success once the id shows up afterward.
        This isolates submit_and_wait's own `if cancel_requested:` guard by
        mocking wait_for_job: without the guard, execution would fall through
        into wait_for_job, which the mock forces to report a fake success, so
        deleting the guard would flip both the (rc, result) assertion below
        and the "wait_for_job must never be called" assertion.
        """
        scancelled = []
        handlers = {}
        wait_for_job_calls = []

        def fake_signal(sig, handler):
            handlers[sig] = handler
            return None

        def fake_run(cmd, **kwargs):
            # Simulate a SIGTERM landing while sbatch is in flight, before
            # this process knows the job id yet.
            handlers[submit_slurm_job.signal.SIGTERM](
                submit_slurm_job.signal.SIGTERM, None
            )
            return mock.Mock(returncode=0, stdout="19010\n", stderr="")

        def fake_wait_for_job(*a, **k):
            wait_for_job_calls.append((a, k))
            return (0, JobResult(state="COMPLETED", exit_code="0:0"))

        with mock.patch.object(submit_slurm_job.signal, "signal", fake_signal):
            with mock.patch.object(submit_slurm_job.subprocess, "run", fake_run):
                with mock.patch.object(
                    submit_slurm_job, "scancel_job", scancelled.append
                ):
                    with mock.patch.object(
                        submit_slurm_job, "wait_for_job", fake_wait_for_job
                    ):
                        rc, job_id, result = submit_slurm_job.submit_and_wait(
                            submit_slurm_job.Path("job.sbatch"), "ALL", None, None
                        )

        self.assertEqual((rc, job_id, result), (1, "19010", None))
        # Once from the signal handler (job_id still unknown), once more from
        # the post-sbatch cancel_requested check now that it is known.
        self.assertEqual(scancelled, ["", "19010"])
        self.assertEqual(wait_for_job_calls, [])

    def test_submit_returns_wait_for_jobs_terminal_result(self) -> None:
        """submit_and_wait must hand back the JobResult wait_for_job saw.

        Argus flagged that main() used to discard this and re-query sacct
        itself, which can race the accounting DB right after a job ends and
        read a real FAILED job as "no data" (i.e. success). Pinning that the
        result flows through here is what makes that re-query unnecessary.
        """
        failed = JobResult(state="FAILED", exit_code="1:0")

        def fake_run(cmd, **kwargs):
            return mock.Mock(returncode=0, stdout="19010\n", stderr="")

        with mock.patch.object(submit_slurm_job.subprocess, "run", fake_run):
            with mock.patch.object(
                submit_slurm_job, "wait_for_job", lambda *a, **k: (0, failed)
            ):
                rc, job_id, result = submit_slurm_job.submit_and_wait(
                    submit_slurm_job.Path("job.sbatch"), "ALL", None, None
                )

        self.assertEqual((rc, job_id), (0, "19010"))
        self.assertIs(result, failed)


class EvaluateTest(unittest.TestCase):
    """Direct calls to evaluate(): previously only reached indirectly via
    MainRegressionTest, which only ever fed it a COMPLETED|0:0 case."""

    def test_non_completed_state_is_failure(self) -> None:
        self.assertEqual(
            submit_slurm_job.evaluate(0, "19010", JobResult("FAILED", "1:0")), 1
        )

    def test_nonzero_exit_code_is_failure(self) -> None:
        self.assertEqual(
            submit_slurm_job.evaluate(0, "19010", JobResult("COMPLETED", "2:0")), 1
        )

    def test_completed_zero_exit_is_success(self) -> None:
        self.assertEqual(
            submit_slurm_job.evaluate(0, "19010", JobResult("COMPLETED", "0:0")), 0
        )


class QueryJobTest(unittest.TestCase):
    def test_called_process_error_is_treated_as_no_data(self) -> None:
        """Argus flagged this path was only ever exercised via literal ""
        inputs, never a real mocked exception from subprocess itself."""

        def fake_check_output(cmd, **kwargs):
            raise submit_slurm_job.subprocess.CalledProcessError(1, cmd)

        with mock.patch.object(
            submit_slurm_job.subprocess, "check_output", fake_check_output
        ):
            result = submit_slurm_job.query_job("19010", retries=1, interval=0)

        self.assertEqual(result, JobResult(state="", exit_code=""))


class ScancelJobTest(unittest.TestCase):
    """scancel_job's own body was never executed: every caller-level test
    mocks the whole function away."""

    def test_cancels_before_logging(self) -> None:
        seen = []

        def fake_run(cmd, **kwargs):
            seen.append(cmd)
            return mock.Mock(returncode=0)

        def raising_log(*a, **k):
            raise RuntimeError("reentrant call")

        with mock.patch.object(submit_slurm_job.subprocess, "run", fake_run):
            with mock.patch.object(submit_slurm_job, "log", raising_log):
                with self.assertRaises(RuntimeError):
                    submit_slurm_job.scancel_job("19010")

        self.assertEqual(seen, [["scancel", "19010"]])

    def test_empty_job_id_is_a_no_op(self) -> None:
        with mock.patch.object(submit_slurm_job.subprocess, "run") as mock_run:
            submit_slurm_job.scancel_job("")

        mock_run.assert_not_called()


class MainRegressionTest(unittest.TestCase):
    """Regression test through main() itself, per Argus's ask.

    The unit tests above pin submit_and_wait/wait_for_job in isolation, but
    Argus's own probing found that removing id persistence, the timer reset,
    or reintroducing main()'s unconditional re-query all still passed those.
    This drives the real main() with a fake sbatch/sacct and pins all three at
    once.
    """

    def test_main_end_to_end_uses_wait_result_without_requery(self) -> None:
        check_output_calls = []

        def fake_run(cmd, **kwargs):
            return mock.Mock(returncode=0, stdout="19010\n", stderr="")

        def fake_check_output(cmd, **kwargs):
            check_output_calls.append(cmd)
            return "COMPLETED|0:0\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = submit_slurm_job.Path(tmpdir)
            script = tmp / "job.sbatch"
            script.write_text("#!/bin/bash\necho hi\n")
            chdir = tmp / "run"

            with mock.patch.object(submit_slurm_job.subprocess, "run", fake_run):
                with mock.patch.object(
                    submit_slurm_job.subprocess, "check_output", fake_check_output
                ):
                    rc = submit_slurm_job.main(
                        [
                            "--script",
                            str(script),
                            "--chdir",
                            str(chdir),
                            "--poll-interval",
                            "0",
                            "--wait-poll-interval",
                            "0",
                        ]
                    )

            self.assertEqual(rc, 0)
            # id persistence: the backup-cancel step reads this file.
            self.assertEqual((chdir / "slurm-job-id").read_text(), "19010\n")
        # wait_for_job already saw the terminal COMPLETED row; main() must
        # trust it rather than re-querying sacct a second time.
        self.assertEqual(len(check_output_calls), 1)


if __name__ == "__main__":
    unittest.main()
