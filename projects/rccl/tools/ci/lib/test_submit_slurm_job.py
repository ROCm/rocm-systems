#!/usr/bin/env python3
"""Unit tests for submit_slurm_job.py helpers."""

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
                rc = wait_for_job("19010", 0.0, cancelled or (lambda: False), **kwargs)
        return rc, scancelled, remaining

    def test_waits_through_non_terminal_states(self) -> None:
        rc, scancelled, remaining = self._run(["PENDING", "RUNNING", "COMPLETED"])
        self.assertEqual(rc, 0)
        self.assertEqual(scancelled, [])
        self.assertEqual(remaining, [])

    def test_held_states_are_not_terminal(self) -> None:
        # These used to fall through as "finished" while the job was still
        # alive and still holding the reservation, with nothing scancelling it.
        for state in ("SUSPENDED", "RESV_DEL_HOLD", "REQUEUE_HOLD", "SPECIAL_EXIT"):
            with self.subTest(state=state):
                rc, _, remaining = self._run([state, "COMPLETED"])
                self.assertEqual(rc, 0)
                self.assertEqual(remaining, [])

    def test_terminal_failure_still_returns_zero(self) -> None:
        # wait_for_job only reports "did we cancel it"; sacct is the success
        # oracle, and evaluate() turns FAILED into a non-zero exit.
        rc, scancelled, _ = self._run(["FAILED"])
        self.assertEqual(rc, 0)
        self.assertEqual(scancelled, [])

    def test_cancel_flag_scancels(self) -> None:
        rc, scancelled, _ = self._run(["RUNNING"], cancelled=lambda: True)
        self.assertEqual(rc, 1)
        self.assertEqual(scancelled, ["19010"])

    def test_missing_sacct_row_is_bounded(self) -> None:
        # "" is a non-terminal state, so an sacct that never produces a row
        # would otherwise spin until the GitHub job timeout with the
        # allocation still held.
        rc, scancelled, _ = self._run(["", ""], missing_row_timeout=0.0)
        self.assertEqual(rc, 1)
        self.assertEqual(scancelled, ["19010"])

    def test_transient_empty_state_does_not_give_up(self) -> None:
        # query_job maps a CalledProcessError from sacct to state="". That must
        # not cancel a job that is otherwise reporting state.
        rc, scancelled, remaining = self._run(
            ["RUNNING", "", "COMPLETED"], missing_row_timeout=30.0
        )
        self.assertEqual(rc, 0)
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
            with mock.patch.object(submit_slurm_job, "wait_for_job", lambda *a, **k: 0):
                rc, job_id = submit_slurm_job.submit_and_wait(
                    submit_slurm_job.Path("job.sbatch"), "ALL", None, None
                )

        self.assertEqual((rc, job_id), (0, "19010"))
        self.assertNotIn("--wait", seen["cmd"])
        self.assertIn("--parsable", seen["cmd"])


if __name__ == "__main__":
    unittest.main()
