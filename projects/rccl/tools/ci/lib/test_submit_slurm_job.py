#!/usr/bin/env python3
"""Unit tests for submit_slurm_job.py helpers."""

import unittest

from submit_slurm_job import parse_parsable_job_id


class ParseParsableJobIdTest(unittest.TestCase):
    def test_bare_id(self) -> None:
        self.assertEqual(parse_parsable_job_id("19010\n"), "19010")

    def test_id_and_cluster(self) -> None:
        self.assertEqual(parse_parsable_job_id("19010;tensorwave\n"), "19010")

    def test_empty(self) -> None:
        self.assertEqual(parse_parsable_job_id(""), "")
        self.assertEqual(parse_parsable_job_id("   \n"), "")


if __name__ == "__main__":
    unittest.main()
