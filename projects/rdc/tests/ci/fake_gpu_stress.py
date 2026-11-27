"""The `stress` marker, for tests/ci/conftest.py and the fake-GPU suite's installed conftest.py.

A stress test (a long loop) runs only with RDC_TEST_STRESS=1. With RDC_CI_REQUIRE_STRESS=1, a
session that ran no stress test fails, so a CI job cannot lose its stress tests without noticing.
"""

import os
import sys

import pytest

_ran = set()  # stress tests whose body ran


def pytest_configure(config):
    config.addinivalue_line("markers", "stress: long loop, runs only with RDC_TEST_STRESS=1")


def pytest_runtest_setup(item):
    if item.get_closest_marker("stress") and os.environ.get("RDC_TEST_STRESS") != "1":
        pytest.skip("stress test: set RDC_TEST_STRESS=1 to run it")


def pytest_runtest_logreport(report):
    # The body ran if the call passed, failed or xfailed; a test skipped in setup or by itself did
    # not run. (A failed setup or teardown fails the session anyway.)
    if (
        "stress" in report.keywords
        and report.when == "call"
        and (not report.skipped or hasattr(report, "wasxfail"))
    ):
        _ran.add(report.nodeid)


def pytest_sessionfinish(session, exitstatus):
    if os.environ.get("RDC_CI_REQUIRE_STRESS") != "1" or session.exitstatus != pytest.ExitCode.OK:
        return
    if not _ran:
        print("RDC_CI_REQUIRE_STRESS=1, but no stress test ran", file=sys.stderr)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
