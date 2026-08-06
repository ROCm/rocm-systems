# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Diagnostic probes for the capability gates.

Every probe is a no-op: it resolves the same ``*_unavailable_reason`` helper
that ``pytest_collection_modifyitems`` applies to the real tests, then passes
when the gate is satisfied or skips carrying that reason. Running them answers
"would the <x> tests have run on this machine?" without paying for the tests
themselves, which makes it possible to tell a genuinely unavailable capability
apart from a mis-written gate when reading a CI log.

``test_summary`` is the exception: it fails on purpose so that CTest prints its
report. Expect this file to turn the component job red for as long as it is
present.

The probes deliberately carry none of the functional markers they report on.
tests/test_categories.yaml vetoes labels such as ``attach`` and ``overflow``
from every tier, so a probe marked ``attach`` would never be labelled into a
tier and would disappear from CI instead of reporting on it.

Gates parametrized by a marker argument (version minimums,
``mpi_implementation``) are out of scope here since they have no single answer.
"""

from __future__ import annotations

import os

import pytest

from conftest import (
    ainic_unavailable_reason,
    annotate_unavailable_reason,
    attach_unavailable_reason,
    gpu_unavailable_reason,
    julia_unavailable_reason,
    mpi_unavailable_reason,
    nic_unavailable_reason,
    no_docker_unavailable_reason,
    overflow_unavailable_reason,
    python_base_unavailable_reason,
    python_versions_unavailable_reason,
    shmem_unavailable_reason,
    xnack_unavailable_reason,
)

pytestmark = [pytest.mark.conditions]


def _python_reason(rocprof_config, _pytest_config):
    """conftest applies both python checks in sequence; first failure wins."""
    return python_base_unavailable_reason(
        rocprof_config
    ) or python_versions_unavailable_reason(rocprof_config)


def _ucx_reason(rocprof_config, _pytest_config):
    """conftest gates ucx inline rather than through a named helper."""
    if rocprof_config.capabilities.ucx_availability:
        return None
    return "UCX not available"


# Marker name -> callable(rocprof_config, pytest_config) -> skip reason or None.
# Mirrors the marker dispatch in conftest.pytest_collection_modifyitems.
GATES = {
    "ainic_required": lambda cfg, _: ainic_unavailable_reason(cfg),
    "annotate": lambda cfg, _: annotate_unavailable_reason(cfg),
    "attach": lambda cfg, _: attach_unavailable_reason(cfg),
    "gpu": lambda cfg, _: gpu_unavailable_reason(),
    "julia": lambda cfg, _: julia_unavailable_reason(cfg),
    "mpi": mpi_unavailable_reason,
    "nic": lambda cfg, _: nic_unavailable_reason(cfg),
    "no_docker": lambda cfg, _: no_docker_unavailable_reason(cfg),
    "overflow": lambda cfg, _: overflow_unavailable_reason(cfg),
    "python": _python_reason,
    "shmem": lambda cfg, _: shmem_unavailable_reason(cfg),
    "ucx": _ucx_reason,
    "xnack": lambda cfg, _: xnack_unavailable_reason(cfg),
}


# =============================================================================
# Condition probes
# =============================================================================


# The "rocprofiler-systems-" prefix is load-bearing: it makes the generated CTest
# names match the quick tier's "rocprofiler-systems.*" include in
# tests/test_categories.yaml, so the probes report on every tier rather than only
# on standard and above
@pytest.mark.class_name("rocprofiler-systems-conditions")
class TestConditions:
    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("gate", sorted(GATES))
    def test_gate(self, gate, rocprof_config, pytestconfig):
        reason = GATES[gate](rocprof_config, pytestconfig)
        if reason is not None:
            pytest.skip(reason)

    @pytest.mark.timeout(60)
    def test_summary(self, rocprof_config, pytestconfig):
        """Report every gate verdict and the inputs behind them.

        Fails unconditionally. TheRock runs this component's CTests without
        ``-V`` (see ``ctest_verbose`` in test_runner.py), so a passing test's
        output never reaches the CI log; only ``--output-on-failure`` gets
        through. Failing is the sole way to make the report visible.
        """
        caps = rocprof_config.capabilities

        report = ["System state:"]
        for label, value in (
            ("euid", os.geteuid()),
            ("inside docker", caps.is_inside_docker),
            ("yama ptrace_scope", caps.ptrace_scope),
            ("perf_event_paranoid", caps.perf_event_paranoid),
            ("CAP_SYS_ADMIN", caps.cap_sys_admin),
            ("CAP_PERFMON", caps.cap_perfmon),
            ("perf events usable", caps.perf_events_usable),
            ("PAPI available", caps.papi_availability),
        ):
            report.append(f"  {label:<22} {value}")

        report.append("Gates:")
        for gate in sorted(GATES):
            reason = GATES[gate](rocprof_config, pytestconfig)
            verdict = "available" if reason is None else f"skipped - {reason}"
            report.append(f"  {gate:<22} {verdict}")

        pytest.fail("\n".join(report), pytrace=False)
