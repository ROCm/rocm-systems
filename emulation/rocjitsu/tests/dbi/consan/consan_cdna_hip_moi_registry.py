#!/usr/bin/env python3
"""Target-specific CDNA hip-moi simulator suite registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

EXPECTED_SUITES = 13
EXPECTED_TESTS = 33


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class Target:
    id: str
    build_dir_name: str
    executable_family: str
    default_config_name: str


@dataclass(frozen=True)
class Suite:
    id: str
    ctest_name: str
    executable_template: str
    expected_tests: int
    # Set when an offline suite is also a campaign workload. The runner uses
    # this link to keep promotion coverage and simulator qualification aligned.
    validation_workload_id: str | None = None


SUITES = (
    Suite(
        "jakub-matmul",
        "JakubMatmul",
        "hip_moi_reference_{family}_jakub_matmul",
        2,
        "jakub-attention",
    ),
    Suite(
        "mfma-attention",
        "MfmaAttention",
        "hip_moi_instrumented_{family}_mfma_attention_block_test",
        2,
        "wmma-attention",
    ),
    Suite(
        "d128-block",
        "D128Block",
        "hip_moi_instrumented_{family}_d128_attention_block_test",
        2,
        "d128-block",
    ),
    Suite(
        "d128-pressure",
        "D128Pressure",
        "hip_moi_instrumented_{family}_d128_attention_pressure_test",
        4,
        "d128-pressure",
    ),
    Suite(
        "mfma-register-handoff",
        "MfmaRegisterHandoff",
        "hip_moi_instrumented_{family}_mfma_register_handoff_test",
        4,
    ),
    Suite(
        "mfma-no-score",
        "MfmaNoScore",
        "hip_moi_instrumented_{family}_mfma_no_score_lds_attention_test",
        2,
    ),
    Suite(
        "d128-no-score",
        "D128NoScore",
        "hip_moi_instrumented_{family}_d128_no_score_lds_attention_test",
        2,
    ),
    Suite(
        "pingpong-private",
        "PingpongPrivate",
        "hip_moi_instrumented_{family}_pingpong_private_lds_test",
        2,
    ),
    Suite(
        "pingpong-cooperative",
        "PingpongCooperative",
        "hip_moi_instrumented_{family}_pingpong_cooperative_lds_test",
        3,
    ),
    Suite(
        "pingpong-wide",
        "PingpongWide",
        "hip_moi_instrumented_{family}_pingpong_wide_cooperative_lds_test",
        3,
    ),
    Suite(
        "streamk-arrival",
        "StreamKArrival",
        "hip_moi_instrumented_{family}_mfma_streamk_arrival_counter_test",
        2,
        "streamk-arrival",
    ),
    Suite(
        "tree-atomic-or",
        "TreeAtomicOr",
        "hip_moi_instrumented_{family}_mfma_streamk_tree_atomic_or_test",
        2,
        "tree-atomic-or",
    ),
    Suite(
        "mfma-lds-alias-handoff",
        "MfmaLdsAliasHandoff",
        "hip_moi_instrumented_{family}_mfma_attention_lds_alias_handoff_test",
        3,
    ),
)
SUITE_BY_ID = {suite.id: suite for suite in SUITES}
TARGETS = {
    target.id: target
    for target in (
        Target(
            "gfx942",
            "hip-moi-build-gfx942-tests",
            "cdna3",
            "gfx942_cdna3_kmd.json",
        ),
        # The gfx950 KMD config has an unbounded tick budget. Keep the
        # simulator smoke on the standalone config's 100000-tick bound.
        Target(
            "gfx950",
            "hip-moi-build-gfx950-tests",
            "cdna4",
            "gfx950_cdna4.json",
        ),
    )
}


def validate() -> None:
    if len(SUITES) != EXPECTED_SUITES:
        raise RegistryError(
            f"CDNA suite registry must contain {EXPECTED_SUITES} suites"
        )
    if len(SUITE_BY_ID) != len(SUITES):
        raise RegistryError("CDNA suite registry IDs must be unique")
    ctest_names = {suite.ctest_name for suite in SUITES}
    if len(ctest_names) != len(SUITES):
        raise RegistryError("CDNA suite registry CTest names must be unique")
    if sum(suite.expected_tests for suite in SUITES) != EXPECTED_TESTS:
        raise RegistryError(f"CDNA suite registry must contain {EXPECTED_TESTS} tests")


def relative_executable_path(target_id: str, suite_id: str) -> Path:
    try:
        target = TARGETS[target_id]
        suite = SUITE_BY_ID[suite_id]
    except KeyError as error:
        raise RegistryError(
            f"unknown CDNA hip-moi registry key: {error.args[0]}"
        ) from error
    executable = suite.executable_template.format(family=target.executable_family)
    return Path(target.build_dir_name) / "tests" / executable
