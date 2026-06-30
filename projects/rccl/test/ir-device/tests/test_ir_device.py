# *************************************************************************
#  * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************
"""Functional tests for librccl_device.bc, driven through the GoogleTest
binary built from bindings/ir/test/IR_test.cpp.

Each pytest case runs the binary restricted to a single --gtest_filter so
failures map cleanly to the underlying device API. The GoogleTest binary
returns non-zero on any failure; we assert on that and surface the log path.
The `run_gtest` fixture (see conftest.py) builds the binary once per session
and skips the whole suite when prerequisites are missing.
"""

import pytest


@pytest.mark.ir_device
@pytest.mark.peer_pointer
def test_get_peer_pointer_team(run_gtest):
    """[A] ncclGetPeerPointerTeam — 20 pointer-arithmetic cases."""
    proc, log = run_gtest("IRDeviceTest.A_GetPeerPointerTeam",
                          "ir_peer_pointer.log")
    assert proc.returncode == 0, f"ncclGetPeerPointerTeam failed, see {log}"


@pytest.mark.ir_device
@pytest.mark.coop
def test_coop_init_and_accessors(run_gtest):
    """[B1-B5] ncclCoopAnyInit* + ncclCoopSize/ThreadRank/NumThreads."""
    proc, log = run_gtest(
        "IRDeviceTest.B1_*:IRDeviceTest.B2_*:IRDeviceTest.B3*:"
        "IRDeviceTest.B4*:IRDeviceTest.B5_*",
        "ir_coop_init.log",
    )
    assert proc.returncode == 0, f"ncclCoopAny init/accessors failed, see {log}"


@pytest.mark.ir_device
@pytest.mark.coop
def test_coop_sync(run_gtest):
    """[B6] ncclCoopSync — all five coop types complete."""
    proc, log = run_gtest("IRDeviceTest.B6_CoopSync", "ir_coop_sync.log")
    assert proc.returncode == 0, f"ncclCoopSync failed, see {log}"


@pytest.mark.ir_device
@pytest.mark.lsa_barrier
def test_lsa_barrier_session(run_gtest):
    """[B7] ncclLsaBarrierSession structural check (+ runtime cases skipped).

    GoogleTest returns 0 when the structural case passes and the runtime
    cases are SKIPPED, so a clean exit means the suite behaved as expected.
    """
    proc, log = run_gtest("IRDeviceTest.B7*", "ir_lsa_barrier.log")
    assert proc.returncode == 0, f"ncclLsaBarrierSession failed, see {log}"
