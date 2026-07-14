# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import common


def test_workload_fixture_path_uses_alias_for_gfx115x_point_parts() -> None:
    assert (
        common.workload_fixture_path("dispatch_0", "RDNA35_POINT_2")
        == "tests/workloads/dispatch_0/RDNA35_HALO"
    )
    assert (
        common.workload_fixture_path("kernel", "RDNA35_POINT_1")
        == "tests/workloads/kernel/RDNA35_HALO"
    )
    assert (
        common.workload_fixture_path("vcopy", "RDNA35_KRACKAN2")
        == "tests/workloads/vcopy/RDNA35_HALO"
    )


def test_workload_fixture_path_passthrough_for_unaliased_arch() -> None:
    assert (
        common.workload_fixture_path("vcopy", "RDNA35_HALO")
        == "tests/workloads/vcopy/RDNA35_HALO"
    )
