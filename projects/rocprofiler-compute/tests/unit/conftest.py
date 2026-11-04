# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Fixtures used only by unit tests.

Nothing here touches a GPU, the rocprof-compute CLI, or the network.
"""

import pytest

from utils.analysis_orm import Database
from utils.utils_analysis import CallTreeNode, KernelStats


@pytest.fixture
def db_session():
    """An initialized in-memory analysis database, torn down after the test."""
    Database.init(":memory:")
    yield Database.get_session()
    Database._session.close()
    Database._engine.dispose()
    Database._session = None
    Database._engine = None


@pytest.fixture
def user_region_with_nested_addmm_and_outside_relu():
    """User region1 wraps addmm; relu is a sibling root outside the region."""
    addmm = CallTreeNode(name="aten::addmm", backend="torch")
    addmm.kernels["addmm_kernel"] = KernelStats(
        launches=1,
        total_duration_ns=50.0,
        pmc_group_keys=[(1, "addmm_kernel", 0, 0, 0)],
    )
    region1 = CallTreeNode(name="region1", backend="user")
    region1.children = [addmm]
    relu = CallTreeNode(name="aten::relu", backend="torch")
    relu.kernels["relu_kernel"] = KernelStats(
        launches=1,
        total_duration_ns=10.0,
        pmc_group_keys=[(2, "relu_kernel", 0, 0, 0)],
    )
    return {"1": [region1, relu]}
