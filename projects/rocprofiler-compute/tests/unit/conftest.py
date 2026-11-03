# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Fixtures used only by unit tests.

Nothing here touches a GPU, the rocprof-compute CLI, or the network.
"""

import pytest

from ml_api_tracing.models import CallTreeNode, KernelStats
from utils.analysis_orm import Database

ADDMM_PMC_GROUP_KEY = (1, "addmm_kernel", 0, 0, 0)
RELU_PMC_GROUP_KEY = (2, "relu_kernel", 0, 0, 0)


def build_simple_model_forest_with_relu_and_addmm():
    """SimpleModel.forward with Linear/addmm and a relu sibling."""
    addmm = CallTreeNode(name="aten::addmm", backend="torch")
    addmm.kernels["addmm_kernel"] = KernelStats(
        launches=1,
        total_duration_ns=50.0,
        pmc_group_keys=[ADDMM_PMC_GROUP_KEY],
    )
    relu = CallTreeNode(name="aten::relu", backend="torch")
    relu.kernels["relu_kernel"] = KernelStats(
        launches=1,
        total_duration_ns=10.0,
        pmc_group_keys=[RELU_PMC_GROUP_KEY],
    )
    linear = CallTreeNode(name="nn.Module.Linear.forward", backend="torch")
    linear.children = [addmm]
    simple = CallTreeNode(name="nn.Module.SimpleModel.forward", backend="torch")
    simple.children = [linear, relu]
    return {"1": [simple]}


def build_torch_parent_triton_child_forest():
    child = CallTreeNode(name="triton.JITFunction.matmul_kernel", backend="triton")
    child.kernels["triton_matmul_kernel"] = KernelStats(
        launches=1, total_duration_ns=40.0
    )
    parent = CallTreeNode(name="nn.Module.Linear.forward", backend="torch")
    parent.kernels["torch_gemm_kernel"] = KernelStats(
        launches=1,
        total_duration_ns=10.0,
        pmc_group_keys=[(3, "torch_gemm_kernel", 0, 0, 0)],
    )
    parent.children = [child]
    return {"1": [parent]}


@pytest.fixture(scope="function")
def simple_model_forest_with_relu_and_addmm():
    """Return a callable that creates a fresh simple-model forest."""
    return build_simple_model_forest_with_relu_and_addmm


@pytest.fixture(scope="function")
def torch_parent_triton_child_forest():
    """Return a callable that creates a fresh mixed-backend forest."""
    return build_torch_parent_triton_child_forest


@pytest.fixture
def db_session():
    """An initialized in-memory analysis database, torn down after the test."""
    Database.init(":memory:")
    yield Database.get_session()
    Database._session.close()
    Database._engine.dispose()
    Database._session = None
    Database._engine = None
