# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Fixtures used only by unit tests.

Nothing here touches a GPU, the rocprof-compute CLI, or the network.
"""

import pytest

from ml_api_tracing.analyze.call_tree import rollup_node_stats
from ml_api_tracing.analyze.definitions import CallTreeNode, KernelStats
from utils.analysis_orm import Database

ADDMM_PMC_GROUP_KEY = (1, "addmm_kernel", 0, 0, 0)
RELU_PMC_GROUP_KEY = (2, "relu_kernel", 0, 0, 0)


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


@pytest.fixture
def mixed_user_region_forest():
    """Mixed user subtree with repeated statistical keys and outside Torch work."""
    torch_child = CallTreeNode(
        name="aten::addmm",
        backend="torch",
        kernels={
            "addmm_kernel": KernelStats(
                launches=2,
                total_duration_ns=50.0,
                min_duration_ns=20.0,
                max_duration_ns=30.0,
                pmc_group_keys=[("torch",), ("torch",)],
            )
        },
        invocation_ids={"first", "second"},
        args_invocations={"(x=float32[2x2])": {"first", "second"}},
        file_name="model.py",
        line_number=11,
    )
    triton_child = CallTreeNode(
        name="triton.matmul",
        backend="triton",
        kernels={
            "triton_kernel": KernelStats(
                launches=1, total_duration_ns=30.0, pmc_group_keys=[("triton",)]
            )
        },
    )
    nested_user = CallTreeNode(
        name="nested-region",
        backend="user",
        children=[triton_child],
        kernels={
            "nested_kernel": KernelStats(
                launches=1, total_duration_ns=20.0, pmc_group_keys=[("nested",)]
            )
        },
    )
    region = CallTreeNode(
        name="region1",
        backend="user",
        children=[torch_child, nested_user],
        kernels={
            "user_kernel": KernelStats(
                launches=1, total_duration_ns=10.0, pmc_group_keys=[("user",)]
            )
        },
    )
    outside = CallTreeNode(
        name="aten::relu",
        backend="torch",
        kernels={
            "outside_kernel": KernelStats(
                launches=1, total_duration_ns=40.0, pmc_group_keys=[("outside",)]
            )
        },
    )
    root = CallTreeNode(
        name="outer",
        backend="torch",
        children=[region, outside],
        kernels={
            "scaffolding_kernel": KernelStats(
                launches=1, total_duration_ns=90.0, pmc_group_keys=[("scaffolding",)]
            )
        },
    )
    rollup_node_stats(root)
    return {"1": [root]}


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
