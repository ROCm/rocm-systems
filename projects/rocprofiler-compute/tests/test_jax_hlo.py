# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from pathlib import Path

import common  # noqa: F401  (adds src to sys.path)
import pandas as pd

from utils.jax_hlo import build_jax_kernel_source_map, normalize_kernel_name
from utils.utils_analysis import build_call_trees, write_kernel_source_map_csv

# Optimized-HLO text with the metadata and stack-frame tables the parser reads.
_HLO_TRAIN_STEP = "\n".join([
    "HloModule jit_train_step, is_scheduled=true",
    "",
    "FileNames",
    '1 "/work/model.py"',
    "",
    "FunctionNames",
    '1 "train_step"',
    "",
    "FileLocations",
    "1 {file_name_id=1 function_name_id=1 line=7 end_line=7 column=4}",
    "2 {file_name_id=1 function_name_id=1 line=13 end_line=13 column=4}",
    "",
    "StackFrames",
    "1 {file_location_id=1 parent_frame_id=1}",
    "2 {file_location_id=2 parent_frame_id=2}",
    "",
    "",
    "ENTRY %main.3 (x.1: f32[8,8]) -> f32[] {",
    '  %x.1 = f32[8,8]{1,0} parameter(0), metadata={op_name="x"}',
    (
        "  %gemm_fusion_dot_general.0 = f32[8,8]{1,0} fusion(%x.1), "
        "kind=kCustom, calls=%c, "
        'metadata={op_name="jit(train_step)/dot_general" stack_frame_id=1}'
    ),
    (
        "  %input_reduce_fusion = f32[8]{0} "
        "fusion(%gemm_fusion_dot_general.0), kind=kInput, calls=%r, "
        'metadata={op_name="jit(train_step)/reduce_sum" stack_frame_id=2}'
    ),
    (
        "  ROOT %input_reduce_fusion.1 = f32[] "
        "fusion(%input_reduce_fusion), kind=kInput, calls=%r1, "
        'metadata={op_name="jit(train_step)/reduce_sum" stack_frame_id=2}'
    ),
    "}",
    "",
])


def _write_dump(tmp_path: Path) -> Path:
    dump = tmp_path / "hlo_dump"
    dump.mkdir()
    (dump / "module_0001.jit_train_step.gfx942_gpu_after_optimizations.txt").write_text(
        _HLO_TRAIN_STEP
    )
    return dump


def test_normalize_kernel_name_replaces_dots():
    assert normalize_kernel_name("%input_reduce_fusion.1") == "input_reduce_fusion_1"


def test_map_resolves_exact_and_stem(tmp_path):
    source_map = build_jax_kernel_source_map(_write_dump(tmp_path))

    # Exact match after '.' -> '_' normalization.
    reduce = source_map.resolve("jax.jit.train_step", "input_reduce_fusion_1")
    assert reduce.operator == "jit(train_step)/reduce_sum"
    assert reduce.operator_path == "reduce_sum"
    assert reduce.source == "/work/model.py:13"
    assert reduce.shape == "f32[]"

    # Stem fallback: instruction '.0' launches as kernel '_1'.
    gemm = source_map.resolve("jax.jit.train_step", "gemm_fusion_dot_general_1")
    assert gemm.operator == "jit(train_step)/dot_general"
    assert gemm.operator_path == "dot_general"
    assert gemm.source == "/work/model.py:7"


def test_map_ignores_non_jax_and_unknown(tmp_path):
    source_map = build_jax_kernel_source_map(_write_dump(tmp_path))
    assert source_map.resolve("aten::mm", "input_reduce_fusion") is None
    assert source_map.resolve("jax.jit.other_fn", "input_reduce_fusion") is None
    assert source_map.resolve("jax.jit.train_step", "Cijk_library_gemm") is None


def test_missing_dump_is_empty(tmp_path):
    source_map = build_jax_kernel_source_map(tmp_path / "absent")
    assert source_map.is_empty()


def test_call_tree_annotated_and_csv(tmp_path):
    source_map = build_jax_kernel_source_map(_write_dump(tmp_path))
    df = pd.DataFrame([
        {
            "Operator_Name": "jax.jit.train_step",
            "Context_Id": "1@model.py:13",
            "Kernel_Name": "input_reduce_fusion_1",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 1000,
        }
    ])
    trees = build_call_trees(df, kernel_source_map=source_map)

    # The kernel is placed under an operator node, not the module node.
    train_step = trees["model.py:13"].children["jax.jit.train_step"]
    reduce_node = train_step.children["reduce_sum"]
    assert reduce_node.source == "/work/model.py:13"
    assert reduce_node.args == "f32[]"
    kstats = reduce_node.kernels["input_reduce_fusion_1"]
    assert kstats.operator == "jit(train_step)/reduce_sum"
    assert kstats.source == "/work/model.py:13"

    out = tmp_path / "ml_api_trace"
    out.mkdir()
    write_kernel_source_map_csv(trees, out)
    written = pd.read_csv(out / "kernel_source_map.csv")
    assert written.loc[0, "Operator"] == "jit(train_step)/reduce_sum"
    assert written.loc[0, "Source"] == "/work/model.py:13"
