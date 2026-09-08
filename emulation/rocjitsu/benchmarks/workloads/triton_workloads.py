#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Fixed, deterministic Triton workloads for end-to-end rocjitsu runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable

import torch
import triton
import triton.language as tl

SCHEMA = "rocjitsu.benchmark.workload.v1"
CASES: dict[str, dict[str, Any]] = {
    "triton.copy_fp32_32m": {
        "elements": 8_388_608,
        "bytes": 33_554_432,
        "dtype": "fp32",
    },
    "triton.vector_add_fp32_boundary": {
        "elements": 8_388_611,
        "dtype": "fp32",
    },
    "triton.transpose_fp16_2048": {
        "rows": 2048,
        "columns": 2048,
        "dtype": "fp16",
    },
    "triton.gather_fp32_irregular": {
        "source_elements": 8_388_608,
        "output_elements": 4_194_307,
        "index_stride": 104_729,
        "index_offset": 17,
        "dtype": "fp32",
        "index_dtype": "int64",
    },
    "triton.atomic_add_fp32_contended": {
        "elements": 131_072,
        "buckets": 256,
        "dtype": "fp32",
    },
    "triton.softmax_fp16_aligned": {
        "rows": 64,
        "columns": 4096,
        "dtype": "fp16",
    },
    "triton.softmax_fp16_boundary": {
        "rows": 64,
        "columns": 4097,
        "dtype": "fp16",
    },
    "triton.rmsnorm_bf16": {
        "rows": 128,
        "columns": 4096,
        "dtype": "bf16",
        "epsilon": 1.0e-5,
    },
    "triton.gemm_bf16_aligned": {
        "m": 256,
        "n": 256,
        "k": 512,
        "trans_a": "N",
        "trans_b": "N",
        "input_dtype": "bf16",
        "output_dtype": "bf16",
        "accumulator_dtype": "fp32",
    },
    "triton.gemm_bf16_ragged": {
        "m": 250,
        "n": 250,
        "k": 510,
        "trans_a": "N",
        "trans_b": "N",
        "input_dtype": "bf16",
        "output_dtype": "bf16",
        "accumulator_dtype": "fp32",
    },
    "triton.attention_fp16": {
        "batch": 1,
        "heads": 8,
        "sequence": 128,
        "head_dimension": 64,
        "causal": False,
        "dtype": "fp16",
    },
    "triton.gpt_oss_rmsnorm_bf16": {
        "rows": 128,
        "columns": 2880,
        "dtype": "bf16",
        "epsilon": 1.0e-5,
    },
    "triton.gpt_oss_gqa_bf16": {
        "batch": 1,
        "query_heads": 64,
        "key_value_heads": 8,
        "sequence": 128,
        "window": 128,
        "head_dimension": 64,
        "causal": True,
        "dtype": "bf16",
    },
}


@triton.jit
def _copy_kernel(
    input_pointer, output_pointer, elements: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elements
    values = tl.load(input_pointer + offsets, mask=mask)
    tl.store(output_pointer + offsets, values, mask=mask)


@triton.jit
def _vector_add_kernel(
    left_pointer,
    right_pointer,
    output_pointer,
    elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elements
    left = tl.load(left_pointer + offsets, mask=mask)
    right = tl.load(right_pointer + offsets, mask=mask)
    tl.store(output_pointer + offsets, left + right, mask=mask)


@triton.jit
def _transpose_kernel(
    input_pointer,
    output_pointer,
    rows: tl.constexpr,
    columns: tl.constexpr,
    TILE: tl.constexpr,
):
    tile_row = tl.program_id(0)
    tile_column = tl.program_id(1)
    row_offsets = tile_row * TILE + tl.arange(0, TILE)
    column_offsets = tile_column * TILE + tl.arange(0, TILE)
    mask = (row_offsets[:, None] < rows) & (column_offsets[None, :] < columns)
    values = tl.load(
        input_pointer + row_offsets[:, None] * columns + column_offsets[None, :],
        mask=mask,
    )
    tl.store(
        output_pointer + column_offsets[None, :] * rows + row_offsets[:, None],
        values,
        mask=mask,
    )


@triton.jit
def _gather_kernel(
    input_pointer,
    index_pointer,
    output_pointer,
    elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elements
    indices = tl.load(index_pointer + offsets, mask=mask, other=0)
    values = tl.load(input_pointer + indices, mask=mask)
    tl.store(output_pointer + offsets, values, mask=mask)


@triton.jit
def _atomic_add_kernel(
    input_pointer,
    output_pointer,
    elements: tl.constexpr,
    buckets: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elements
    values = tl.load(input_pointer + offsets, mask=mask, other=0.0)
    tl.atomic_add(output_pointer + offsets % buckets, values, mask=mask)


@triton.jit
def _softmax_kernel(
    input_pointer, output_pointer, row_width: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < row_width
    values = tl.load(
        input_pointer + row * row_width + columns, mask=mask, other=-float("inf")
    ).to(tl.float32)
    values -= tl.max(values, axis=0)
    numerator = tl.exp(values)
    result = numerator / tl.sum(numerator, axis=0)
    tl.store(output_pointer + row * row_width + columns, result, mask=mask)


@triton.jit
def _rmsnorm_kernel(
    input_pointer,
    output_pointer,
    row_width: tl.constexpr,
    epsilon: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < row_width
    values = tl.load(
        input_pointer + row * row_width + columns, mask=mask, other=0.0
    ).to(tl.float32)
    mean_square = tl.sum(values * values, axis=0) / row_width
    normalized = values * tl.rsqrt(mean_square + epsilon)
    tl.store(output_pointer + row * row_width + columns, normalized, mask=mask)


@triton.jit
def _weighted_rmsnorm_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    row_width: tl.constexpr,
    epsilon: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < row_width
    values = tl.load(
        input_pointer + row * row_width + columns, mask=mask, other=0.0
    ).to(tl.float32)
    weights = tl.load(weight_pointer + columns, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(values * values, axis=0) / row_width
    normalized = values * tl.rsqrt(mean_square + epsilon) * weights
    tl.store(output_pointer + row * row_width + columns, normalized, mask=mask)


@triton.jit
def _gemm_kernel(
    left_pointer,
    right_pointer,
    output_pointer,
    rows: tl.constexpr,
    columns: tl.constexpr,
    reduction: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    program = tl.program_id(0)
    programs_n = tl.cdiv(columns, BLOCK_N)
    program_m = program // programs_n
    program_n = program % programs_n

    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for reduction_start in range(0, reduction, BLOCK_K):
        reduction_offsets = reduction_start + offsets_k
        left = tl.load(
            left_pointer + offsets_m[:, None] * reduction + reduction_offsets[None, :],
            mask=(offsets_m[:, None] < rows) & (reduction_offsets[None, :] < reduction),
            other=0.0,
        )
        right = tl.load(
            right_pointer + reduction_offsets[:, None] * columns + offsets_n[None, :],
            mask=(reduction_offsets[:, None] < reduction)
            & (offsets_n[None, :] < columns),
            other=0.0,
        )
        accumulator += tl.dot(left, right)

    tl.store(
        output_pointer + offsets_m[:, None] * columns + offsets_n[None, :],
        accumulator,
        mask=(offsets_m[:, None] < rows) & (offsets_n[None, :] < columns),
    )


@triton.jit
def _attention_kernel(
    query_pointer,
    key_pointer,
    value_pointer,
    output_pointer,
    sequence: tl.constexpr,
    head_dimension: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    tensor_offset = batch_head * sequence * head_dimension

    query_rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    key_rows = tl.arange(0, BLOCK_N)
    dimensions = tl.arange(0, BLOCK_D)

    query = tl.load(
        query_pointer
        + tensor_offset
        + query_rows[:, None] * head_dimension
        + dimensions[None, :],
        mask=(query_rows[:, None] < sequence) & (dimensions[None, :] < head_dimension),
        other=0.0,
    )
    key_transposed = tl.load(
        key_pointer
        + tensor_offset
        + key_rows[None, :] * head_dimension
        + dimensions[:, None],
        mask=(key_rows[None, :] < sequence) & (dimensions[:, None] < head_dimension),
        other=0.0,
    )
    scores = tl.dot(query, key_transposed).to(tl.float32) * scale
    scores = tl.where(key_rows[None, :] < sequence, scores, -float("inf"))
    scores -= tl.max(scores, axis=1)[:, None]
    probabilities = tl.exp(scores)
    probabilities /= tl.sum(probabilities, axis=1)[:, None]

    value = tl.load(
        value_pointer
        + tensor_offset
        + key_rows[:, None] * head_dimension
        + dimensions[None, :],
        mask=(key_rows[:, None] < sequence) & (dimensions[None, :] < head_dimension),
        other=0.0,
    )
    attention = tl.dot(probabilities.to(tl.float16), value)
    tl.store(
        output_pointer
        + tensor_offset
        + query_rows[:, None] * head_dimension
        + dimensions[None, :],
        attention,
        mask=(query_rows[:, None] < sequence) & (dimensions[None, :] < head_dimension),
    )


@triton.jit
def _gqa_kernel(
    query_pointer,
    key_pointer,
    value_pointer,
    output_pointer,
    query_heads: tl.constexpr,
    key_value_heads: tl.constexpr,
    sequence: tl.constexpr,
    window: tl.constexpr,
    head_dimension: tl.constexpr,
    scale: tl.constexpr,
    causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_query_head = tl.program_id(1)
    batch = batch_query_head // query_heads
    query_head = batch_query_head % query_heads
    key_value_head = query_head // (query_heads // key_value_heads)

    query_offset = batch_query_head * sequence * head_dimension
    key_value_offset = (
        (batch * key_value_heads + key_value_head) * sequence * head_dimension
    )
    query_rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    key_rows = tl.arange(0, BLOCK_N)
    dimensions = tl.arange(0, BLOCK_D)

    query = tl.load(
        query_pointer
        + query_offset
        + query_rows[:, None] * head_dimension
        + dimensions[None, :],
        mask=(query_rows[:, None] < sequence) & (dimensions[None, :] < head_dimension),
        other=0.0,
    )
    key_transposed = tl.load(
        key_pointer
        + key_value_offset
        + key_rows[None, :] * head_dimension
        + dimensions[:, None],
        mask=(key_rows[None, :] < sequence) & (dimensions[:, None] < head_dimension),
        other=0.0,
    )
    scores = tl.dot(query, key_transposed).to(tl.float32) * scale
    valid_keys = key_rows[None, :] < sequence
    if causal:
        valid_keys = valid_keys & (key_rows[None, :] <= query_rows[:, None])
        valid_keys = valid_keys & (
            key_rows[None, :] > query_rows[:, None] - window
        )
    scores = tl.where(valid_keys, scores, -float("inf"))
    scores -= tl.max(scores, axis=1)[:, None]
    probabilities = tl.exp(scores)
    probabilities /= tl.sum(probabilities, axis=1)[:, None]

    value = tl.load(
        value_pointer
        + key_value_offset
        + key_rows[:, None] * head_dimension
        + dimensions[None, :],
        mask=(key_rows[:, None] < sequence) & (dimensions[None, :] < head_dimension),
        other=0.0,
    )
    attention = tl.dot(probabilities.to(tl.bfloat16), value)
    tl.store(
        output_pointer
        + query_offset
        + query_rows[:, None] * head_dimension
        + dimensions[None, :],
        attention,
        mask=(query_rows[:, None] < sequence) & (dimensions[None, :] < head_dimension),
    )


def _target_matches(reported: str, expected: str) -> bool:
    return reported == expected or reported.startswith(expected + ":")


def _reported_target() -> str:
    properties = torch.cuda.get_device_properties(0)
    for attribute in ("gcnArchName", "gcn_arch_name"):
        value = getattr(properties, attribute, None)
        if value:
            return str(value)
    raise RuntimeError(
        "ROCm PyTorch did not expose gcnArchName in device properties; "
        "target provenance cannot be validated"
    )


def _measure(launch: Callable[[], None], warmups: int, samples: int) -> list[int]:
    # Compile and initialize Triton's launch path before the requested warmups.
    launch()
    torch.cuda.synchronize()

    for _ in range(warmups):
        launch()
        torch.cuda.synchronize()

    durations: list[int] = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        launch()
        torch.cuda.synchronize()
        duration = time.perf_counter_ns() - start
        if duration <= 0:
            raise RuntimeError("measured a non-positive dispatch duration")
        durations.append(duration)
    return durations


def _deterministic_tensor(
    shape: tuple[int, ...], dtype: torch.dtype, phase: int = 0
) -> torch.Tensor:
    """Create repeatable, nonzero input data with one guest dispatch."""
    value = ((phase % 251) + 1) / 251.0
    return torch.full(shape, value, device="cuda", dtype=dtype)


def _case_parameters(case_id: str) -> dict[str, Any]:
    parameters = dict(CASES[case_id])
    parameters["input_pattern"] = "deterministic_nonzero_constant_by_phase"
    return parameters


def _run_copy(warmups: int, samples: int) -> tuple[dict[str, Any], list[int]]:
    parameters = _case_parameters("triton.copy_fp32_32m")
    elements = parameters["elements"]
    input_gpu = _deterministic_tensor((elements,), torch.float32)
    output_gpu = torch.empty_like(input_gpu)
    block_size, num_warps, num_stages = 1024, 4, 1
    grid = (triton.cdiv(elements, block_size),)
    parameters["launch"] = {
        "grid": list(grid),
        "block_size": block_size,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }

    def launch() -> None:
        _copy_kernel[grid](
            input_gpu,
            output_gpu,
            elements=elements,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return parameters, _measure(launch, warmups, samples)


def _run_vector_add(
    warmups: int, samples: int
) -> tuple[dict[str, Any], list[int]]:
    parameters = _case_parameters("triton.vector_add_fp32_boundary")
    elements = parameters["elements"]
    left_gpu = _deterministic_tensor((elements,), torch.float32)
    right_gpu = _deterministic_tensor((elements,), torch.float32, phase=83)
    output_gpu = torch.empty_like(left_gpu)
    block_size, num_warps, num_stages = 1024, 4, 1
    grid = (triton.cdiv(elements, block_size),)
    parameters["launch"] = {
        "grid": list(grid),
        "block_size": block_size,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }

    def launch() -> None:
        _vector_add_kernel[grid](
            left_gpu,
            right_gpu,
            output_gpu,
            elements=elements,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return parameters, _measure(launch, warmups, samples)


def _run_transpose(
    warmups: int, samples: int
) -> tuple[dict[str, Any], list[int]]:
    parameters = _case_parameters("triton.transpose_fp16_2048")
    rows, columns = parameters["rows"], parameters["columns"]
    input_gpu = _deterministic_tensor((rows, columns), torch.float16)
    output_gpu = torch.empty((columns, rows), device="cuda", dtype=torch.float16)
    tile, num_warps, num_stages = 32, 8, 1
    grid = (triton.cdiv(rows, tile), triton.cdiv(columns, tile))
    parameters["launch"] = {
        "grid": list(grid),
        "tile": [tile, tile],
        "num_warps": num_warps,
        "num_stages": num_stages,
    }

    def launch() -> None:
        _transpose_kernel[grid](
            input_gpu,
            output_gpu,
            rows=rows,
            columns=columns,
            TILE=tile,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return parameters, _measure(launch, warmups, samples)


def _run_gather(warmups: int, samples: int) -> tuple[dict[str, Any], list[int]]:
    parameters = _case_parameters("triton.gather_fp32_irregular")
    source_elements = parameters["source_elements"]
    output_elements = parameters["output_elements"]
    input_gpu = _deterministic_tensor((source_elements,), torch.float32)
    last_index = (
        parameters["index_offset"] + output_elements * parameters["index_stride"]
    )
    indices_gpu = torch.arange(
        parameters["index_offset"],
        last_index,
        parameters["index_stride"],
        device="cuda",
        dtype=torch.int64,
    )
    indices_gpu.bitwise_and_(source_elements - 1)
    output_gpu = torch.empty(output_elements, device="cuda", dtype=torch.float32)
    block_size, num_warps, num_stages = 512, 4, 1
    grid = (triton.cdiv(output_elements, block_size),)
    parameters["launch"] = {
        "grid": list(grid),
        "block_size": block_size,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }

    def launch() -> None:
        _gather_kernel[grid](
            input_gpu,
            indices_gpu,
            output_gpu,
            elements=output_elements,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return parameters, _measure(launch, warmups, samples)


def _run_atomic_add(
    warmups: int, samples: int
) -> tuple[dict[str, Any], list[int]]:
    parameters = _case_parameters("triton.atomic_add_fp32_contended")
    elements, buckets = parameters["elements"], parameters["buckets"]
    input_gpu = _deterministic_tensor((elements,), torch.float32)
    output_gpu = torch.zeros(buckets, device="cuda", dtype=torch.float32)
    block_size, num_warps, num_stages = 256, 8, 1
    grid = (triton.cdiv(elements, block_size),)
    parameters["launch"] = {
        "grid": list(grid),
        "block_size": block_size,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }

    def launch() -> None:
        _atomic_add_kernel[grid](
            input_gpu,
            output_gpu,
            elements=elements,
            buckets=buckets,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return parameters, _measure(launch, warmups, samples)


def _run_softmax(
    case_id: str, warmups: int, samples: int
) -> tuple[dict[str, Any], list[int]]:
    parameters = _case_parameters(case_id)
    rows = parameters["rows"]
    columns = parameters["columns"]
    input_gpu = _deterministic_tensor((rows, columns), torch.float16)
    output_gpu = torch.empty_like(input_gpu)
    block_size = triton.next_power_of_2(columns)
    num_warps, num_stages = 8, 1
    grid = (rows,)
    parameters["launch"] = {
        "grid": list(grid),
        "block_size": block_size,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }

    def launch() -> None:
        _softmax_kernel[grid](
            input_gpu,
            output_gpu,
            row_width=columns,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return parameters, _measure(launch, warmups, samples)


def _run_rmsnorm(warmups: int, samples: int) -> tuple[dict[str, Any], list[int]]:
    parameters = _case_parameters("triton.rmsnorm_bf16")
    rows = parameters["rows"]
    columns = parameters["columns"]
    epsilon = parameters["epsilon"]
    input_gpu = _deterministic_tensor((rows, columns), torch.bfloat16)
    output_gpu = torch.empty_like(input_gpu)
    block_size = triton.next_power_of_2(columns)
    num_warps, num_stages = 8, 1
    grid = (rows,)
    parameters["launch"] = {
        "grid": list(grid),
        "block_size": block_size,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }

    def launch() -> None:
        _rmsnorm_kernel[grid](
            input_gpu,
            output_gpu,
            row_width=columns,
            epsilon=epsilon,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return parameters, _measure(launch, warmups, samples)


def _run_gemm(
    case_id: str, warmups: int, samples: int
) -> tuple[dict[str, Any], list[int]]:
    parameters = _case_parameters(case_id)
    rows, columns, reduction = parameters["m"], parameters["n"], parameters["k"]
    left_gpu = _deterministic_tensor((rows, reduction), torch.bfloat16)
    right_gpu = _deterministic_tensor(
        (reduction, columns), torch.bfloat16, phase=83
    )
    output_gpu = torch.empty((rows, columns), device="cuda", dtype=torch.bfloat16)
    block_m, block_n, block_k = 64, 64, 32
    num_warps, num_stages = 4, 2
    grid = (triton.cdiv(rows, block_m) * triton.cdiv(columns, block_n),)
    parameters["launch"] = {
        "grid": list(grid),
        "block": [block_m, block_n, block_k],
        "num_warps": num_warps,
        "num_stages": num_stages,
    }

    def launch() -> None:
        _gemm_kernel[grid](
            left_gpu,
            right_gpu,
            output_gpu,
            rows=rows,
            columns=columns,
            reduction=reduction,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return parameters, _measure(launch, warmups, samples)


def _run_attention(warmups: int, samples: int) -> tuple[dict[str, Any], list[int]]:
    parameters = _case_parameters("triton.attention_fp16")
    batch = parameters["batch"]
    heads = parameters["heads"]
    sequence = parameters["sequence"]
    head_dimension = parameters["head_dimension"]
    shape = (batch, heads, sequence, head_dimension)
    scale = 1.0 / math.sqrt(head_dimension)
    query_gpu = _deterministic_tensor(shape, torch.float16)
    key_gpu = _deterministic_tensor(shape, torch.float16, phase=83)
    value_gpu = _deterministic_tensor(shape, torch.float16, phase=167)
    output_gpu = torch.empty_like(query_gpu)
    block_m, block_n, block_d = 16, 128, 64
    num_warps, num_stages = 8, 2
    grid = (triton.cdiv(sequence, block_m), batch * heads)
    parameters["launch"] = {
        "grid": list(grid),
        "block": [block_m, block_n, block_d],
        "num_warps": num_warps,
        "num_stages": num_stages,
    }

    def launch() -> None:
        _attention_kernel[grid](
            query_gpu,
            key_gpu,
            value_gpu,
            output_gpu,
            sequence=sequence,
            head_dimension=head_dimension,
            scale=scale,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return parameters, _measure(launch, warmups, samples)


def _run_gpt_oss_rmsnorm(
    warmups: int, samples: int
) -> tuple[dict[str, Any], list[int]]:
    parameters = _case_parameters("triton.gpt_oss_rmsnorm_bf16")
    rows, columns = parameters["rows"], parameters["columns"]
    epsilon = parameters["epsilon"]
    input_gpu = _deterministic_tensor((rows, columns), torch.bfloat16)
    weight_gpu = _deterministic_tensor((columns,), torch.bfloat16, phase=83)
    output_gpu = torch.empty_like(input_gpu)
    block_size = triton.next_power_of_2(columns)
    num_warps, num_stages = 8, 1
    grid = (rows,)
    parameters["weighted"] = True
    parameters["launch"] = {
        "grid": list(grid),
        "block_size": block_size,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }

    def launch() -> None:
        _weighted_rmsnorm_kernel[grid](
            input_gpu,
            weight_gpu,
            output_gpu,
            row_width=columns,
            epsilon=epsilon,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return parameters, _measure(launch, warmups, samples)


def _run_gpt_oss_gqa(
    warmups: int, samples: int
) -> tuple[dict[str, Any], list[int]]:
    parameters = _case_parameters("triton.gpt_oss_gqa_bf16")
    batch = parameters["batch"]
    query_heads = parameters["query_heads"]
    key_value_heads = parameters["key_value_heads"]
    sequence = parameters["sequence"]
    window = parameters["window"]
    head_dimension = parameters["head_dimension"]
    query_shape = (batch, query_heads, sequence, head_dimension)
    key_value_shape = (batch, key_value_heads, sequence, head_dimension)
    query_gpu = _deterministic_tensor(query_shape, torch.bfloat16)
    key_gpu = _deterministic_tensor(key_value_shape, torch.bfloat16, phase=83)
    value_gpu = _deterministic_tensor(key_value_shape, torch.bfloat16, phase=167)
    output_gpu = torch.empty_like(query_gpu)
    scale = 1.0 / math.sqrt(head_dimension)
    block_m, block_n, block_d = 16, 128, 64
    num_warps, num_stages = 8, 2
    grid = (triton.cdiv(sequence, block_m), batch * query_heads)
    parameters["query_heads_per_key_value_head"] = query_heads // key_value_heads
    parameters["launch"] = {
        "grid": list(grid),
        "block": [block_m, block_n, block_d],
        "num_warps": num_warps,
        "num_stages": num_stages,
    }

    def launch() -> None:
        _gqa_kernel[grid](
            query_gpu,
            key_gpu,
            value_gpu,
            output_gpu,
            query_heads=query_heads,
            key_value_heads=key_value_heads,
            sequence=sequence,
            window=window,
            head_dimension=head_dimension,
            scale=scale,
            causal=parameters["causal"],
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return parameters, _measure(launch, warmups, samples)


def _write_result(path: str, result: dict[str, Any]) -> None:
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path == "-":
        sys.stdout.write(encoded)
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--output", default="-")
    arguments = parser.parse_args()
    if arguments.warmups < 0:
        parser.error("--warmups must be nonnegative")
    if arguments.samples <= 0:
        parser.error("--samples must be positive")
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError(
            "a ROCm PyTorch build and visible rocjitsu HIP device are required"
        )

    torch.cuda.set_device(0)
    reported_target = _reported_target()
    if not _target_matches(reported_target, arguments.target):
        raise RuntimeError(
            f"runtime reported target {reported_target!r}, expected {arguments.target!r}"
        )

    if arguments.case == "triton.copy_fp32_32m":
        parameters, durations = _run_copy(arguments.warmups, arguments.samples)
    elif arguments.case == "triton.vector_add_fp32_boundary":
        parameters, durations = _run_vector_add(arguments.warmups, arguments.samples)
    elif arguments.case == "triton.transpose_fp16_2048":
        parameters, durations = _run_transpose(arguments.warmups, arguments.samples)
    elif arguments.case == "triton.gather_fp32_irregular":
        parameters, durations = _run_gather(arguments.warmups, arguments.samples)
    elif arguments.case == "triton.atomic_add_fp32_contended":
        parameters, durations = _run_atomic_add(arguments.warmups, arguments.samples)
    elif arguments.case.startswith("triton.softmax"):
        parameters, durations = _run_softmax(
            arguments.case, arguments.warmups, arguments.samples
        )
    elif arguments.case == "triton.rmsnorm_bf16":
        parameters, durations = _run_rmsnorm(arguments.warmups, arguments.samples)
    elif arguments.case.startswith("triton.gemm"):
        parameters, durations = _run_gemm(
            arguments.case, arguments.warmups, arguments.samples
        )
    elif arguments.case == "triton.attention_fp16":
        parameters, durations = _run_attention(arguments.warmups, arguments.samples)
    elif arguments.case == "triton.gpt_oss_rmsnorm_bf16":
        parameters, durations = _run_gpt_oss_rmsnorm(
            arguments.warmups, arguments.samples
        )
    else:
        parameters, durations = _run_gpt_oss_gqa(
            arguments.warmups, arguments.samples
        )

    result = {
        "schema": SCHEMA,
        "case": arguments.case,
        "target": arguments.target,
        "provider": "triton",
        "parameters": parameters,
        "timings_ns": durations,
    }
    _write_result(arguments.output, result)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # Keep the runner's stderr artifact actionable.
        print(f"rocjitsu Triton benchmark: {error}", file=sys.stderr)
        raise SystemExit(2) from error
