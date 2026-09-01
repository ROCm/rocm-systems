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
}


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


def _run_softmax(
    case_id: str, warmups: int, samples: int
) -> tuple[dict[str, Any], list[int]]:
    parameters = dict(CASES[case_id])
    rows = parameters["rows"]
    columns = parameters["columns"]
    input_gpu = torch.zeros((rows, columns), device="cuda", dtype=torch.float16)
    output_gpu = torch.empty_like(input_gpu)
    block_size = triton.next_power_of_2(columns)

    def launch() -> None:
        _softmax_kernel[(rows,)](
            input_gpu, output_gpu, row_width=columns, BLOCK_SIZE=block_size, num_warps=8
        )

    return parameters, _measure(launch, warmups, samples)


def _run_rmsnorm(warmups: int, samples: int) -> tuple[dict[str, Any], list[int]]:
    parameters = dict(CASES["triton.rmsnorm_bf16"])
    rows = parameters["rows"]
    columns = parameters["columns"]
    epsilon = parameters["epsilon"]
    input_gpu = torch.zeros((rows, columns), device="cuda", dtype=torch.bfloat16)
    output_gpu = torch.empty_like(input_gpu)

    def launch() -> None:
        _rmsnorm_kernel[(rows,)](
            input_gpu,
            output_gpu,
            row_width=columns,
            epsilon=epsilon,
            BLOCK_SIZE=columns,
            num_warps=8,
        )

    return parameters, _measure(launch, warmups, samples)


def _run_gemm(
    case_id: str, warmups: int, samples: int
) -> tuple[dict[str, Any], list[int]]:
    parameters = dict(CASES[case_id])
    rows, columns, reduction = parameters["m"], parameters["n"], parameters["k"]
    left_gpu = torch.zeros((rows, reduction), device="cuda", dtype=torch.bfloat16)
    right_gpu = torch.zeros((reduction, columns), device="cuda", dtype=torch.bfloat16)
    output_gpu = torch.empty((rows, columns), device="cuda", dtype=torch.bfloat16)
    block_m, block_n, block_k = 64, 64, 32
    grid = (triton.cdiv(rows, block_m) * triton.cdiv(columns, block_n),)

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
            num_warps=4,
            num_stages=2,
        )

    return parameters, _measure(launch, warmups, samples)


def _run_attention(warmups: int, samples: int) -> tuple[dict[str, Any], list[int]]:
    parameters = dict(CASES["triton.attention_fp16"])
    batch = parameters["batch"]
    heads = parameters["heads"]
    sequence = parameters["sequence"]
    head_dimension = parameters["head_dimension"]
    shape = (batch, heads, sequence, head_dimension)
    scale = 1.0 / math.sqrt(head_dimension)
    query_gpu = torch.zeros(shape, device="cuda", dtype=torch.float16)
    key_gpu = torch.zeros(shape, device="cuda", dtype=torch.float16)
    value_gpu = torch.zeros(shape, device="cuda", dtype=torch.float16)
    output_gpu = torch.empty_like(query_gpu)
    block_m, block_n, block_d = 16, 128, 64
    grid = (triton.cdiv(sequence, block_m), batch * heads)

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
            num_warps=8,
            num_stages=2,
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

    if arguments.case.startswith("triton.softmax"):
        parameters, durations = _run_softmax(
            arguments.case, arguments.warmups, arguments.samples
        )
    elif arguments.case == "triton.rmsnorm_bf16":
        parameters, durations = _run_rmsnorm(arguments.warmups, arguments.samples)
    elif arguments.case.startswith("triton.gemm"):
        parameters, durations = _run_gemm(
            arguments.case, arguments.warmups, arguments.samples
        )
    else:
        parameters, durations = _run_attention(arguments.warmups, arguments.samples)

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
