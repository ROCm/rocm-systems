#!/usr/bin/env python3
"""Runs deterministic PyTorch/Triton workloads used by ConSan validation."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import statistics
import time

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl


def _write_oracle_result(outcome: str, detail: object) -> None:
    result_path = os.environ.get("CONSAN_ROW_RESULT_PATH") or os.environ.get(
        "CONSAN_WORKLOAD_RESULT_PATH"
    )
    if not result_path:
        return
    payload = {
        "schema_version": 1,
        "oracle": outcome,
        "detail": detail,
        "source_diagnostics": {
            "outcome": "not_applicable",
            "count": None,
            "expectation": "not_applicable",
            "detail": "PyTorch numeric oracle has no separate source-diagnostic channel",
        },
    }
    path = Path(result_path)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


@triton.jit
def _descriptor_add_kernel(
    lhs,
    rhs,
    output,
    rows: tl.constexpr,
    columns: tl.constexpr,
    block_rows: tl.constexpr,
    block_columns: tl.constexpr,
):
    lhs_descriptor = tl.make_tensor_descriptor(
        lhs,
        shape=[rows, columns],
        strides=[columns, 1],
        block_shape=[block_rows, block_columns],
    )
    rhs_descriptor = tl.make_tensor_descriptor(
        rhs,
        shape=[rows, columns],
        strides=[columns, 1],
        block_shape=[block_rows, block_columns],
    )
    output_descriptor = tl.make_tensor_descriptor(
        output,
        shape=[rows, columns],
        strides=[columns, 1],
        block_shape=[block_rows, block_columns],
    )
    offsets = [tl.program_id(0) * block_rows, tl.program_id(1) * block_columns]
    output_descriptor.store(
        offsets, lhs_descriptor.load(offsets) + rhs_descriptor.load(offsets)
    )


def _descriptor_allocator(size: int, alignment: int, stream: int | None):
    del alignment, stream
    return torch.empty(size, device="cuda", dtype=torch.int8)


@gluon.jit
def _cluster_load_kernel(
    input_pointer,
    output_pointer,
    rows,
    columns,
    block_rows: ttgl.constexpr,
    block_columns: ttgl.constexpr,
    blocked_layout: ttgl.constexpr,
):
    program = ttgl.program_id(axis=0)
    row_programs = ttgl.cdiv(rows, block_rows)
    row_program = program % row_programs
    column_program = program // row_programs
    row_offsets = row_program * block_rows + ttgl.arange(
        0, block_rows, layout=ttgl.SliceLayout(1, blocked_layout)
    )
    column_offsets = column_program * block_columns + ttgl.arange(
        0, block_columns, layout=ttgl.SliceLayout(0, blocked_layout)
    )
    offsets = row_offsets[:, None] * columns + column_offsets[None, :]
    mask = (row_offsets[:, None] < rows) & (column_offsets[None, :] < columns)
    values = ttgl.load(input_pointer + offsets, mask)
    ttgl.store(output_pointer + offsets, values, mask)


@gluon.jit
def _cluster_barrier_kernel(
    output_pointer,
    output_elements: ttgl.constexpr,
    blocked_layout: ttgl.constexpr,
    shared_layout: ttgl.constexpr,
):
    offsets = ttgl.arange(0, output_elements, layout=blocked_layout)
    program = ttgl.program_id(axis=0)
    values = program * output_elements + offsets
    shared = ttgl.allocate_shared_memory(ttgl.int32, [output_elements], shared_layout)
    shared.store(values)
    ttgl.amd.gfx1250.cluster.arrive()
    ttgl.amd.gfx1250.cluster.wait()
    restored = shared.load(blocked_layout)
    ttgl.store(output_pointer + program * output_elements + offsets, restored)


def _run_cluster_load_sync(repetitions: int) -> dict[str, object]:
    rows = 128
    columns = 128
    block_rows = 64
    block_columns = 64
    cluster_ctas = 2
    blocked_layout = ttgl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, 8],
        warps_per_cta=[1, 2],
        order=[1, 0],
        cga_layout=[[0, 1]],
    )
    host_input = torch.arange(rows * columns, dtype=torch.float32).reshape(
        rows, columns
    )
    input_tensor = host_input.to(device="cuda")
    grid = (rows // block_rows * (columns // block_columns), 1)
    # Clustered launch geometry changes physical CTA grouping without
    # multiplying the logical program-id domain exposed to the kernel.
    cluster_programs = 4
    barrier_elements = 512
    barrier_blocked_layout = ttgl.BlockedLayout(
        size_per_thread=[1],
        threads_per_warp=[32],
        warps_per_cta=[8],
        order=[0],
        cga_layout=[[1]],
    )
    barrier_shared_layout = ttgl.PaddedSharedLayout.with_identity_for(
        interval_padding_pairs=[[256, 1]],
        shape=[barrier_elements],
        order=[0],
        cga_layout=[[1]],
    )
    expected_barrier = torch.arange(
        cluster_programs * barrier_elements, dtype=torch.int32
    ).reshape(cluster_programs, barrier_elements)
    elapsed_ms = []
    output = None
    barrier_output = None
    for _ in range(repetitions):
        output = torch.full_like(input_tensor, float("nan"))
        barrier_output = torch.full(
            (cluster_programs, barrier_elements),
            -1,
            dtype=torch.int32,
            device="cuda",
        )
        start = time.monotonic()
        _cluster_load_kernel[grid](
            input_tensor,
            output,
            rows,
            columns,
            block_rows,
            block_columns,
            blocked_layout,
            num_warps=2,
            num_ctas=cluster_ctas,
        )
        _cluster_barrier_kernel[(4,)](
            barrier_output,
            barrier_elements,
            barrier_blocked_layout,
            barrier_shared_layout,
            num_warps=8,
            num_ctas=cluster_ctas,
        )
        torch.cuda.synchronize()
        elapsed_ms.append((time.monotonic() - start) * 1000.0)
    assert output is not None
    assert barrier_output is not None
    if not torch.equal(output.cpu(), host_input):
        raise RuntimeError("cluster-load exact-copy oracle failed")
    if not torch.equal(barrier_output.cpu(), expected_barrier):
        raise RuntimeError(
            "cluster-barrier sentinel oracle failed: "
            f"got {barrier_output.cpu().tolist()}, expected {expected_barrier.tolist()}"
        )
    return {
        "median_ms": statistics.median(elapsed_ms),
        "repetitions": repetitions,
        "oracle": "exact-cluster-copy-and-barrier-sentinels",
        "oracle_passed": True,
        "cluster_ctas": cluster_ctas,
        "cluster_load_shape": [rows, columns],
        "barrier_programs": cluster_programs,
        "barrier_lds_elements_per_program": barrier_elements,
    }


def _run_descriptor_add(num_ctas: int, repetitions: int) -> dict[str, object]:
    rows = 32
    columns = 32
    block_rows = 16
    block_columns = 16
    lhs = torch.arange(rows * columns, device="cuda", dtype=torch.float32).reshape(
        rows, columns
    )
    rhs = torch.arange(
        rows * columns, 2 * rows * columns, device="cuda", dtype=torch.float32
    ).reshape(rows, columns)
    expected = lhs.cpu() + rhs.cpu()
    elapsed_ms = []
    output = None
    for _ in range(repetitions):
        output = torch.full_like(lhs, float("nan"))
        start = time.monotonic()
        _descriptor_add_kernel[(rows // block_rows, columns // block_columns)](
            lhs,
            rhs,
            output,
            rows,
            columns,
            block_rows,
            block_columns,
            num_warps=4,
            num_ctas=num_ctas,
        )
        torch.cuda.synchronize()
        elapsed_ms.append((time.monotonic() - start) * 1000.0)
    assert output is not None
    actual = output.cpu()
    matches = torch.equal(actual, expected)
    if not matches:
        maximum_error = torch.max(torch.abs(actual - expected)).item()
        raise RuntimeError(f"descriptor-add oracle failed: max error {maximum_error}")
    return {
        "median_ms": statistics.median(elapsed_ms),
        "repetitions": repetitions,
        "oracle": "exact",
        "oracle_passed": True,
        "num_ctas": num_ctas,
    }


def _run_mode(repetitions: int) -> dict[str, object]:
    rows = 1
    columns = 128
    mode_stride = 32
    host_input = torch.arange(rows * columns, dtype=torch.int32).reshape(rows, columns)
    expected_values = torch.empty(rows, dtype=torch.int32)
    for row in range(rows):
        mode_value = -(row + 1)
        host_input[row, ::mode_stride] = mode_value
        expected_values[row] = mode_value

    input_tensor = host_input.to(device="cuda")
    elapsed_ms = []
    actual_values = None
    actual_indices = None
    for _ in range(repetitions):
        start = time.monotonic()
        actual_values, actual_indices = torch.mode(input_tensor, dim=1)
        torch.cuda.synchronize()
        elapsed_ms.append((time.monotonic() - start) * 1000.0)
    assert actual_values is not None
    assert actual_indices is not None
    host_values = actual_values.cpu()
    host_indices = actual_indices.cpu()
    if not torch.equal(host_values, expected_values):
        raise RuntimeError(
            f"mode value oracle failed: got {host_values.tolist()}, "
            f"expected {expected_values.tolist()}"
        )
    indices_in_range = bool(torch.all((host_indices >= 0) & (host_indices < columns)))
    indexed_values = (
        host_input[torch.arange(rows), host_indices] if indices_in_range else None
    )
    if indexed_values is None or not torch.equal(indexed_values, expected_values):
        raise RuntimeError(
            f"mode index oracle failed: indices {host_indices.tolist()} do not select "
            f"the expected values {expected_values.tolist()}"
        )
    return {
        "median_ms": statistics.median(elapsed_ms),
        "repetitions": repetitions,
        "oracle": "exact-value-and-valid-index",
        "oracle_passed": True,
        "rows": rows,
        "columns": columns,
    }


def _topk_input(rows: int, columns: int, dtype: torch.dtype) -> torch.Tensor:
    permutation = (torch.arange(columns, dtype=torch.int64) * 73) % columns
    if dtype == torch.bfloat16:
        # Use distinct positive finite BF16 bit patterns.  Converting a long
        # integer range to BF16 would introduce ties and make the index oracle
        # ambiguous.
        ordered = (
            (torch.arange(columns, dtype=torch.int32) + 0x3C00)
            .to(torch.int16)
            .view(torch.bfloat16)
        )
    else:
        ordered = torch.arange(columns, dtype=dtype)
    row = ordered[permutation]
    return torch.stack([row.clone() for _ in range(rows)])


def _run_topk_case(
    repetitions: int,
    *,
    dtype: torch.dtype,
    rows: int,
    columns: int,
    k: int,
) -> dict[str, object]:
    host_input = _topk_input(rows, columns, dtype)
    expected_values, expected_indices = torch.topk(
        host_input, k, dim=1, largest=True, sorted=True
    )
    input_tensor = host_input.to(device="cuda")
    elapsed_ms = []
    actual_values = None
    actual_indices = None
    for _ in range(repetitions):
        start = time.monotonic()
        actual_values, actual_indices = torch.topk(
            input_tensor, k, dim=1, largest=True, sorted=True
        )
        torch.cuda.synchronize()
        elapsed_ms.append((time.monotonic() - start) * 1000.0)
    assert actual_values is not None
    assert actual_indices is not None
    host_values = actual_values.cpu()
    host_indices = actual_indices.cpu()
    if not torch.equal(host_values, expected_values):
        raise RuntimeError(
            f"topk value oracle failed for {dtype}: got {host_values.tolist()}, "
            f"expected {expected_values.tolist()}"
        )
    if not torch.equal(host_indices, expected_indices):
        raise RuntimeError(
            f"topk index oracle failed for {dtype}: got {host_indices.tolist()}, "
            f"expected {expected_indices.tolist()}"
        )
    return {
        "median_ms": statistics.median(elapsed_ms),
        "repetitions": repetitions,
        "oracle": "exact-sorted-values-and-indices",
        "oracle_passed": True,
        "dtype": str(dtype),
        "rows": rows,
        "columns": columns,
        "k": k,
    }


def _run_topk(repetitions: int) -> dict[str, object]:
    return {
        "double-spill": _run_topk_case(
            repetitions, dtype=torch.float64, rows=8, columns=128, k=16
        ),
        "bf16-coverage": _run_topk_case(
            repetitions, dtype=torch.bfloat16, rows=8, columns=1024, k=32
        ),
    }


def _run_rdna4_llm_topk(repetitions: int) -> dict[str, object]:
    """Runs decode-style top-k selection over a Qwen-sized vocabulary."""
    return {
        "qwen-vocabulary": _run_topk_case(
            repetitions,
            dtype=torch.float32,
            rows=1,
            columns=151936,
            k=50,
        )
    }


def _run_sort(repetitions: int) -> dict[str, object]:
    rows = 4
    columns = 256
    permutation = (torch.arange(columns, dtype=torch.int64) * 73) % columns
    ordered = torch.arange(columns, dtype=torch.float32)
    host_input = torch.stack(
        [ordered[permutation] + row * columns for row in range(rows)]
    )
    expected_values, expected_indices = torch.sort(host_input, dim=1)
    input_tensor = host_input.to(device="cuda")
    elapsed_ms = []
    actual_values = None
    actual_indices = None
    for _ in range(repetitions):
        start = time.monotonic()
        actual_values, actual_indices = torch.sort(input_tensor, dim=1)
        torch.cuda.synchronize()
        elapsed_ms.append((time.monotonic() - start) * 1000.0)
    assert actual_values is not None
    assert actual_indices is not None
    if not torch.equal(actual_values.cpu(), expected_values):
        raise RuntimeError("sort value oracle failed")
    if not torch.equal(actual_indices.cpu(), expected_indices):
        raise RuntimeError("sort index oracle failed")
    return {
        "median_ms": statistics.median(elapsed_ms),
        "repetitions": repetitions,
        "oracle": "exact-sorted-values-and-indices",
        "oracle_passed": True,
        "rows": rows,
        "columns": columns,
    }


def _run_scatter_reduce_case(repetitions: int, dtype: torch.dtype) -> dict[str, object]:
    bins = 64
    elements = 1024
    host_indices = torch.arange(elements, dtype=torch.int64) % bins
    expected = torch.bincount(host_indices, minlength=bins).to(dtype=dtype)
    indices = host_indices.to(device="cuda")
    source = torch.ones(elements, dtype=dtype, device="cuda")
    elapsed_ms = []
    actual = None
    for _ in range(repetitions):
        output = torch.zeros(bins, dtype=dtype, device="cuda")
        start = time.monotonic()
        actual = output.scatter_reduce(
            0, indices, source, reduce="sum", include_self=True
        )
        torch.cuda.synchronize()
        elapsed_ms.append((time.monotonic() - start) * 1000.0)
    assert actual is not None
    if not torch.equal(actual.cpu(), expected):
        raise RuntimeError(f"scatter-reduce oracle failed for {dtype}")
    return {
        "median_ms": statistics.median(elapsed_ms),
        "repetitions": repetitions,
        "oracle": "exact-collision-counts",
        "oracle_passed": True,
        "dtype": str(dtype),
        "bins": bins,
        "elements": elements,
    }


def _run_scatter_reduce(repetitions: int) -> dict[str, object]:
    return {
        "bf16": _run_scatter_reduce_case(repetitions, torch.bfloat16),
        "fp32": _run_scatter_reduce_case(repetitions, torch.float32),
    }


def _run_histc(repetitions: int) -> dict[str, object]:
    bins = 64
    repeats = 16
    host_input = torch.arange(bins, dtype=torch.float32).repeat_interleave(repeats)
    expected = torch.full((bins,), repeats, dtype=torch.float32)
    input_tensor = host_input.to(device="cuda")
    elapsed_ms = []
    actual = None
    for _ in range(repetitions):
        start = time.monotonic()
        actual = torch.histc(input_tensor, bins=bins, min=0, max=bins - 1)
        torch.cuda.synchronize()
        elapsed_ms.append((time.monotonic() - start) * 1000.0)
    assert actual is not None
    if not torch.equal(actual.cpu(), expected):
        raise RuntimeError("histc oracle failed")
    return {
        "median_ms": statistics.median(elapsed_ms),
        "repetitions": repetitions,
        "oracle": "exact-bin-counts",
        "oracle_passed": True,
        "bins": bins,
        "elements": bins * repeats,
    }


def _run_norm_softmax(
    repetitions: int, *, run_norm: bool = True, run_softmax: bool = True
) -> dict[str, object]:
    norm_input = torch.zeros((8, 1024), dtype=torch.float32)
    norm_input[:, 0] = 3.0
    norm_input[:, 1] = 4.0
    expected_norm = torch.full((8,), 5.0, dtype=torch.float32)
    softmax_input = torch.linspace(-4.0, 4.0, 1024, dtype=torch.float32).repeat(4, 1)
    expected_softmax = torch.softmax(softmax_input, dim=1)
    device_norm_input = norm_input.to(device="cuda")
    device_softmax_input = softmax_input.to(device="cuda")
    elapsed_ms = []
    actual_norm = None
    actual_softmax = None
    for _ in range(repetitions):
        start = time.monotonic()
        if run_norm:
            actual_norm = torch.linalg.vector_norm(device_norm_input, dim=1)
        if run_softmax:
            actual_softmax = torch.softmax(device_softmax_input, dim=1)
        torch.cuda.synchronize()
        elapsed_ms.append((time.monotonic() - start) * 1000.0)
    if run_norm and (
        actual_norm is None or not torch.equal(actual_norm.cpu(), expected_norm)
    ):
        raise RuntimeError("vector-norm oracle failed")
    if run_softmax and (
        actual_softmax is None
        or not torch.allclose(
            actual_softmax.cpu(), expected_softmax, rtol=1.0e-5, atol=1.0e-7
        )
    ):
        raise RuntimeError("softmax oracle failed")
    return {
        "median_ms": statistics.median(elapsed_ms),
        "repetitions": repetitions,
        "oracle": "exact-3-4-5-norm-and-cpu-softmax",
        "oracle_passed": True,
        "norm_shape": list(norm_input.shape) if run_norm else None,
        "softmax_shape": list(softmax_input.shape) if run_softmax else None,
    }


def _run_rdna4_compiled_softmax(repetitions: int) -> dict[str, object]:
    """Runs a target-native Inductor/Triton softmax selected on gfx1201."""
    rows = 128
    columns = 256
    host_input = (
        (torch.arange(rows * columns, dtype=torch.float32) % 257) - 128
    ).reshape(rows, columns) / 17.0
    expected = torch.softmax(host_input, dim=1)
    input_tensor = host_input.to(device="cuda")

    # Keep this an ordinary PyTorch operation.  torch.compile, rather than a
    # hand-written Triton kernel, is what makes the frozen validation client
    # exercise the target-native kernel selected by the installed wheel.
    compiled_softmax = torch.compile(
        lambda value: torch.softmax(value, dim=1), fullgraph=True
    )
    elapsed_ms = []
    actual = None
    for _ in range(repetitions):
        start = time.monotonic()
        actual = compiled_softmax(input_tensor)
        torch.cuda.synchronize()
        elapsed_ms.append((time.monotonic() - start) * 1000.0)
    assert actual is not None
    host_actual = actual.cpu()
    if not torch.allclose(host_actual, expected, rtol=1.0e-5, atol=1.0e-7):
        maximum_error = torch.max(torch.abs(host_actual - expected)).item()
        raise RuntimeError(
            f"compiled-softmax oracle failed: maximum error {maximum_error}"
        )
    return {
        "median_ms": statistics.median(elapsed_ms),
        "repetitions": repetitions,
        "oracle": "cpu-softmax-allclose",
        "oracle_passed": True,
        "shape": [rows, columns],
        "dtype": str(host_input.dtype),
        "compiler": "torch.compile-fullgraph",
    }


def _run_rdna4_split_softmax(repetitions: int) -> dict[str, object]:
    """Runs upstream PyTorch's target-native split online-softmax shape."""
    rows = 1
    columns = 2**20 + 13
    host_input = (
        (((torch.arange(columns, dtype=torch.int64) * 17) % 257) - 128)
        .to(torch.bfloat16)
        .reshape(rows, columns)
    )
    # Derive the reference independently in FP32, then round it once to the
    # operation's BF16 result type. This retains an exact output oracle without
    # loading PyTorch's unrelated precompiled RNG object during setup.
    expected = torch.softmax(host_input.to(torch.float32), dim=-1).to(torch.bfloat16)
    input_tensor = host_input.to(device="cuda")
    compiled_softmax = torch.compile(
        lambda value: torch.softmax(value, dim=-1), fullgraph=True
    )
    elapsed_ms = []
    actual = None
    for _ in range(repetitions):
        start = time.monotonic()
        actual = compiled_softmax(input_tensor)
        torch.cuda.synchronize()
        elapsed_ms.append((time.monotonic() - start) * 1000.0)
    assert actual is not None
    host_actual = actual.cpu()
    if not torch.equal(host_actual, expected):
        maximum_error = torch.max(
            torch.abs(host_actual.to(torch.float32) - expected.to(torch.float32))
        ).item()
        raise RuntimeError(
            f"split-softmax oracle failed: maximum error {maximum_error}"
        )
    return {
        "median_ms": statistics.median(elapsed_ms),
        "repetitions": repetitions,
        "oracle": "independent-fp32-cpu-softmax-rounded-exactly-to-bf16",
        "oracle_passed": True,
        "shape": [rows, columns],
        "dtype": str(host_input.dtype),
        "compiler": "torch.compile-fullgraph",
        "selection_source": "test/inductor/test_online_softmax.py::test_split_reduction",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload",
        choices=(
            "tdm-descriptor-add",
            "cluster-load-sync",
            "torch-mode",
            "torch-topk",
            "torch-sort",
            "scatter-reduce",
            "torch-histc",
            "norm-softmax",
            "vector-norm",
            "softmax",
            "rdna4-compiled-softmax",
            "rdna4-split-softmax",
            "rdna4-llm-topk",
        ),
        required=True,
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    return args


def main() -> int:
    args = _parse_args()
    try:
        if args.workload == "tdm-descriptor-add":
            triton.set_allocator(_descriptor_allocator)
            result = {
                "one-cta": _run_descriptor_add(1, args.repetitions),
                "two-cta-cluster": _run_descriptor_add(2, args.repetitions),
            }
        elif args.workload == "cluster-load-sync":
            result = {"cluster-load-and-sync": _run_cluster_load_sync(args.repetitions)}
        elif args.workload == "torch-mode":
            result = {"large-row": _run_mode(args.repetitions)}
        elif args.workload == "torch-topk":
            result = _run_topk(args.repetitions)
        elif args.workload == "torch-sort":
            result = {"segmented-rows": _run_sort(args.repetitions)}
        elif args.workload == "scatter-reduce":
            result = _run_scatter_reduce(args.repetitions)
        elif args.workload == "torch-histc":
            result = {"shared-bin-count": _run_histc(args.repetitions)}
        elif args.workload == "norm-softmax":
            component = os.environ.get("CONSAN_PYTORCH_REDUCTION_COMPONENT", "both")
            if component not in ("both", "norm", "softmax"):
                raise RuntimeError(
                    "CONSAN_PYTORCH_REDUCTION_COMPONENT must be both, norm, or softmax"
                )
            result = {
                "norm-and-softmax": _run_norm_softmax(
                    args.repetitions,
                    run_norm=component != "softmax",
                    run_softmax=component != "norm",
                )
            }
        elif args.workload == "vector-norm":
            result = {
                "vector-norm": _run_norm_softmax(args.repetitions, run_softmax=False)
            }
        elif args.workload == "softmax":
            result = {"softmax": _run_norm_softmax(args.repetitions, run_norm=False)}
        elif args.workload == "rdna4-compiled-softmax":
            result = {
                "rdna4-compiled-softmax": _run_rdna4_compiled_softmax(args.repetitions)
            }
        elif args.workload == "rdna4-split-softmax":
            result = {"rdna4-split-softmax": _run_rdna4_split_softmax(args.repetitions)}
        elif args.workload == "rdna4-llm-topk":
            result = _run_rdna4_llm_topk(args.repetitions)
        else:
            raise AssertionError(f"unhandled PyTorch workload: {args.workload}")
    except Exception as exc:
        _write_oracle_result("fail", {"workload": args.workload, "reason": str(exc)})
        raise
    _write_oracle_result("pass", {"workload": args.workload, "result": result})
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.workload != "tdm-descriptor-add":
        # The large precompiled operator library can spend far longer tearing
        # down a software GPU process than executing this fully synchronized
        # workload. Finalize an active DBI hook so its validation verdict is
        # preserved, then avoid charging unrelated runtime shutdown to the
        # validation timeout.
        hook_path = os.environ.get("HSA_TOOLS_LIB")
        if hook_path:
            hook = ctypes.CDLL(hook_path)
            hook.OnUnload.argtypes = []
            hook.OnUnload.restype = None
            hook.OnUnload()
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
