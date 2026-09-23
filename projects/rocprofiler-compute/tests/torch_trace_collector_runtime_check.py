#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Exercise the native collector in an isolated allowlisted Torch process."""

import ctypes
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List

from utils.inject_roctx._backends import torch_trace_collector

SKIP_RETURN_CODE = 77
REQUIRE_NATIVE_TEST_ENV = "ROCPROFCOMPUTE_REQUIRE_NATIVE_TORCH_TEST"
RAW_ARGUMENT_LIMIT = 512
TRUNCATION_SUFFIX = "...)"
TORCH_BACKEND_SUFFIX = "|torch"
MARKER_PATTERN = re.compile(
    r"^.+:n/a\|seqNr=(?P<seq_nr>n/a|[0-9]+)\|tid=(?P<tid>[0-9]+)"
    r"\|ftid=(?P<forward_tid>[0-9]+)\|ltid=(?P<launcher_tid>n/a|[0-9]+)"
    r"\|scope=(?P<scope>[A-Z_]+)\|args=.*\|torch$"
)


def skip_or_fail(reason: str) -> int:
    """Skip normally, or fail when a dedicated native-Torch job requires it."""
    required = os.environ.get(REQUIRE_NATIVE_TEST_ENV, "").strip().lower()
    if required in ("1", "true", "yes", "on"):
        raise RuntimeError(reason)
    print(f"SKIP: {reason}")
    return SKIP_RETURN_CODE


def configure_interceptor(interceptor: ctypes.CDLL) -> None:
    """Declare the test interceptor's C ABI and clear prior state."""
    interceptor.torch_trace_test_reset.argtypes = []
    interceptor.torch_trace_test_reset.restype = None
    interceptor.torch_trace_test_push_count.argtypes = []
    interceptor.torch_trace_test_push_count.restype = ctypes.c_size_t
    interceptor.torch_trace_test_pop_count.argtypes = []
    interceptor.torch_trace_test_pop_count.restype = ctypes.c_size_t
    interceptor.torch_trace_test_fail_next_push.argtypes = []
    interceptor.torch_trace_test_fail_next_push.restype = None
    interceptor.torch_trace_test_marker.argtypes = [
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    interceptor.torch_trace_test_marker.restype = ctypes.c_size_t
    interceptor.torch_trace_test_reset()


def read_markers(interceptor: ctypes.CDLL) -> List[str]:
    """Return all markers captured by the test ROCTX implementation."""
    markers = []
    for index in range(interceptor.torch_trace_test_push_count()):
        required = interceptor.torch_trace_test_marker(index, None, 0)
        if required == 0:
            raise RuntimeError(f"interceptor rejected marker index {index}")

        if required > 1:
            undersized = ctypes.create_string_buffer(required - 1)
            copied = interceptor.torch_trace_test_marker(
                index,
                undersized,
                len(undersized),
            )
            if copied != 0:
                raise RuntimeError(
                    f"interceptor accepted a short marker buffer {index}"
                )

        buffer = ctypes.create_string_buffer(required)
        copied = interceptor.torch_trace_test_marker(index, buffer, len(buffer))
        if copied != required:
            raise RuntimeError(
                f"marker {index} size changed from {required} to {copied}"
            )
        markers.append(buffer.value.decode("utf-8"))
    return markers


def install_native_collector(collector_value: str) -> None:
    """Verify the pre-install gate, then install the native collector."""
    collector_path = Path(collector_value).resolve()
    if torch_trace_collector._torch_cpu_library is None:
        torch_trace_collector._torch_cpu_library = (
            torch_trace_collector._promote_torch_cpu()
        )
    uninstalled_collector = ctypes.CDLL(
        str(collector_path),
        mode=torch_trace_collector._COLLECTOR_LOAD_MODE,
    )
    if not torch_trace_collector._validate_collector_interface(uninstalled_collector):
        raise RuntimeError("native collector interface unexpectedly mismatched")
    if uninstalled_collector.torch_trace_collector_push_launcher_tid(1) == 0:
        raise RuntimeError("launcher-thread push succeeded before collector install")
    if uninstalled_collector.torch_trace_collector_pop_launcher_tid() == 0:
        raise RuntimeError("launcher-thread pop succeeded before collector install")

    torch_trace_collector._find_collector = lambda: collector_path
    if not torch_trace_collector.install():
        raise RuntimeError("native collector unexpectedly fell back")


def exercise_nested_launcher_tid_stack(torch: Any) -> None:
    """Verify native launcher IDs nest, restore, and reject underflow."""
    collector = torch_trace_collector._collector_library
    if collector is None:
        raise RuntimeError("native collector handle is unavailable")
    push = collector.torch_trace_collector_push_launcher_tid
    pop = collector.torch_trace_collector_pop_launcher_tid

    if push(101) != 0:
        raise RuntimeError("outer launcher-thread push failed")
    try:
        with torch.autograd.profiler.record_function("launcher_outer_before"):
            pass
        if push(202) != 0:
            raise RuntimeError("inner launcher-thread push failed")
        try:
            with torch.autograd.profiler.record_function("launcher_inner"):
                pass
        finally:
            if pop() != 0:
                raise RuntimeError("inner launcher-thread pop failed")
        with torch.autograd.profiler.record_function("launcher_outer_after"):
            pass
    finally:
        if pop() != 0:
            raise RuntimeError("outer launcher-thread pop failed")
    if pop() == 0:
        raise RuntimeError("launcher-thread stack underflow unexpectedly succeeded")


def exercise_marker_cases(torch: Any) -> int:
    """Emit representative argument, TensorList, autograd, and long-name ranges."""
    first = torch.zeros((2, 3), dtype=torch.float32)
    second = torch.ones((2, 3), dtype=torch.float32)
    torch.add(first, second, alpha=2)
    torch.cat([first, second], dim=0)

    high_rank = torch.zeros((1,) * 64, dtype=torch.float32)
    torch.cat([high_rank] * 8, dim=0)

    launcher_tid = threading.get_native_id()
    if not torch_trace_collector.push_launcher_tid():
        raise RuntimeError("native launcher-thread push failed")
    try:
        gradient_input = torch.ones((2, 3), dtype=torch.float32, requires_grad=True)
        (gradient_input * gradient_input).sum().backward()
    finally:
        if not torch_trace_collector.pop_launcher_tid():
            raise RuntimeError("native launcher-thread pop failed")

    long_name = ("/" * 3000) + "%tail"
    with torch.autograd.profiler.record_function(long_name):
        pass
    return launcher_tid


def exercise_concurrent_launcher_tids(torch: Any) -> Dict[str, int]:
    """Run two overlapping autograd launches and return marker-to-thread IDs."""
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    launcher_tids: Dict[str, int] = {}
    failures: List[BaseException] = []

    def run_backward(index: int) -> None:
        pushed = False
        try:
            launcher_tid = threading.get_native_id()
            if not torch_trace_collector.push_launcher_tid():
                raise RuntimeError(f"launcher-thread push failed for worker {index}")
            pushed = True
            barrier.wait(timeout=10)
            marker_name = f"launcher_thread_{index}"
            with torch.autograd.profiler.record_function(marker_name):
                value = torch.ones((2, 3), dtype=torch.float32, requires_grad=True)
                (value * value).sum().backward()
            with lock:
                launcher_tids[marker_name] = launcher_tid
        except BaseException as error:
            with lock:
                failures.append(error)
        finally:
            if pushed and not torch_trace_collector.pop_launcher_tid():
                with lock:
                    failures.append(
                        RuntimeError(f"launcher-thread pop failed for worker {index}")
                    )

    threads = [
        threading.Thread(target=run_backward, args=(index,)) for index in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if failures:
        raise RuntimeError(f"concurrent launcher-thread probe failed: {failures!r}")
    if len(set(launcher_tids.values())) != len(threads):
        raise RuntimeError("concurrent launcher threads did not have distinct IDs")
    return launcher_tids


def verify_failed_push_does_not_pop(torch: Any, interceptor: ctypes.CDLL) -> None:
    """Exercise the collector's negative ROCTX-push branch."""
    pushes_before = interceptor.torch_trace_test_push_count()
    pops_before = interceptor.torch_trace_test_pop_count()
    interceptor.torch_trace_test_fail_next_push()
    with torch.autograd.profiler.record_function("expected push failure"):
        pass
    if interceptor.torch_trace_test_push_count() != pushes_before:
        raise RuntimeError("failed ROCTX push unexpectedly recorded a marker")
    if interceptor.torch_trace_test_pop_count() != pops_before:
        raise RuntimeError("failed ROCTX push unexpectedly emitted a pop")


def validate_argument_markers(markers: List[str]) -> None:
    """Validate named Tensor, TensorList, and truncation output."""
    add_args = (
        "|ltid=n/a|scope=FUNCTION|args=(self=float32[2x3], other=float32[2x3], "
        "alpha=Int)|torch"
    )
    if not any(
        marker.startswith("aten::add:n/a|seqNr=") and add_args in marker
        for marker in markers
    ):
        raise RuntimeError("native aten::add marker did not preserve arguments")

    cat_args = (
        "|ltid=n/a|scope=FUNCTION|args=(tensors=[float32[2x3], float32[2x3]], "
        "dim=Int)|torch"
    )
    if not any(
        marker.startswith("aten::cat:n/a|seqNr=") and cat_args in marker
        for marker in markers
    ):
        raise RuntimeError("native aten::cat marker did not preserve TensorList")

    truncated_arguments = []
    for marker in markers:
        if not marker.startswith("aten::cat:n/a|seqNr=") or "|args=" not in marker:
            continue
        arguments = marker.split("|args=", maxsplit=1)[1]
        if arguments.endswith(TORCH_BACKEND_SUFFIX):
            truncated_arguments.append(arguments[: -len(TORCH_BACKEND_SUFFIX)])
    expected_length = RAW_ARGUMENT_LIMIT + len(TRUNCATION_SUFFIX)
    if not any(
        len(arguments) == expected_length
        and arguments.startswith("(")
        and arguments.endswith(TRUNCATION_SUFFIX)
        for arguments in truncated_arguments
    ):
        raise RuntimeError(
            "native argument truncation did not preserve its 512-byte limit"
        )


def validate_launcher_stack_markers(markers: List[str]) -> None:
    """Validate nested launcher IDs and restoration in emitted markers."""
    expected = {
        "launcher_outer_before": "101",
        "launcher_inner": "202",
        "launcher_outer_after": "101",
    }
    for name, launcher_tid in expected.items():
        if not any(
            marker.startswith(f"{name}:n/a|seqNr=")
            and f"|ltid={launcher_tid}|scope=USER_SCOPE|" in marker
            for marker in markers
        ):
            raise RuntimeError(f"native launcher-thread stack lost {name}")


def validate_concurrent_launcher_markers(
    markers: List[str], launcher_tids: Dict[str, int]
) -> None:
    """Validate that each launcher marker retains its own native thread ID."""
    for marker_name, launcher_tid in launcher_tids.items():
        if not any(
            marker.startswith(f"{marker_name}:n/a|seqNr=")
            and f"|ltid={launcher_tid}|scope=USER_SCOPE|" in marker
            for marker in markers
        ):
            raise RuntimeError(
                f"native launcher marker {marker_name} did not retain thread ID "
                f"{launcher_tid}"
            )


def parse_and_validate_markers(markers: List[str]) -> List[re.Match]:
    """Parse markers and validate their field order and current thread IDs."""
    parsed_markers = []
    for marker in markers:
        match = MARKER_PATTERN.fullmatch(marker)
        if match is None:
            raise RuntimeError("native marker field order or encoding is invalid")
        if int(match.group("tid")) == 0:
            raise RuntimeError("native marker has an invalid current thread id")
        parsed_markers.append(match)
    return parsed_markers


def validate_backward_linkage(
    parsed_markers: List[re.Match], launcher_tids: List[int]
) -> None:
    """Validate backward sequence, forward-thread, and launcher-thread links."""
    forward_correlations = {
        (match.group("seq_nr"), match.group("tid"))
        for match in parsed_markers
        if match.group("seq_nr") != "n/a" and match.group("scope") == "FUNCTION"
    }
    linked_launcher_tids = {
        int(match.group("launcher_tid"))
        for match in parsed_markers
        if match.group("scope") == "BACKWARD_FUNCTION"
        and match.group("seq_nr") != "n/a"
        and match.group("launcher_tid") != "n/a"
        and (match.group("seq_nr"), match.group("forward_tid")) in forward_correlations
    }
    missing_launcher_tids = set(launcher_tids) - linked_launcher_tids
    if missing_launcher_tids:
        raise RuntimeError(
            "native autograd markers lack sequence/thread linkage for launcher IDs "
            f"{sorted(missing_launcher_tids)}"
        )


def validate_long_name_marker(markers: List[str]) -> None:
    """Validate exact escaping on the heap-fallback marker path."""
    encoded_long_name = ("%2F" * 3000) + "%25tail:n/a|seqNr="
    if not any(marker.startswith(encoded_long_name) for marker in markers):
        raise RuntimeError("long escaped marker did not use the heap fallback")


def validate_results(
    markers: List[str],
    pop_count: int,
    launcher_tids: List[int],
    concurrent_launcher_tids: Dict[str, int],
) -> None:
    """Validate the complete marker set emitted by the native probe."""
    if len(markers) != pop_count:
        raise RuntimeError(
            f"unbalanced ranges: pushes={len(markers)}, pops={pop_count}"
        )
    validate_argument_markers(markers)
    validate_launcher_stack_markers(markers)
    validate_concurrent_launcher_markers(markers, concurrent_launcher_tids)
    parsed_markers = parse_and_validate_markers(markers)
    validate_backward_linkage(parsed_markers, launcher_tids)
    validate_long_name_marker(markers)


def run() -> int:
    """Run the native probe, returning CTest's skip code when unavailable."""
    collector_value = os.environ.get("ROCPROFCOMPUTE_TEST_TORCH_COLLECTOR")
    interceptor_value = os.environ.get("ROCPROFCOMPUTE_TEST_ROCTX_INTERCEPT")
    if collector_value is None or interceptor_value is None:
        return skip_or_fail("native collector paths are supplied by CTest")

    interceptor = ctypes.CDLL(
        str(Path(interceptor_value).resolve()),
        mode=os.RTLD_GLOBAL | os.RTLD_NOW | os.RTLD_NODELETE,
    )
    configure_interceptor(interceptor)

    try:
        import torch
    except ImportError as error:
        return skip_or_fail(f"PyTorch is unavailable ({error})")

    identity = torch_trace_collector._workload_torch_identity()
    if identity not in torch_trace_collector._VALIDATED_TORCH_BUILDS:
        return skip_or_fail(f"PyTorch build is not allowlisted: {identity!r}")

    install_native_collector(collector_value)
    exercise_nested_launcher_tid_stack(torch)
    launcher_tid = exercise_marker_cases(torch)
    concurrent_launcher_tids = exercise_concurrent_launcher_tids(torch)
    verify_failed_push_does_not_pop(torch, interceptor)
    markers = read_markers(interceptor)
    validate_results(
        markers,
        interceptor.torch_trace_test_pop_count(),
        [launcher_tid, *concurrent_launcher_tids.values()],
        concurrent_launcher_tids,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
