#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Exercise the live-stack collector through its installed plain-C ABI."""

import ctypes
import os
import threading
import time
from pathlib import Path
from typing import Any, List

from utils.inject_roctx._backends import torch_cpp_loader

SKIP_RETURN_CODE = 77
REQUIRE_NATIVE_TEST_ENV = "ROCPROFCOMPUTE_REQUIRE_NATIVE_TORCH_TEST"
LIFECYCLE_THREAD_COUNT = 8
LIFECYCLE_ITERATION_COUNT = 16
LIFECYCLE_START_TIMEOUT_SECONDS = 10
LIFECYCLE_JOIN_TIMEOUT_SECONDS = 30


def skip_or_fail(reason: str) -> int:
    """Skip normally, or fail when a native-Torch job requires this test."""
    required = os.environ.get(REQUIRE_NATIVE_TEST_ENV, "").strip().lower()
    if required in ("1", "true", "yes", "on"):
        raise RuntimeError(reason)
    print(f"SKIP: {reason}")
    return SKIP_RETURN_CODE


def configure_interceptor(interceptor: ctypes.CDLL) -> None:
    """Declare the test interceptor's C ABI and clear its captured ranges."""
    interceptor.torch_trace_test_reset.argtypes = []
    interceptor.torch_trace_test_reset.restype = None
    interceptor.torch_trace_test_push_count.argtypes = []
    interceptor.torch_trace_test_push_count.restype = ctypes.c_size_t
    interceptor.torch_trace_test_pop_count.argtypes = []
    interceptor.torch_trace_test_pop_count.restype = ctypes.c_size_t
    interceptor.torch_trace_test_marker.argtypes = [
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    interceptor.torch_trace_test_marker.restype = ctypes.c_size_t
    interceptor.torch_trace_test_reset()


def read_markers(interceptor: ctypes.CDLL) -> List[str]:
    """Return all ROCTX range names captured by the interceptor."""
    markers = []
    for index in range(interceptor.torch_trace_test_push_count()):
        required = interceptor.torch_trace_test_marker(index, None, 0)
        if required == 0:
            raise RuntimeError(f"interceptor rejected marker index {index}")
        output = ctypes.create_string_buffer(required)
        copied = interceptor.torch_trace_test_marker(index, output, len(output))
        if copied != required:
            raise RuntimeError(
                f"marker {index} size changed from {required} to {copied}"
            )
        markers.append(output.value.decode("utf-8"))
    return markers


def exercise_live_stack(collector: torch_cpp_loader.TorchTraceCollector, torch) -> None:
    """Emit a user scope containing real forward and backward Torch work."""
    collector.push_user_scope("test.outer/%", "#1@test:7", "runtime")
    try:
        value = torch.ones((8, 8), dtype=torch.float32, requires_grad=True)
        result = (value * 2).sum()
        result.backward()
    finally:
        collector.pop_user_scope()


def validate_intercepted_markers(interceptor: ctypes.CDLL) -> None:
    """Validate live-stack marker shape and balance."""
    markers = read_markers(interceptor)
    pop_count = interceptor.torch_trace_test_pop_count()
    if not markers:
        raise RuntimeError("native collector emitted no ROCTX markers")
    if len(markers) != pop_count:
        raise RuntimeError(
            f"unbalanced ranges: pushes={len(markers)}, pops={pop_count}"
        )
    if "test.outer%2F%25:#1@test:7|runtime" not in markers:
        raise RuntimeError("plain-C user-scope push did not preserve encoding")
    if not any(
        marker.startswith("test.outer%2F%25/") and marker.endswith("|torch")
        for marker in markers
    ):
        raise RuntimeError("native Torch markers did not inherit the live user scope")


def validate_collector_stats(collector: torch_cpp_loader.TorchTraceCollector) -> None:
    """Validate native collector counters after forward and backward work."""
    stats = collector.dump_stats()
    if not stats["installed"]:
        raise RuntimeError("collector does not report itself installed")
    if stats["pushes"] != stats["pops"]:
        raise RuntimeError(f"collector counters are unbalanced: {stats}")
    if stats["user_scope_pushes"] != stats["user_scope_pops"]:
        raise RuntimeError(f"user-scope counters are unbalanced: {stats}")
    if stats["snapshots_saved"] == 0:
        raise RuntimeError("real Torch operations did not save any forward snapshots")
    if stats["snapshots_consumed"] == 0:
        raise RuntimeError("backward operations did not consume a forward snapshot")
    if stats["callback_errors"] != 0:
        raise RuntimeError(f"collector reported callback errors: {stats}")


def load_interceptor(path_value: str) -> ctypes.CDLL:
    """Load and configure the ROCTX interception library."""
    interceptor = ctypes.CDLL(
        str(Path(path_value).resolve()),
        mode=os.RTLD_GLOBAL | os.RTLD_NOW | os.RTLD_NODELETE,
    )
    configure_interceptor(interceptor)
    return interceptor


def load_collector(path_value: str) -> torch_cpp_loader.TorchTraceCollector:
    """Load the collector from the explicit CTest artifact path."""
    collector_path = Path(path_value).resolve()
    original_discover_collector = torch_cpp_loader._discover_collector_artifact
    torch_cpp_loader._discover_collector_artifact = lambda: collector_path
    try:
        return torch_cpp_loader.load()
    finally:
        torch_cpp_loader._discover_collector_artifact = original_discover_collector


def load_raw_collector_library(collector_path: Path) -> ctypes.CDLL:
    """Load and bind the collector for concurrent raw-C calls."""
    library = ctypes.CDLL(
        str(collector_path),
        mode=torch_cpp_loader._COLLECTOR_LOAD_MODE,
    )
    library.torch_trace_collector_install.argtypes = []
    library.torch_trace_collector_install.restype = ctypes.c_int32
    library.torch_trace_collector_uninstall.argtypes = []
    library.torch_trace_collector_uninstall.restype = ctypes.c_int32
    library.torch_trace_collector_is_installed.argtypes = []
    library.torch_trace_collector_is_installed.restype = ctypes.c_int32
    library.torch_trace_collector_push_user_scope.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
    ]
    library.torch_trace_collector_push_user_scope.restype = ctypes.c_int32
    library.torch_trace_collector_pop_user_scope.argtypes = []
    library.torch_trace_collector_pop_user_scope.restype = ctypes.c_int32
    return library


def validate_preinstall_rejection(
    collector: torch_cpp_loader.TorchTraceCollector,
) -> None:
    """Verify user scopes cannot be pushed before callback installation."""
    try:
        collector.push_user_scope("before-install", "n/a", "runtime")
    except RuntimeError:
        return
    raise RuntimeError("user-scope push succeeded before collector install")


def cycle_collector_lifecycle(
    library: ctypes.CDLL,
    start_barrier: threading.Barrier,
    failures: List[str],
    failures_lock: Any,
) -> None:
    """Run repeated install/uninstall pairs through the raw C interface."""
    try:
        start_barrier.wait(timeout=LIFECYCLE_START_TIMEOUT_SECONDS)
        for _ in range(LIFECYCLE_ITERATION_COUNT):
            if library.torch_trace_collector_install() != 0:
                raise RuntimeError("install failed")
            if library.torch_trace_collector_uninstall() != 0:
                raise RuntimeError("uninstall failed")
    except Exception as error:
        with failures_lock:
            failures.append(f"{threading.current_thread().name}: {error}")


def validate_scope_cleanup_after_uninstall(
    library: ctypes.CDLL,
    interceptor: ctypes.CDLL,
) -> None:
    """Verify an accepted user scope can be popped after collector teardown."""
    interceptor.torch_trace_test_reset()
    if library.torch_trace_collector_install() != 0:
        raise RuntimeError("collector install failed before scope cleanup test")
    if (
        library.torch_trace_collector_push_user_scope(
            b"lifecycle-scope",
            b"n/a",
            b"runtime",
        )
        != 0
    ):
        raise RuntimeError("user-scope push failed before collector uninstall")
    if library.torch_trace_collector_uninstall() != 0:
        raise RuntimeError("collector uninstall failed with an active user scope")
    if library.torch_trace_collector_pop_user_scope() != 0:
        raise RuntimeError("active user scope could not be popped after uninstall")

    push_count = interceptor.torch_trace_test_push_count()
    pop_count = interceptor.torch_trace_test_pop_count()
    if push_count != 1 or pop_count != 1:
        raise RuntimeError(
            "scope cleanup across uninstall was unbalanced: "
            f"pushes={push_count}, pops={pop_count}"
        )
    interceptor.torch_trace_test_reset()


def validate_concurrent_lifecycle(library: ctypes.CDLL) -> None:
    """Verify raw C callers can install and uninstall concurrently."""
    failures: List[str] = []
    failures_lock = threading.Lock()
    start_barrier = threading.Barrier(LIFECYCLE_THREAD_COUNT)
    threads = [
        threading.Thread(
            target=cycle_collector_lifecycle,
            args=(library, start_barrier, failures, failures_lock),
            name=f"collector-lifecycle-{index}",
            daemon=True,
        )
        for index in range(LIFECYCLE_THREAD_COUNT)
    ]
    for thread in threads:
        thread.start()

    join_deadline = time.monotonic() + LIFECYCLE_JOIN_TIMEOUT_SECONDS
    for thread in threads:
        remaining = max(0.0, join_deadline - time.monotonic())
        thread.join(timeout=remaining)

    live_threads = [thread.name for thread in threads if thread.is_alive()]
    if live_threads:
        raise RuntimeError(f"concurrent lifecycle threads hung: {live_threads}")
    if failures:
        raise RuntimeError(f"concurrent lifecycle failures: {failures}")
    if library.torch_trace_collector_is_installed() != 0:
        raise RuntimeError(
            "collector remained installed after concurrent lifecycle test"
        )


def exercise_installed_collector(
    collector: torch_cpp_loader.TorchTraceCollector,
    interceptor: ctypes.CDLL,
    torch,
) -> None:
    """Exercise the installed collector and verify deterministic teardown."""
    collector.install()
    try:
        if not collector.is_installed():
            raise RuntimeError("collector install did not become visible")
        exercise_live_stack(collector, torch)
        validate_intercepted_markers(interceptor)
        validate_collector_stats(collector)
    finally:
        collector.uninstall()

    if collector.is_installed():
        raise RuntimeError("collector remained installed after uninstall")


def run() -> int:
    """Run the native smoke test, skipping when its runtime is unavailable."""
    collector_value = os.environ.get("ROCPROFCOMPUTE_TEST_TORCH_COLLECTOR")
    interceptor_value = os.environ.get("ROCPROFCOMPUTE_TEST_ROCTX_INTERCEPT")
    if collector_value is None or interceptor_value is None:
        return skip_or_fail("native collector paths are supplied by CTest")

    try:
        import torch
    except ImportError as error:
        return skip_or_fail(f"PyTorch is unavailable ({error})")

    interceptor = load_interceptor(interceptor_value)
    try:
        collector = load_collector(collector_value)
    except torch_cpp_loader.CollectorUnavailableError as error:
        return skip_or_fail(str(error))

    raw_library = load_raw_collector_library(Path(collector_value).resolve())
    validate_concurrent_lifecycle(raw_library)
    validate_scope_cleanup_after_uninstall(raw_library, interceptor)
    validate_preinstall_rejection(collector)
    exercise_installed_collector(collector, interceptor, torch)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
