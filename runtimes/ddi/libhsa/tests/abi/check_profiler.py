#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Exercise native HSA profiler snapshots, generation admission and retirement."""

import argparse
from collections.abc import Callable
import ctypes
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time


def exercise_timeout(
    hsa: ctypes.CDLL,
    tool: ctypes.CDLL,
    query: Callable[[], tuple[int, int]],
    stage: str,
) -> int:
    """Keep a known operation live until shutdown returns its drain timeout."""
    update_result: list[int] = []
    updater = None
    if stage == "ordinary":
        updater = threading.Thread(
            target=lambda: update_result.append(tool.test_update(2))
        )
        updater.start()
        tool.test_wait_paused()
    else:
        tool.test_hold_cleanup()

    shutdown_result: list[int] = []
    shutdown = threading.Thread(
        target=lambda: shutdown_result.append(hsa.hsa_shut_down())
    )
    shutdown.start()
    if stage == "cleanup":
        tool.test_wait_paused()
    try:
        # The operation stays parked until the runtime's real five-second
        # deadline expires. No scheduling delay determines this ordering.
        shutdown.join(timeout=10)
        assert not shutdown.is_alive(), "shutdown did not bound its drain"
        assert shutdown_result == [46], shutdown_result  # RESOURCE_BUSY
        assert query()[0] != 0, "timeout reopened public admission"
        assert hsa.hsa_init() == 0x1025, "pending retirement admitted a generation"
        assert tool.test_count(4) == int(stage == "cleanup"), "wrong prepare stage"
        assert tool.test_count(2) == 0, "profiler retired before its work drained"
        assert tool.test_count(7) == 1 and tool.test_count(8) == 0
    finally:
        tool.test_release()
        if updater is not None:
            updater.join(timeout=5)
        else:
            tool.test_wait_cleanup_finished()
        shutdown.join(timeout=5)

    if updater is not None:
        assert not updater.is_alive()
        assert update_result == [0], update_result
        assert tool.test_count(3) == 0, "admitted updater lost its generation"
    else:
        assert tool.test_count(5) == 0, "delegated cleanup did not complete"
        assert tool.test_count(6) == 0, "cleanup descendant lost its generation"
    assert query()[0] != 0, "drained generation reopened public admission"
    assert hsa.hsa_shut_down() == 0, "shutdown retry did not resume retirement"
    assert tool.test_count(4) == 1, "retry repeated profiler preparation"
    assert tool.test_count(2) == 1, "profiler retirement was not exactly once"
    assert tool.test_count(8) == 1, "legacy unload was not exactly once"
    assert tool.test_count(9) == 0, "legacy unload lost cleanup admission"
    assert hsa.hsa_init() == 0, "completed retirement prevented reinitialization"
    assert query() == (0, 111)
    assert hsa.hsa_shut_down() == 0
    assert tool.test_count(4) == 2 and tool.test_count(2) == 2
    assert tool.test_count(7) == 2 and tool.test_count(8) == 2
    assert tool.test_count(9) == 0, "legacy unload lost cleanup admission after reinit"
    print(f"profiler {stage} drain timeout: retry and reinitialization passed")
    return 0


def exercise(
    library: Path, fixture: Path, failure: bool, timeout_stage: str | None
) -> int:
    tool = ctypes.CDLL(str(fixture))
    hsa = ctypes.CDLL(str(library))
    hsa.hsa_system_get_info.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
    hsa.hsa_system_get_info.restype = ctypes.c_uint32
    for name in ("hsa_init", "hsa_shut_down"):
        getattr(hsa, name).restype = ctypes.c_uint32
    tool.test_update.argtypes = [ctypes.c_uint32]
    tool.test_update.restype = ctypes.c_uint32
    tool.test_count.argtypes = [ctypes.c_uint32]
    tool.test_count.restype = ctypes.c_uint32
    initialized = hsa.hsa_init()
    if initialized:
        print(f"SKIP: native CPU-only HSA initialization returned {initialized:#x}")
        return 77

    def query() -> tuple[int, int]:
        value = ctypes.c_uint16()
        status = hsa.hsa_system_get_info(0, ctypes.byref(value))
        return status, value.value

    assert query() == (0, 111), query()
    if timeout_stage is not None:
        return exercise_timeout(hsa, tool, query, timeout_stage)
    if failure:
        tool.test_fail_quiesce()
        assert hsa.hsa_shut_down() != 0
        assert hsa.hsa_init() != 0, "failed quiesce must quarantine the generation"
        return 0

    errors: list[object] = []
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            result = query()
            if result not in ((0, 111), (0, 222)):
                errors.append(result)
                stop.set()

    readers = [threading.Thread(target=reader) for _ in range(4)]
    for thread in readers:
        thread.start()
    for index in range(500):
        assert tool.test_update(index % 2) == 0
    stop.set()
    for thread in readers:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert not errors, errors

    before = query()
    update_result: list[int] = []
    updater = threading.Thread(target=lambda: update_result.append(tool.test_update(2)))
    updater.start()
    tool.test_wait_paused()
    assert query() == before, "private staging must not change live dispatch"
    shutdown_result: list[int] = []
    shutdown = threading.Thread(target=lambda: shutdown_result.append(hsa.hsa_shut_down()))
    shutdown.start()
    deadline = time.monotonic() + 5
    while query()[0] == 0 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert query()[0] != 0, "shutdown did not close public admission"
    assert shutdown.is_alive(), "shutdown did not wait for the admitted updater"
    tool.test_release()
    updater.join(timeout=5)
    shutdown.join(timeout=5)
    assert not updater.is_alive() and not shutdown.is_alive()
    assert update_result == [0] and shutdown_result == [0]
    assert tool.test_count(3) == 0, "updater lost its nested generation lease"
    assert tool.test_update(0) == 9, "closed generation admitted a late update"
    assert tool.test_count(2) == 1

    # More generations than the old registrar's per-library instance limit.
    for generation in range(1, 24):
        assert hsa.hsa_init() == 0
        assert query() == (0, 111)
        assert tool.test_update(1) == 0
        assert query() == (0, 222)
        assert hsa.hsa_shut_down() == 0
        assert tool.test_update(0) == 9
        assert tool.test_count(0) == 1
        assert tool.test_count(1) == generation
        assert tool.test_count(2) == generation + 1
    print("profiler publication: concurrent reads, staged update, shutdown fence, 24 generations passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--failure", action="store_true")
    parser.add_argument("--timeout", choices=("ordinary", "cleanup"))
    args = parser.parse_args()
    if args.fixture:
        return exercise(
            args.library.resolve(), args.fixture.resolve(), args.failure, args.timeout
        )
    source = Path(__file__).resolve().parent
    headers = source.parents[3] / "api-headers" / "include"
    with tempfile.TemporaryDirectory(prefix="hsa-profiler-") as temporary:
        fixture = Path(temporary) / "librocprofiler-register.so.0"
        subprocess.run([args.cxx, "-std=c++17", "-shared", "-fPIC", "-pthread", "-I", str(headers),
                        str(source / "profiler_snapshot_tool.cpp"), "-o", str(fixture)], check=True)
        env = os.environ.copy()
        env["ROCR_VISIBLE_DEVICES"] = ""
        env["LD_LIBRARY_PATH"] = temporary + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        env.pop("HSA_TOOLS_LIB", None)
        command = [sys.executable, str(Path(__file__).resolve()), str(args.library.resolve()),
                   "--fixture", str(fixture)]
        scenarios = (
            [],
            ["--failure"],
            ["--timeout", "ordinary"],
            ["--timeout", "cleanup"],
        )
        for suffix in scenarios:
            child_env = env.copy()
            if "--timeout" in suffix:
                child_env["HSA_TOOLS_LIB"] = str(fixture)
                child_env["HSA_TOOLS_ROCPROFILER_V1_TOOLS"] = "1"
            result = subprocess.run(
                command + suffix, env=child_env, timeout=30, check=False
            )
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
