#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Exercise native HSA profiler snapshots, generation admission and retirement."""

import argparse
import ctypes
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time


def exercise(library: Path, fixture: Path, failure: bool) -> int:
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
    args = parser.parse_args()
    if args.fixture:
        return exercise(args.library.resolve(), args.fixture.resolve(), args.failure)
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
        for suffix in ([], ["--failure"]):
            result = subprocess.run(command + suffix, env=env, timeout=30, check=False)
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
