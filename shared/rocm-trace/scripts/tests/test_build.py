# Copyright Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT

"""Compile the generated probes, run them over stub entries, and record them.

Optional, by what the machine has:
  compile   TraceLogging headers: ROCM_TRACE_GEN_TRACELOGGING_INCLUDE, or installed
  run       also the LTTng flavour's helper sources (ROCM_TRACE_GEN_TRACELOGGING_SOURCE,
            the LTTng/ directory of microsoft/tracelogging) and liblttng-ust (pkg-config)
  record    also lttng, lttng-sessiond and babeltrace2
"""

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from conftest import DATA, EXPECTED, ROCM_TRACE_INCLUDE, tracelogging_include

CXX = os.environ.get("CXX", "c++")
CC = os.environ.get("CC", "cc")
WARNINGS = ["-Wall", "-Wextra", "-Werror"]


def _tracelogging_include():
    inc = tracelogging_include()
    if inc is None:
        pytest.skip(
            "TraceLogging headers not found; set ROCM_TRACE_GEN_TRACELOGGING_INCLUDE"
        )
    if shutil.which(CXX) is None:
        pytest.skip(f"no C++ compiler ({CXX})")
    return inc


def _includes(tl_include):
    return [f"-I{EXPECTED}", f"-I{DATA}", f"-I{ROCM_TRACE_INCLUDE}", f"-I{tl_include}"]


def _run(cmd, **kwargs):
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    assert (
        result.returncode == 0
    ), f"{' '.join(map(str, cmd))}\n{result.stdout}\n{result.stderr}"
    return result


def test_generated_code_compiles():
    tl = _tracelogging_include()
    for source in (EXPECTED / "fake_probes.cpp", DATA / "driver.cpp"):
        _run(
            [CXX, "-std=c++17", "-fsyntax-only", *WARNINGS, *_includes(tl), str(source)]
        )


@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    tl = _tracelogging_include()
    source = os.environ.get("ROCM_TRACE_GEN_TRACELOGGING_SOURCE")
    if not source or not (Path(source) / "src" / "LttngHelpers.c").exists():
        pytest.skip(
            "set ROCM_TRACE_GEN_TRACELOGGING_SOURCE to tracelogging's LTTng/ directory"
        )
    pkg = subprocess.run(
        ["pkg-config", "--libs", "lttng-ust"],
        capture_output=True,
        text=True,
        check=False,
    )
    if pkg.returncode != 0:
        pytest.skip("liblttng-ust not found by pkg-config")

    out = tmp_path_factory.mktemp("driver")
    objects = []
    for c in sorted((Path(source) / "src").glob("Lttng*.c")):
        obj = out / (c.stem + ".o")
        _run(
            [CC, "-std=gnu11", "-O1", "-fPIC", f"-I{tl}", "-c", str(c), "-o", str(obj)]
        )
        objects.append(str(obj))
    exe = out / "driver"
    _run(
        [
            CXX,
            "-std=c++17",
            "-O1",
            *WARNINGS,
            *_includes(tl),
            str(EXPECTED / "fake_probes.cpp"),
            str(DATA / "driver.cpp"),
            *objects,
            "-o",
            str(exe),
            *pkg.stdout.split(),
            "-llttng-ust-tracepoint",
            "-ldl",
        ]
    )
    return exe


def test_probes_pass_calls_through(driver):
    result = _run([str(driver)])
    assert "all checks passed" in result.stdout


@pytest.fixture
def lttng_home(tmp_path):
    tools = ["lttng", "lttng-sessiond", "babeltrace2"]
    if any(shutil.which(t) is None for t in tools):
        pytest.skip("lttng, lttng-sessiond or babeltrace2 not installed")
    env = dict(os.environ, LTTNG_HOME=str(tmp_path))
    sessiond = subprocess.Popen(
        ["lttng-sessiond", "--no-kernel"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            if (
                subprocess.run(
                    ["lttng", "list"], env=env, capture_output=True
                ).returncode
                == 0
            ):
                break
            time.sleep(0.1)
        else:
            pytest.fail("lttng-sessiond did not start")
        yield env
    finally:
        sessiond.terminate()
        sessiond.wait(timeout=30)


def test_session_records_enter_and_exit(driver, lttng_home, tmp_path):
    env = lttng_home
    trace = tmp_path / "trace"
    _run(["lttng", "create", "rtg", f"--output={trace}"], env=env)
    _run(["lttng", "enable-event", "--userspace", "fake_provider:*"], env=env)
    _run(["lttng", "start"], env=env)
    _run([str(driver)], env=env)
    _run(["lttng", "stop"], env=env)
    _run(["lttng", "destroy", "rtg"], env=env)
    text = _run(["babeltrace2", str(trace)]).stdout

    def event(name):
        lines = [
            line
            for line in text.splitlines()
            if re.search(rf"fake_provider:{name}[;:]", line)
        ]
        assert lines, (
            f"no {name} event in:\n{text[:2000]}\nIf the trace holds only metadata, the ring "
            "buffers may not fit in /dev/shm (in Docker, raise --shm-size)."
        )
        return lines

    alloc_enter = event("fakeAlloc_enter")
    assert "size = 16" in alloc_enter[0] and ";k0;k6;" in alloc_enter[0]
    alloc_exit = event("fakeAlloc_exit")
    # Outputs under their own names, only after success (RC-API-1, RC-API-2).
    assert "result = 0" in alloc_exit[0] and "ptr = 0x1234" in alloc_exit[0]
    assert "result = 1" in alloc_exit[1] and "ptr = 0x0" in alloc_exit[1]
    assert all('exception = ( "false"' in line for line in alloc_exit)
    assert 'name = "fake"' in event("fakeName_exit")[0]

    # const char* inputs: capped at 256 bytes, except the kernel name.
    load = event("fakeLoad_enter")[0]
    assert "_path_length = 256" in load and 'path = "/' + "p" * 255 + '"' in load
    assert 'kname = "' + "k" * 300 + '"' in event("fakeGetFunction_enter")[0]
    assert "function = 0x77" in event("fakeGetFunction_exit")[0]

    # Every component of an aggregate; unions as bytes.
    launch = event("fakeLaunchEx_enter")[0]
    assert "config_grid_x = 2" in launch and "config_extra_1_z = 9" in launch
    assert "config_slots = [ [0] = 0xABC" in launch and "config_flags = 3" in launch
    assert "config_weights_0 = 0," in launch and "config_weights_1 = 0.5," in launch
    assert "config_named_b = 12" in launch
    assert "handle_reserved = [ [0] = 104," in event("fakeOpen_enter")[0]
    bits = event("fakeBits_enter")[0]
    assert "bits = [ [0] = 4, [1] = 3, [2] = 2, [3] = 1 ]" in bits
    assert "value_kind = 1" in bits and "value_union = [ [0] = 7," in bits

    # Contract names.
    assert "arg_1 = 1" in event("fakeUnnamed_enter")[0]
    assert 'arg_2 = "unnamed"' in event("fakeUnnamed_enter")[0]
    reserved = event("fakeReserved_enter")[0]
    assert "arg_result = 1" in reserved and "arg_timestamp = 2" in reserved
    assert "arg_exception = 5" in event("fakeReserved_exit")[0]
    assert "public_name = 1" in event("fakeRenamed_enter")[0]

    # Enums keep their signedness.
    wide = event("fakeWide_enter")[0]
    assert "value = 1099511627776" in wide and "flags = 2147483648" in wide

    # An exception: exit written, marked, result zero.
    thrown = event("fakeThrow_exit")[0]
    assert 'exception = ( "true"' in thrown and "result = 0" in thrown

    assert "dim_x = 7" in event("fakeGetDim_exit")[0]
    assert "result_x = 1" in event("fakeMakeDim_exit")[0]
    assert 'result = "invalid value"' in event("fakeErrorString_exit")[0]
    assert "count = 42" in event("fakeCount_exit")[0]
    assert not re.search(r"fake_provider:fakeSkipped", text)

    # Capture levels: scalars drops strings and outputs, none keeps only results.
    load_events = event("fakeLoad_enter")
    assert 'path = ""' in load_events[1] and "stream = 0x20" in load_events[1]
    assert "size = 32" in alloc_enter[2] and "ptr = 0x0" in alloc_exit[2]
    assert "result = 0" in alloc_exit[2]
    assert "size = 0" in alloc_enter[3] and "ptr = 0x0," in alloc_enter[3]
    assert "result = 0" in alloc_exit[3]
    assert len(load_events) == 2

    enters = len(re.findall(r"fake_provider:\w+_enter", text))
    exits = len(re.findall(r"fake_provider:\w+_exit", text))
    assert enters == exits > 0
