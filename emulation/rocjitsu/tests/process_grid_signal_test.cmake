# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(NOT DEFINED TEST_EXECUTABLE)
    message(FATAL_ERROR "TEST_EXECUTABLE is required")
endif()
if(NOT DEFINED TEST_RUNTIME_DIR)
    message(FATAL_ERROR "TEST_RUNTIME_DIR is required")
endif()
if(NOT DEFINED TEST_OUTPUT_DIR)
    message(FATAL_ERROR "TEST_OUTPUT_DIR is required")
endif()
if(NOT DEFINED PYTHON_EXECUTABLE)
    message(FATAL_ERROR "PYTHON_EXECUTABLE is required")
endif()

file(REMOVE_RECURSE "${TEST_OUTPUT_DIR}")
file(MAKE_DIRECTORY "${TEST_OUTPUT_DIR}")

set(_rank_script "${TEST_OUTPUT_DIR}/rank.sh")
file(
    WRITE "${_rank_script}"
    [=[
#!/bin/sh
sleep 60 &
worker=$!
printf "%s\n" "$worker" > "$RJ_GRID_OUT/worker-$RANK"
trap 'printf "term\n" > "$RJ_GRID_OUT/term-$RANK"; exit 0' TERM INT
printf "%s\n" "$$" > "$RJ_GRID_OUT/pid-$RANK"
while :; do
    sleep 1
done
]=]
)

set(_driver_script "${TEST_OUTPUT_DIR}/driver.sh")
file(
    WRITE "${_driver_script}"
    [=[
#!/bin/sh
"$RJ_EXECUTABLE" --profile mi350x --num-nodes 1 --nproc-per-node 2 --env "RJ_GRID_OUT=$RJ_GRID_OUT" -- /bin/sh "$RJ_RANK_SCRIPT" > "$RJ_GRID_OUT/launcher.out" 2> "$RJ_GRID_OUT/launcher.err" &
launcher=$!

i=0
while [ "$i" -lt 100 ]; do
    if [ -s "$RJ_GRID_OUT/pid-0" ] && [ -s "$RJ_GRID_OUT/pid-1" ] && [ -s "$RJ_GRID_OUT/worker-0" ] && [ -s "$RJ_GRID_OUT/worker-1" ]; then
        break
    fi
    i=$((i + 1))
    sleep 0.1
done

if [ ! -s "$RJ_GRID_OUT/pid-0" ] || [ ! -s "$RJ_GRID_OUT/pid-1" ]; then
    kill -TERM "$launcher" 2>/dev/null
    wait "$launcher" 2>/dev/null
    echo "ranks did not start" >&2
    cat "$RJ_GRID_OUT/launcher.err" >&2
    exit 10
fi

pid0=$(cat "$RJ_GRID_OUT/pid-0")
pid1=$(cat "$RJ_GRID_OUT/pid-1")
worker0=$(cat "$RJ_GRID_OUT/worker-0")
worker1=$(cat "$RJ_GRID_OUT/worker-1")
kill -TERM "$launcher"
wait "$launcher"
status=$?
if [ "$status" -ne 143 ]; then
    echo "launcher exited with $status, expected 143" >&2
    cat "$RJ_GRID_OUT/launcher.err" >&2
    exit 11
fi

i=0
while [ "$i" -lt 50 ]; do
    alive=0
    kill -0 "$pid0" 2>/dev/null && alive=1
    kill -0 "$pid1" 2>/dev/null && alive=1
    kill -0 "$worker0" 2>/dev/null && alive=1
    kill -0 "$worker1" 2>/dev/null && alive=1
    [ "$alive" -eq 0 ] && break
    i=$((i + 1))
    sleep 0.1
done

if kill -0 "$pid0" 2>/dev/null || kill -0 "$pid1" 2>/dev/null || kill -0 "$worker0" 2>/dev/null || kill -0 "$worker1" 2>/dev/null; then
    echo "rank or worker process survived launcher SIGTERM" >&2
    exit 12
fi
]=]
)

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -E env "ROCJITSU_RUNTIME_DIR=${TEST_RUNTIME_DIR}"
        "RJ_EXECUTABLE=${TEST_EXECUTABLE}" "RJ_GRID_OUT=${TEST_OUTPUT_DIR}"
        "RJ_RANK_SCRIPT=${_rank_script}" /bin/sh "${_driver_script}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
    TIMEOUT 20
)

if(NOT "${_result}" STREQUAL "0")
    message(
        FATAL_ERROR
        "process-grid signal test failed with ${_result}\nstdout:\n${_stdout}\nstderr:\n${_stderr}"
    )
endif()

foreach(_rank RANGE 0 1)
    if(NOT EXISTS "${TEST_OUTPUT_DIR}/term-${_rank}")
        message(FATAL_ERROR "rank ${_rank} did not record SIGTERM")
    endif()
endforeach()

set(_subreaper_dir "${TEST_OUTPUT_DIR}/subreaper")
file(MAKE_DIRECTORY "${_subreaper_dir}")
set(_subreaper_driver "${_subreaper_dir}/driver.py")
file(
    WRITE "${_subreaper_driver}"
    [=[
import ctypes
import os
import signal
import subprocess
import sys
import time

PR_SET_CHILD_SUBREAPER = 36
if ctypes.CDLL(None).prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
    print("failed to become child subreaper", file=sys.stderr)
    sys.exit(2)

out = os.environ["RJ_GRID_OUT"]
runtime = os.environ["ROCJITSU_RUNTIME_DIR"]
launcher_out = open(os.path.join(out, "subreaper-launcher.out"), "w", encoding="utf-8")
launcher_err = open(os.path.join(out, "subreaper-launcher.err"), "w", encoding="utf-8")
cmd = [
    os.environ["RJ_EXECUTABLE"],
    "--profile",
    "mi350x",
    "--num-nodes",
    "1",
    "--nproc-per-node",
    "2",
    "--env",
    f"RJ_GRID_OUT={out}",
    "--",
    "/bin/sh",
    os.environ["RJ_RANK_SCRIPT"],
]
proc = subprocess.Popen(
    cmd,
    stdout=launcher_out,
    stderr=launcher_err,
    env={**os.environ, "ROCJITSU_RUNTIME_DIR": runtime},
)

required = ["pid-0", "pid-1", "worker-0", "worker-1"]
for _ in range(100):
    if all(os.path.exists(os.path.join(out, name)) and os.path.getsize(os.path.join(out, name)) > 0 for name in required):
        break
    time.sleep(0.1)
else:
    proc.terminate()
    proc.wait(timeout=5)
    print("ranks did not start", file=sys.stderr)
    sys.exit(10)

os.kill(proc.pid, signal.SIGTERM)
try:
    status = proc.wait(timeout=12)
except subprocess.TimeoutExpired:
    proc.kill()
    proc.wait(timeout=5)
    print("launcher did not exit while workers were held by a non-reaping subreaper", file=sys.stderr)
    sys.exit(12)
finally:
    launcher_out.close()
    launcher_err.close()

while True:
    try:
        pid, _ = os.waitpid(-1, os.WNOHANG)
    except ChildProcessError:
        break
    if pid == 0:
        break

if status != 143:
    print(f"launcher exited with {status}, expected 143", file=sys.stderr)
    sys.exit(11)
]=]
)

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -E env
        "ROCJITSU_RUNTIME_DIR=${TEST_RUNTIME_DIR}/subreaper"
        "RJ_EXECUTABLE=${TEST_EXECUTABLE}" "RJ_GRID_OUT=${_subreaper_dir}"
        "RJ_RANK_SCRIPT=${_rank_script}" "${PYTHON_EXECUTABLE}"
        "${_subreaper_driver}"
    RESULT_VARIABLE _subreaper_result
    OUTPUT_VARIABLE _subreaper_stdout
    ERROR_VARIABLE _subreaper_stderr
    TIMEOUT 25
)

if(NOT "${_subreaper_result}" STREQUAL "0")
    message(
        FATAL_ERROR
        "process-grid subreaper signal test failed with ${_subreaper_result}\nstdout:\n${_subreaper_stdout}\nstderr:\n${_subreaper_stderr}"
    )
endif()
