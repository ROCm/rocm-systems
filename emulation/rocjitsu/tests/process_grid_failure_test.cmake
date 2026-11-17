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

set(_rank_script "${TEST_OUTPUT_DIR}/rank.py")
file(
    WRITE "${_rank_script}"
    [=[
import os
import signal
import subprocess
import sys
import time

out = os.environ["RJ_GRID_OUT"]
rank = os.environ["RANK"]

def wait_for(path):
    for _ in range(100):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return
        time.sleep(0.1)

if rank != "0":
    worker = subprocess.Popen(["/usr/bin/sleep", "60"])
    with open(os.path.join(out, f"worker-{rank}"), "w", encoding="utf-8") as f:
        print(worker.pid, file=f, flush=True)

    def stop(signum, frame):
        with open(os.path.join(out, f"term-{rank}"), "w", encoding="utf-8") as f:
            print("term", file=f, flush=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

with open(os.path.join(out, f"pid-{rank}"), "w", encoding="utf-8") as f:
    print(os.getpid(), file=f, flush=True)

if rank == "0":
    wait_for(os.path.join(out, "pid-1"))
    sys.exit(42)

while True:
    time.sleep(1)
]=]
)

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -E env
        "ROCJITSU_RUNTIME_DIR=${TEST_RUNTIME_DIR}/peer-failure"
        "${TEST_EXECUTABLE}" --profile mi350x --num-nodes 1 --nproc-per-node 2
        --env "RJ_GRID_OUT=${TEST_OUTPUT_DIR}" -- "${PYTHON_EXECUTABLE}"
        "${_rank_script}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
    TIMEOUT 15
)

if(NOT "${_result}" STREQUAL "42")
    message(
        FATAL_ERROR
        "expected process-grid exit 42, got ${_result}\nstdout:\n${_stdout}\nstderr:\n${_stderr}"
    )
endif()
if(NOT _stderr MATCHES "process-grid rank 0 exited with status 42")
    message(
        FATAL_ERROR
        "expected rank-failure diagnostic naming rank 0 and status 42\nstderr:\n${_stderr}"
    )
endif()

if(NOT EXISTS "${TEST_OUTPUT_DIR}/term-1")
    message(
        FATAL_ERROR
        "rank 1 did not observe termination after rank 0 failed\nstdout:\n${_stdout}\nstderr:\n${_stderr}"
    )
endif()
if(NOT EXISTS "${TEST_OUTPUT_DIR}/worker-1")
    message(FATAL_ERROR "rank 1 did not record its worker pid")
endif()

file(READ "${TEST_OUTPUT_DIR}/worker-1" _worker_pid)
string(STRIP "${_worker_pid}" _worker_pid)
set(_worker_alive 1)
foreach(_attempt RANGE 0 50)
    execute_process(
        COMMAND /bin/sh -c "kill -0 ${_worker_pid} 2>/dev/null"
        RESULT_VARIABLE _worker_alive
    )
    if(NOT _worker_alive EQUAL 0)
        break()
    endif()
    execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep 0.1)
endforeach()

if(_worker_alive EQUAL 0)
    message(
        FATAL_ERROR
        "rank 1 background worker survived process-grid termination"
    )
endif()

set(_leader_exit_dir "${TEST_OUTPUT_DIR}/leader-exit")
file(MAKE_DIRECTORY "${_leader_exit_dir}")
set(_leader_exit_script "${_leader_exit_dir}/rank.py")
file(
    WRITE "${_leader_exit_script}"
    [=[
import os
import subprocess
import sys
import time

out = os.environ["RJ_GRID_OUT"]
rank = os.environ["RANK"]

def wait_for(path):
    for _ in range(100):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return
        time.sleep(0.1)

if rank == "0":
    worker = subprocess.Popen(["/usr/bin/sleep", "60"])
    with open(os.path.join(out, "worker-0"), "w", encoding="utf-8") as f:
        print(worker.pid, file=f, flush=True)
    with open(os.path.join(out, "pid-0"), "w", encoding="utf-8") as f:
        print(os.getpid(), file=f, flush=True)
    sys.exit(0)

wait_for(os.path.join(out, "worker-0"))
with open(os.path.join(out, "pid-1"), "w", encoding="utf-8") as f:
    print(os.getpid(), file=f, flush=True)
time.sleep(0.5)
sys.exit(42)
]=]
)

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -E env
        "ROCJITSU_RUNTIME_DIR=${TEST_RUNTIME_DIR}/leader-exit"
        "${TEST_EXECUTABLE}" --profile mi350x --num-nodes 1 --nproc-per-node 2
        --env "RJ_GRID_OUT=${_leader_exit_dir}" -- "${PYTHON_EXECUTABLE}"
        "${_leader_exit_script}"
    RESULT_VARIABLE _leader_result
    OUTPUT_VARIABLE _leader_stdout
    ERROR_VARIABLE _leader_stderr
    TIMEOUT 15
)

if(NOT "${_leader_result}" STREQUAL "42")
    message(
        FATAL_ERROR
        "expected leader-exit grid to report peer failure 42, got ${_leader_result}\nstdout:\n${_leader_stdout}\nstderr:\n${_leader_stderr}"
    )
endif()
if(NOT EXISTS "${_leader_exit_dir}/worker-0")
    message(FATAL_ERROR "rank 0 did not record its worker pid")
endif()
file(READ "${_leader_exit_dir}/worker-0" _leader_worker_pid)
string(STRIP "${_leader_worker_pid}" _leader_worker_pid)
set(_leader_worker_alive 1)
foreach(_attempt RANGE 0 50)
    execute_process(
        COMMAND /bin/sh -c "kill -0 ${_leader_worker_pid} 2>/dev/null"
        RESULT_VARIABLE _leader_worker_alive
    )
    if(NOT _leader_worker_alive EQUAL 0)
        break()
    endif()
    execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep 0.1)
endforeach()
if(_leader_worker_alive EQUAL 0)
    message(FATAL_ERROR "rank 0 worker survived after its leader exited")
endif()

set(_adopted_live_dir "${TEST_OUTPUT_DIR}/adopted-live-child")
file(MAKE_DIRECTORY "${_adopted_live_dir}")
set(_adopted_live_script "${_adopted_live_dir}/rank.py")
file(
    WRITE "${_adopted_live_script}"
    [=[
import os
import sys
import time

out = os.environ["RJ_GRID_OUT"]

if os.environ["RANK"] == "0":
    child = os.fork()
    if child == 0:
        worker = os.fork()
        if worker == 0:
            with open(os.path.join(out, "worker-pid"), "w", encoding="utf-8") as f:
                print(os.getpid(), file=f, flush=True)
            time.sleep(0.5)
            with open(os.path.join(out, "worker-done"), "w", encoding="utf-8") as f:
                print("done", file=f, flush=True)
            sys.exit(0)
        sys.exit(0)

    os.waitpid(child, 0)
    for _ in range(50):
        if os.path.exists(os.path.join(out, "worker-done")):
            with open(os.path.join(out, "rank-done"), "w", encoding="utf-8") as f:
                print("done", file=f, flush=True)
            sys.exit(0)
        time.sleep(0.1)
    sys.exit(7)

sys.exit(0)
]=]
)

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -E env
        "ROCJITSU_RUNTIME_DIR=${TEST_RUNTIME_DIR}/adopted-live-child"
        "${TEST_EXECUTABLE}" --profile mi350x --num-nodes 1 --nproc-per-node 1
        --env "RJ_GRID_OUT=${_adopted_live_dir}" -- "${PYTHON_EXECUTABLE}"
        "${_adopted_live_script}"
    RESULT_VARIABLE _adopted_live_result
    OUTPUT_VARIABLE _adopted_live_stdout
    ERROR_VARIABLE _adopted_live_stderr
    TIMEOUT 15
)

if(NOT "${_adopted_live_result}" STREQUAL "0")
    message(
        FATAL_ERROR
        "expected live adopted child launch to finish, got ${_adopted_live_result}\nstdout:\n${_adopted_live_stdout}\nstderr:\n${_adopted_live_stderr}"
    )
endif()
if(NOT EXISTS "${_adopted_live_dir}/rank-done")
    message(FATAL_ERROR "rank did not survive its live adopted child")
endif()
if(NOT EXISTS "${_adopted_live_dir}/worker-done")
    message(FATAL_ERROR "live adopted child did not finish")
endif()

set(_adopted_child_dir "${TEST_OUTPUT_DIR}/adopted-child")
file(MAKE_DIRECTORY "${_adopted_child_dir}")
set(_adopted_child_script "${_adopted_child_dir}/rank.py")
file(
    WRITE "${_adopted_child_script}"
    [=[
import os
import signal
import sys
import time

out = os.environ["RJ_GRID_OUT"]
rank = os.environ["RANK"]

def wait_for(path):
    for _ in range(100):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return
        time.sleep(0.1)

if rank == "0":
    child = os.fork()
    if child == 0:
        os.setsid()
        with open(os.path.join(out, "worker-0"), "w", encoding="utf-8") as f:
            print(os.getpid(), file=f, flush=True)

        def stop(signum, frame):
            with open(os.path.join(out, "term-worker-0"), "w", encoding="utf-8") as f:
                print("term", file=f, flush=True)
            sys.exit(0)

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        while True:
            time.sleep(1)
    with open(os.path.join(out, "pid-0"), "w", encoding="utf-8") as f:
        print(os.getpid(), file=f, flush=True)
    sys.exit(0)

wait_for(os.path.join(out, "worker-0"))
with open(os.path.join(out, "pid-1"), "w", encoding="utf-8") as f:
    print(os.getpid(), file=f, flush=True)
time.sleep(0.5)
sys.exit(42)
]=]
)

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -E env
        "ROCJITSU_RUNTIME_DIR=${TEST_RUNTIME_DIR}/adopted-child"
        "${TEST_EXECUTABLE}" --profile mi350x --num-nodes 1 --nproc-per-node 2
        --env "RJ_GRID_OUT=${_adopted_child_dir}" -- "${PYTHON_EXECUTABLE}"
        "${_adopted_child_script}"
    RESULT_VARIABLE _adopted_result
    OUTPUT_VARIABLE _adopted_stdout
    ERROR_VARIABLE _adopted_stderr
    TIMEOUT 15
)

if(NOT "${_adopted_result}" STREQUAL "42")
    message(
        FATAL_ERROR
        "expected adopted-child grid to report peer failure 42, got ${_adopted_result}\nstdout:\n${_adopted_stdout}\nstderr:\n${_adopted_stderr}"
    )
endif()
if(NOT EXISTS "${_adopted_child_dir}/worker-0")
    message(FATAL_ERROR "rank 0 did not record its adopted worker pid")
endif()
file(READ "${_adopted_child_dir}/worker-0" _adopted_worker_pid)
string(STRIP "${_adopted_worker_pid}" _adopted_worker_pid)
set(_adopted_worker_alive 1)
foreach(_attempt RANGE 0 50)
    execute_process(
        COMMAND /bin/sh -c "kill -0 ${_adopted_worker_pid} 2>/dev/null"
        RESULT_VARIABLE _adopted_worker_alive
    )
    if(NOT _adopted_worker_alive EQUAL 0)
        break()
    endif()
    execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep 0.1)
endforeach()
if(_adopted_worker_alive EQUAL 0)
    message(FATAL_ERROR "setsid worker survived after rank exit")
endif()

set(_sigchld_child "${TEST_OUTPUT_DIR}/exit_42.py")
file(WRITE "${_sigchld_child}" "import sys\nsys.exit(42)\n")
set(_sigchld_driver "${TEST_OUTPUT_DIR}/sigchld_ignored.py")
file(
    WRITE "${_sigchld_driver}"
    [=[
import os
import signal

signal.signal(signal.SIGCHLD, signal.SIG_IGN)
os.execv(
    os.environ["RJ_EXECUTABLE"],
    [
        os.environ["RJ_EXECUTABLE"],
        "--profile",
        "mi350x",
        "--num-nodes",
        "1",
        "--nproc-per-node",
        "2",
        "--",
        os.environ["PYTHON_EXECUTABLE"],
        os.environ["RJ_CHILD_SCRIPT"],
    ],
)
]=]
)

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -E env
        "ROCJITSU_RUNTIME_DIR=${TEST_RUNTIME_DIR}/sigchld"
        "RJ_EXECUTABLE=${TEST_EXECUTABLE}"
        "PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}"
        "RJ_CHILD_SCRIPT=${_sigchld_child}" "${PYTHON_EXECUTABLE}"
        "${_sigchld_driver}"
    RESULT_VARIABLE _sigchld_result
    OUTPUT_VARIABLE _sigchld_stdout
    ERROR_VARIABLE _sigchld_stderr
    TIMEOUT 15
)

if(NOT "${_sigchld_result}" STREQUAL "42")
    message(
        FATAL_ERROR
        "expected inherited ignored SIGCHLD launch to report rank failure 42, got ${_sigchld_result}\nstdout:\n${_sigchld_stdout}\nstderr:\n${_sigchld_stderr}"
    )
endif()
