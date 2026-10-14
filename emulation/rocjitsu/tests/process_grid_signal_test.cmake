if(NOT DEFINED TEST_EXECUTABLE)
    message(FATAL_ERROR "TEST_EXECUTABLE is required")
endif()
if(NOT DEFINED TEST_RUNTIME_DIR)
    message(FATAL_ERROR "TEST_RUNTIME_DIR is required")
endif()
if(NOT DEFINED TEST_OUTPUT_DIR)
    message(FATAL_ERROR "TEST_OUTPUT_DIR is required")
endif()

file(REMOVE_RECURSE "${TEST_OUTPUT_DIR}")
file(MAKE_DIRECTORY "${TEST_OUTPUT_DIR}")

set(_rank_script "${TEST_OUTPUT_DIR}/rank.sh")
file(
    WRITE "${_rank_script}"
    [=[#!/bin/sh
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
    [=[#!/bin/sh
"$RJ_EXECUTABLE" --profile mi350x --num-nodes 1 --nproc-per-node 2 \
    --env "RJ_GRID_OUT=$RJ_GRID_OUT" -- /bin/sh "$RJ_RANK_SCRIPT" \
    > "$RJ_GRID_OUT/launcher.out" 2> "$RJ_GRID_OUT/launcher.err" &
launcher=$!

i=0
while [ "$i" -lt 100 ]; do
    [ -s "$RJ_GRID_OUT/pid-0" ] && [ -s "$RJ_GRID_OUT/pid-1" ] && break
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
    [ "$alive" -eq 0 ] && break
    i=$((i + 1))
    sleep 0.1
done

if kill -0 "$pid0" 2>/dev/null || kill -0 "$pid1" 2>/dev/null; then
    echo "rank process survived launcher SIGTERM" >&2
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
