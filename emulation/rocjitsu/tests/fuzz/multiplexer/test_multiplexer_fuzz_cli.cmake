# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

file(REMOVE_RECURSE "${WORK_DIRECTORY}")
file(MAKE_DIRECTORY "${WORK_DIRECTORY}")

set(seed_directory "${WORK_DIRECTORY}/seeds")
execute_process(
    COMMAND "${FUZZ_EXECUTABLE}" --emit-seeds "${seed_directory}"
    RESULT_VARIABLE seed_result
    OUTPUT_VARIABLE seed_stdout
    ERROR_VARIABLE seed_stderr
)
if(NOT seed_result EQUAL 0)
    message(FATAL_ERROR "seed emission failed: ${seed_stdout}${seed_stderr}")
endif()
file(GLOB emitted_seeds LIST_DIRECTORIES false "${seed_directory}/*.bin")
list(LENGTH emitted_seeds emitted_seed_count)
if(NOT emitted_seed_count EQUAL 6)
    message(
        FATAL_ERROR
        "expected 6 multiplexer seeds, found ${emitted_seed_count}"
    )
endif()

foreach(target IN ITEMS decode execution code-object)
    execute_process(
        COMMAND "${FUZZ_EXECUTABLE}" --input "${seed_directory}/${target}.bin"
        RESULT_VARIABLE replay_result
        OUTPUT_VARIABLE replay_stdout
        ERROR_VARIABLE replay_stderr
    )
    if(
        NOT replay_result EQUAL 0
        OR NOT replay_stdout MATCHES "\\\"target\\\":\\\"${target}\\\""
    )
        message(
            FATAL_ERROR
            "${target} replay failed: ${replay_stdout}${replay_stderr}"
        )
    endif()
endforeach()

file(REMOVE_RECURSE "${WORK_DIRECTORY}")
