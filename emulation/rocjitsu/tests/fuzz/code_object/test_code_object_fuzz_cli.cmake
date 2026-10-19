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
file(GLOB emitted_seeds LIST_DIRECTORIES false "${seed_directory}/seed-*.bin")
list(LENGTH emitted_seeds emitted_seed_count)
if(NOT emitted_seed_count EQUAL 3)
    message(FATAL_ERROR "expected 3 emitted seeds, found ${emitted_seed_count}")
endif()

execute_process(
    COMMAND "${FUZZ_EXECUTABLE}" --emit-seeds "${seed_directory}"
    RESULT_VARIABLE reused_seed_result
    OUTPUT_VARIABLE reused_seed_stdout
    ERROR_VARIABLE reused_seed_stderr
)
if(NOT reused_seed_result EQUAL 2)
    message(
        FATAL_ERROR
        "nonempty seed directory returned ${reused_seed_result}, expected 2: "
        "${reused_seed_stdout}${reused_seed_stderr}"
    )
endif()

set(replay_input "${seed_directory}/seed-0000.bin")
execute_process(
    COMMAND "${FUZZ_EXECUTABLE}" --input "${replay_input}"
    RESULT_VARIABLE replay_result
    OUTPUT_VARIABLE replay_stdout
    ERROR_VARIABLE replay_stderr
)
if(
    NOT replay_result EQUAL 0
    OR NOT replay_stdout MATCHES "\\\"inputs\\\":1"
    OR NOT replay_stdout MATCHES "\\\"code_objects_valid\\\":2"
)
    message(FATAL_ERROR "seed replay failed: ${replay_stdout}${replay_stderr}")
endif()

# Keep this smoke test below the first known malformed-loader witness. Sustained
# campaigns require bounds checks for the loader's untrusted count and offsets.
set(success_output "${WORK_DIRECTORY}/success")
execute_process(
    COMMAND
        "${FUZZ_EXECUTABLE}" --random 64 --seed 0 --threads 4 --output
        "${success_output}"
    RESULT_VARIABLE success_result
    OUTPUT_VARIABLE success_stdout
    ERROR_VARIABLE success_stderr
)
if(NOT success_result EQUAL 0)
    message(
        FATAL_ERROR
        "random smoke failed (${success_result}): ${success_stdout}${success_stderr}"
    )
endif()
if(NOT success_stdout MATCHES "\\\"inputs\\\":64")
    message(
        FATAL_ERROR
        "random smoke did not report all inputs: ${success_stdout}"
    )
endif()

set(failure_output "${WORK_DIRECTORY}/setup-failure")
file(MAKE_DIRECTORY "${failure_output}")
file(WRITE "${failure_output}/thread-1" "blocks directory creation")
execute_process(
    COMMAND
        "${FUZZ_EXECUTABLE}" --random 2 --seed 0 --threads 2 --output
        "${failure_output}"
    RESULT_VARIABLE failure_result
    OUTPUT_VARIABLE failure_stdout
    ERROR_VARIABLE failure_stderr
)
if(NOT failure_result EQUAL 2)
    message(
        FATAL_ERROR
        "setup failure returned ${failure_result}, expected 2: ${failure_stdout}${failure_stderr}"
    )
endif()

file(REMOVE_RECURSE "${WORK_DIRECTORY}")
