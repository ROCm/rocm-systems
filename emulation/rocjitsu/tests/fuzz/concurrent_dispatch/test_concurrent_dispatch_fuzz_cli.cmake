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
file(
    GLOB emitted_seeds
    LIST_DIRECTORIES false
    "${seed_directory}/seed-????.bin"
)
list(LENGTH emitted_seeds emitted_seed_count)
if(NOT emitted_seed_count EQUAL 12)
    message(
        FATAL_ERROR
        "expected 12 padded seed names, found ${emitted_seed_count}"
    )
endif()
foreach(seed IN ITEMS seed-0000.bin seed-0011.bin)
    if(NOT EXISTS "${seed_directory}/${seed}")
        message(FATAL_ERROR "missing emitted seed ${seed}")
    endif()
endforeach()

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
if(NOT replay_result EQUAL 0)
    message(FATAL_ERROR "seed replay failed: ${replay_stdout}${replay_stderr}")
endif()

set(random_output "${WORK_DIRECTORY}/random")
execute_process(
    COMMAND "${FUZZ_EXECUTABLE}" --random 1 --seed 0 --output "${random_output}"
    RESULT_VARIABLE random_result
    OUTPUT_VARIABLE random_stdout
    ERROR_VARIABLE random_stderr
)
if(NOT random_result EQUAL 0)
    message(FATAL_ERROR "random smoke failed: ${random_stdout}${random_stderr}")
endif()

foreach(
    field
    IN
    ITEMS
        inputs
        targets
        modes
        completed_modes
        budget_limited_modes
        terminal_faulted_modes
        instructions
        packets
        workgroups
        workitems
        kernel_bytes
        completed_inputs
        incomplete_inputs
        budget_limited_inputs
        multi_xcd_inputs
        cache_primed_inputs
        fanout_inputs
        differential_inputs
        pooled_dispatch_threads
)
    if(NOT replay_stdout MATCHES "\\\"${field}\\\":")
        message(FATAL_ERROR "replay JSON is missing ${field}: ${replay_stdout}")
    endif()
    if(NOT random_stdout MATCHES "\\\"${field}\\\":")
        message(FATAL_ERROR "random JSON is missing ${field}: ${random_stdout}")
    endif()
endforeach()

execute_process(
    COMMAND
        "${FUZZ_EXECUTABLE}" --input "${replay_input}" --random 1 --output
        "${random_output}"
    RESULT_VARIABLE exclusive_result
    OUTPUT_VARIABLE exclusive_stdout
    ERROR_VARIABLE exclusive_stderr
)
if(NOT exclusive_result EQUAL 2)
    message(
        FATAL_ERROR
        "mutually exclusive modes returned ${exclusive_result}, expected 2: "
        "${exclusive_stdout}${exclusive_stderr}"
    )
endif()

file(REMOVE_RECURSE "${WORK_DIRECTORY}")
