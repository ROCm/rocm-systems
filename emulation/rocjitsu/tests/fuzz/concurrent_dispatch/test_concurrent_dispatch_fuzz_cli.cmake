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
file(GLOB random_replays "${random_output}/failure-seed-*.bin")
if(random_replays)
    message(
        FATAL_ERROR
        "successful random run left a replay: ${random_replays}"
    )
endif()

set(retained_replay "${random_output}/failure-seed-0.bin")
file(WRITE "${retained_replay}" "retained input")
execute_process(
    COMMAND "${FUZZ_EXECUTABLE}" --random 1 --seed 0 --output "${random_output}"
    RESULT_VARIABLE existing_replay_result
    OUTPUT_VARIABLE existing_replay_stdout
    ERROR_VARIABLE existing_replay_stderr
)
file(READ "${retained_replay}" retained_contents)
if(
    NOT existing_replay_result EQUAL 1
    OR NOT retained_contents STREQUAL "retained input"
)
    message(
        FATAL_ERROR
        "existing replay was not preserved: ${existing_replay_stdout}${existing_replay_stderr}"
    )
endif()

find_program(PRLIMIT_EXECUTABLE prlimit)
if(PRLIMIT_EXECUTABLE)
    execute_process(
        COMMAND
            "${PRLIMIT_EXECUTABLE}" --fsize=4:4 -- sh -c
            [=[trap '' XFSZ; exec "$@"]=] sh "${FUZZ_EXECUTABLE}" --random 1
            --seed 1 --output "${random_output}"
        RESULT_VARIABLE failed_write_result
        ERROR_VARIABLE failed_write_stderr
    )
    file(GLOB partial_writes "${random_output}/.rj-input-*.tmp")
    if(
        NOT failed_write_result EQUAL 1
        OR NOT failed_write_stderr MATCHES "cannot write input"
        OR EXISTS "${random_output}/failure-seed-1.bin"
        OR partial_writes
    )
        message(
            FATAL_ERROR
            "failed write left an unusable replay: ${failed_write_stderr}"
        )
    endif()

    execute_process(
        COMMAND
            "${PRLIMIT_EXECUTABLE}" --core=0:0 --fsize=0:0 --
            "${FUZZ_EXECUTABLE}" --random 1 --seed 1 --output "${random_output}"
        RESULT_VARIABLE interrupted_write_result
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(
        interrupted_write_result STREQUAL "0"
        OR EXISTS "${random_output}/failure-seed-1.bin"
    )
        message(FATAL_ERROR "interrupted write published a replay")
    endif()

    execute_process(
        COMMAND
            "${FUZZ_EXECUTABLE}" --random 1 --seed 1 --output "${random_output}"
        RESULT_VARIABLE retry_result
        ERROR_VARIABLE retry_stderr
        OUTPUT_QUIET
    )
    if(NOT retry_result EQUAL 0)
        message(
            FATAL_ERROR
            "retry after a write failure failed: ${retry_stderr}"
        )
    endif()
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
        primary_stalled_modes
        fatal_exit_modes
        terminal_faulted_modes
        published_completions
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

foreach(option IN ITEMS --random --seed)
    execute_process(
        COMMAND "${FUZZ_EXECUTABLE}" --input "${replay_input}" "${option}" 0
        RESULT_VARIABLE zero_result
        OUTPUT_VARIABLE zero_stdout
        ERROR_VARIABLE zero_stderr
    )
    if(NOT zero_result EQUAL 2)
        message(
            FATAL_ERROR
            "replay with ${option} 0 returned ${zero_result}, expected 2: "
            "${zero_stdout}${zero_stderr}"
        )
    endif()
endforeach()

foreach(option IN ITEMS --input --emit-seeds --output)
    execute_process(
        COMMAND "${FUZZ_EXECUTABLE}" --input "${replay_input}" "${option}" ""
        RESULT_VARIABLE empty_path_result
        OUTPUT_VARIABLE empty_path_stdout
        ERROR_VARIABLE empty_path_stderr
    )
    if(
        NOT empty_path_result EQUAL 2
        OR NOT empty_path_stderr MATCHES "expected a nonempty path"
    )
        message(
            FATAL_ERROR
            "empty path for ${option} was not rejected: "
            "${empty_path_stdout}${empty_path_stderr}"
        )
    endif()
endforeach()

execute_process(
    COMMAND
        "${FUZZ_EXECUTABLE}" --random 2 --seed 18446744073709551615 --output
        "${random_output}"
    RESULT_VARIABLE overflow_result
    OUTPUT_VARIABLE overflow_stdout
    ERROR_VARIABLE overflow_stderr
)
if(
    NOT overflow_result EQUAL 2
    OR NOT overflow_stderr MATCHES "seed range exceeds uint64"
)
    message(
        FATAL_ERROR
        "wrapped seed range was not rejected: ${overflow_stdout}${overflow_stderr}"
    )
endif()

file(REMOVE_RECURSE "${WORK_DIRECTORY}")
