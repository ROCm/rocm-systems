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
if(NOT emitted_seed_count EQUAL 4)
    message(FATAL_ERROR "expected 4 emitted seeds, found ${emitted_seed_count}")
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
file(COPY_FILE "${replay_input}" "${failure_output}/failure.bin")
file(WRITE "${failure_output}/failure.seed" "987654321\n")
file(SHA256 "${failure_output}/failure.bin" original_failure_hash)
file(SHA256 "${failure_output}/failure.seed" original_seed_hash)

function(assert_saved_failure phase)
    foreach(artifact IN ITEMS failure.bin failure.seed)
        if(NOT EXISTS "${failure_output}/${artifact}")
            message(FATAL_ERROR "${phase} removed the earlier ${artifact}")
        endif()
    endforeach()
    file(SHA256 "${failure_output}/failure.bin" failure_hash)
    file(SHA256 "${failure_output}/failure.seed" seed_hash)
    if(
        NOT failure_hash STREQUAL original_failure_hash
        OR NOT seed_hash STREQUAL original_seed_hash
    )
        message(FATAL_ERROR "${phase} changed the earlier reproducer")
    endif()
endfunction()

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

assert_saved_failure("setup failure")

file(REMOVE "${failure_output}/thread-1")
execute_process(
    COMMAND
        "${FUZZ_EXECUTABLE}" --random 2 --seed 0 --threads 2 --output
        "${failure_output}"
    RESULT_VARIABLE rerun_result
    OUTPUT_VARIABLE rerun_stdout
    ERROR_VARIABLE rerun_stderr
)
if(NOT rerun_result EQUAL 0)
    message(
        FATAL_ERROR
        "rerun failed (${rerun_result}): ${rerun_stdout}${rerun_stderr}"
    )
endif()
if(NOT rerun_stdout MATCHES "\\\"inputs\\\":2")
    message(FATAL_ERROR "rerun did not report both inputs: ${rerun_stdout}")
endif()
assert_saved_failure("successful rerun")

function(assert_cli_failure expected_diagnostic)
    execute_process(
        COMMAND "${FUZZ_EXECUTABLE}" ${ARGN}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr
    )
    if(NOT result EQUAL 2 OR NOT stderr MATCHES "${expected_diagnostic}")
        message(
            FATAL_ERROR
            "expected exit 2 and '${expected_diagnostic}', got ${result}: ${stdout}${stderr}"
        )
    endif()
endfunction()

assert_cli_failure("expected an unsigned integer" --random 18446744073709551616)
assert_cli_failure("expected an unsigned integer" --random 1tail)
assert_cli_failure("threads must be positive" --random 1 --threads 0)
assert_cli_failure("cannot open input: ${WORK_DIRECTORY}/missing.bin" --input "${WORK_DIRECTORY}/missing.bin")
file(WRITE "${WORK_DIRECTORY}/empty.bin" "")
execute_process(
    COMMAND "${FUZZ_EXECUTABLE}" --input "${WORK_DIRECTORY}/empty.bin"
    RESULT_VARIABLE empty_result
    OUTPUT_VARIABLE empty_stdout
    ERROR_VARIABLE empty_stderr
)
if(NOT empty_result EQUAL 0 OR NOT empty_stdout MATCHES "\\\"inputs\\\":1")
    message(
        FATAL_ERROR
        "empty input failed (${empty_result}): ${empty_stdout}${empty_stderr}"
    )
endif()
string(REPEAT "x" 1048577 oversized_bytes)
file(WRITE "${WORK_DIRECTORY}/oversized.bin" "${oversized_bytes}")
assert_cli_failure("input has an invalid size" --input "${WORK_DIRECTORY}/oversized.bin")

set(flush_output "${WORK_DIRECTORY}/seed-flush-failure")
execute_process(
    COMMAND
        "${PYTHON_EXECUTABLE}" -c
        "import os, resource, signal, sys; signal.signal(signal.SIGXFSZ, signal.SIG_IGN); _, hard = resource.getrlimit(resource.RLIMIT_FSIZE); resource.setrlimit(resource.RLIMIT_FSIZE, (16, hard)); os.execv(sys.argv[1], sys.argv[1:])"
        "${FUZZ_EXECUTABLE}" --random 1 --seed 58 --threads 1 --output
        "${flush_output}"
    RESULT_VARIABLE flush_result
    OUTPUT_VARIABLE flush_stdout
    ERROR_VARIABLE flush_stderr
)
if(
    NOT flush_result EQUAL 1
    OR NOT flush_stderr MATCHES "cannot update active fuzz seed"
)
    message(
        FATAL_ERROR
        "seed flush failure returned ${flush_result}: ${flush_stdout}${flush_stderr}"
    )
endif()
file(READ "${flush_output}/failure.seed" flush_seed)
file(SIZE "${flush_output}/failure.bin" flush_input_size)
if(NOT flush_seed STREQUAL "58\n" OR NOT flush_input_size EQUAL 4)
    message(
        FATAL_ERROR
        "seed flush failure did not preserve the seed and replay input"
    )
endif()
execute_process(
    COMMAND "${FUZZ_EXECUTABLE}" --input "${flush_output}/failure.bin"
    RESULT_VARIABLE flush_replay_result
    OUTPUT_VARIABLE flush_replay_stdout
    ERROR_VARIABLE flush_replay_stderr
)
if(
    NOT flush_replay_result EQUAL 0
    OR NOT flush_replay_stdout MATCHES "\\\"inputs\\\":1"
)
    message(
        FATAL_ERROR
        "seed flush reproducer failed (${flush_replay_result}): ${flush_replay_stdout}${flush_replay_stderr}"
    )
endif()

file(REMOVE_RECURSE "${WORK_DIRECTORY}")
