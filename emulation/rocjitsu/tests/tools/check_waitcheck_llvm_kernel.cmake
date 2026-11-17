# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

foreach(required WAITCHECK INPUT TARGET EXPECT_HAZARD)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "missing -D${required}=...")
    endif()
endforeach()

execute_process(
    COMMAND "${WAITCHECK}" "${INPUT}" --list-code-objects
    RESULT_VARIABLE list_result
    OUTPUT_VARIABLE list_stdout
    ERROR_VARIABLE list_stderr
)
set(list_output "${list_stdout}${list_stderr}")
if(NOT list_result EQUAL 0)
    message(FATAL_ERROR "failed to inspect ${INPUT}:\n${list_output}")
endif()
if(NOT list_output MATCHES "${TARGET}: 1")
    message(
        FATAL_ERROR
        "${INPUT} does not contain exactly one ${TARGET} code object:\n${list_output}"
    )
endif()

execute_process(
    COMMAND "${WAITCHECK}" "${INPUT}" --target "${TARGET}" --max-diagnostics 4
    RESULT_VARIABLE waitcheck_result
    OUTPUT_VARIABLE waitcheck_stdout
    ERROR_VARIABLE waitcheck_stderr
)
set(waitcheck_output "${waitcheck_stdout}${waitcheck_stderr}")

if(EXPECT_HAZARD)
    if(NOT waitcheck_result EQUAL 4)
        message(
            FATAL_ERROR
            "expected hazard exit 4, got ${waitcheck_result}:\n${waitcheck_output}"
        )
    endif()
    if(NOT waitcheck_output MATCHES "diagnostics=1")
        message(
            FATAL_ERROR
            "missing one-diagnostic summary:\n${waitcheck_output}"
        )
    endif()
    if(NOT waitcheck_output MATCHES "missing s_waitcnt lgkmcnt\\(0\\)")
        message(
            FATAL_ERROR
            "missing expected perturbed-wait diagnostic:\n${waitcheck_output}"
        )
    endif()
else()
    if(NOT waitcheck_result EQUAL 0)
        message(
            FATAL_ERROR
            "clean LLVM kernel failed waitcheck with ${waitcheck_result}:\n${waitcheck_output}"
        )
    endif()
    if(NOT waitcheck_output MATCHES "diagnostics=0")
        message(
            FATAL_ERROR
            "clean scan did not report diagnostics=0:\n${waitcheck_output}"
        )
    endif()
endif()
