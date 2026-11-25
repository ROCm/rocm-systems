# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

foreach(mode IN ITEMS local daemon attach)
    set(arguments)
    if(NOT mode STREQUAL "local")
        list(APPEND arguments "--${mode}")
    endif()
    execute_process(
        COMMAND
            "${ROCJITSU}" --config "${CONFIG}" ${arguments} --
            "${CMAKE_COMMAND}" -E echo "application started"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
    if(
        result EQUAL 0
        OR NOT output STREQUAL ""
        OR NOT error MATCHES "application launches require the interposer"
    )
        message(
            FATAL_ERROR
            "${mode} launch was not rejected: ${result}\n${output}${error}"
        )
    endif()
endforeach()
