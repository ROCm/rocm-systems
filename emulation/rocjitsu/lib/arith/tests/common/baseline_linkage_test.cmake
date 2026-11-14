# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Exercise the audit with real object symbols, including Itanium substitutions.
file(MAKE_DIRECTORY "${WORK}")
foreach(case IN ITEMS allowed allocator)
    if(case STREQUAL "allowed")
        set(source
            "#include <type_traits>\nvoid helper(std::conditional<true, int, float>) {}\n"
        )
    else()
        set(source
            "extern \"C\" void helper() asm(\"_ZNSaIiEaSERKS_\");\nvoid helper() {}\n"
        )
    endif()
    file(WRITE "${WORK}/${case}.cpp" "${source}")
    execute_process(
        COMMAND "${CXX}" -c "${WORK}/${case}.cpp" -o "${WORK}/${case}.o"
        COMMAND_ERROR_IS_FATAL ANY
    )
    execute_process(
        COMMAND
            ${CMAKE_COMMAND} -DNM=${NM} -DARCHIVE=${WORK}/${case}.o -P
            ${CMAKE_CURRENT_LIST_DIR}/baseline_linkage.cmake
        RESULT_VARIABLE status
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
    if(case STREQUAL "allowed")
        if(NOT status EQUAL 0)
            message(FATAL_ERROR "Parameter-only std:: type rejected: ${error}")
        endif()
    elseif(
        status EQUAL 0
        OR NOT error MATCHES "Runtime standard-library definition"
    )
        message(FATAL_ERROR "Allocator definition escaped the audit: ${error}")
    endif()
endforeach()
