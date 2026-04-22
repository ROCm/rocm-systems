# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# -------------------------------------------------------------------------------------- #
#
# ROCm tests
#
# -------------------------------------------------------------------------------------- #

find_package(ROCmVersion)

if(NOT ROCmVersion_FOUND)
    message(
        WARNING
        "ROCmVersion_FOUND not found, skipping tests in ${CMAKE_CURRENT_LIST_FILE}"
    )
    return()
endif()

if(${ROCmVersion_FULL_VERSION} VERSION_GREATER_EQUAL "7.0")
    message(STATUS "Adding Group-By Tests")

    rocprofiler_systems_add_test(
        SKIP_REWRITE SKIP_RUNTIME SKIP_BASELINE
        NAME transpose-group-by-queue
        TARGET transpose
        MPI ${TRANSPOSE_USE_MPI}
        GPU ON
        NUM_PROCS ${NUM_PROCS}
        ENVIRONMENT "${_base_environment};ROCPROFSYS_ROCM_GROUP_BY_QUEUE=YES"
        LABEL "group-by-queue"
        RUNTIME_TIMEOUT 480
    )

    rocprofiler_systems_add_test(
        SKIP_REWRITE SKIP_RUNTIME SKIP_BASELINE
        NAME transpose-group-by-stream
        TARGET transpose
        MPI ${TRANSPOSE_USE_MPI}
        GPU ON
        NUM_PROCS ${NUM_PROCS}
        ENVIRONMENT "${_base_environment};ROCPROFSYS_ROCM_GROUP_BY_QUEUE=NO"
        LABEL "group-by-queue"
        RUNTIME_TIMEOUT 480
    )
endif()
