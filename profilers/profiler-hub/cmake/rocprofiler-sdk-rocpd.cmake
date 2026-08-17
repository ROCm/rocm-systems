# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ----------------------------------------------------------------------------------------#
#
# ROCpd schema files
#
# ----------------------------------------------------------------------------------------#
function(ROCPD_COPY_ROCPD_SCHEMA_FILES SCHEMA_DIR DEST_DIR)
    file(MAKE_DIRECTORY "${DEST_DIR}")

    # Copy everything (top-level .sql files, versions.yml, and the versioned
    # versions/<version>/*.sql subdirectories) except build-only CMakeLists.txt
    # files, preserving the directory layout.
    file(
        COPY "${SCHEMA_DIR}/"
        DESTINATION "${DEST_DIR}"
        PATTERN "CMakeLists.txt"
        EXCLUDE
    )

    message(
        STATUS
        "[profiler-hub] Copied rocpd schema files from ${SCHEMA_DIR} to ${DEST_DIR}"
    )
endfunction()

# ----------------------------------------------------------------------------------------#
#
# Clone rocprofiler-sdk-rocpd to obtain the schema files if not found in the installed library
#
# ----------------------------------------------------------------------------------------#
function(ROCPD_CLONE_ROCPD_SCHEMA_FILES OUTPUT_SCHEMA_DIR)
    find_package(Git REQUIRED)

    set(CLONE_DIR "${PROJECT_BINARY_DIR}/external/rocprofiler-sdk-rocpd")
    set(SCHEMA_DIR "${CLONE_DIR}/${ROCPD_SCHEMA_SDK_SUBDIR}")

    # clone only when the schema directory is not already present
    if(NOT EXISTS "${SCHEMA_DIR}")
        if(EXISTS "${CLONE_DIR}")
            file(REMOVE_RECURSE "${CLONE_DIR}")
        endif()

        message(
            STATUS
            "[profiler-hub] Cloning rocprofiler-sdk-rocpd schema from ${ROCPD_SCHEMA_GIT_URL} @ ${ROCPD_SCHEMA_GIT_BRANCH}"
        )

        execute_process(
            COMMAND
                ${GIT_EXECUTABLE} clone --depth 1 --filter=blob:none --sparse
                --branch ${ROCPD_SCHEMA_GIT_BRANCH} ${ROCPD_SCHEMA_GIT_URL}
                ${CLONE_DIR}
            RESULT_VARIABLE RESULT
        )

        if(RESULT EQUAL 0)
            execute_process(
                COMMAND
                    ${GIT_EXECUTABLE} sparse-checkout set
                    ${ROCPD_SCHEMA_SDK_SUBDIR}
                WORKING_DIRECTORY ${CLONE_DIR}
                RESULT_VARIABLE RESULT
            )
        endif()

        if(NOT RESULT EQUAL 0)
            message(
                FATAL_ERROR
                "[profiler-hub] Failed to clone rocprofiler-sdk-rocpd (return code=${RESULT})"
            )
        endif()
    endif()

    message(
        STATUS
        "[profiler-hub] Using cloned rocprofiler-sdk-rocpd schema at ${SCHEMA_DIR}"
    )

    set(${OUTPUT_SCHEMA_DIR} "${SCHEMA_DIR}" PARENT_SCOPE)
endfunction()

set(ROCPD_SCHEMA_GIT_URL
    "https://github.com/ROCm/rocm-systems.git"
    CACHE STRING
    "Git repository to clone for rocprofiler-sdk-rocpd schema files"
)
set(ROCPD_SCHEMA_GIT_BRANCH
    "develop"
    CACHE STRING
    "Git branch/tag to clone for rocprofiler-sdk-rocpd schema files"
)
set(ROCPD_SCHEMA_SDK_SUBDIR
    "projects/rocprofiler-sdk/source/share/rocprofiler-sdk-rocpd"
    CACHE STRING
    "Path (within the cloned repo) to the rocprofiler-sdk-rocpd schema files"
)

rocpd_clone_rocpd_schema_files(_ROCPD_SCHEMA_DIR)
rocpd_copy_rocpd_schema_files(
    ${_ROCPD_SCHEMA_DIR}
    ${SQL_SCHEMA_DIR}
)
