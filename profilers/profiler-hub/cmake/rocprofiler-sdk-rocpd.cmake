# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ----------------------------------------------------------------------------------------#
#
# ROCpd schema files
#
# ----------------------------------------------------------------------------------------#
function(ROCPD_CONFIGURE_ROCPD_SCHEMA_FILES SCHEMA_DIR SCHEMA_BINARY_DIR)
    # verify the schema files are present in the schema directory
    foreach(SCHEMA_FILE ${SCHEMA_FILES})
        if(NOT EXISTS "${SCHEMA_DIR}/${SCHEMA_FILE}")
            message(
                FATAL_ERROR
                "Schema file ${SCHEMA_FILE} not found in ${SCHEMA_DIR}"
            )
        endif()
    endforeach()

    # The template is profiler-hub's own, not part of the rocpd schema set, so it
    # is read from where it lives rather than from SCHEMA_DIR alongside the .sql.
    set(TEMPLATE_FILE "${SQL_SCHEMA_DIR}/rocpd_schema.in")

    file(MAKE_DIRECTORY ${SCHEMA_BINARY_DIR}/schema)

    foreach(SCHEMA_FILE ${SCHEMA_FILES})
        file(READ "${SCHEMA_DIR}/${SCHEMA_FILE}" SQL_CONTENT)

        string(REPLACE "\\" "\\\\" SQL_CONTENT "${SQL_CONTENT}")
        string(REPLACE "\"" "\\\"" SQL_CONTENT "${SQL_CONTENT}")
        string(REPLACE "\n" "\\n\"\n\"" SQL_CONTENT "${SQL_CONTENT}")

        get_filename_component(SCHEMA_NAME ${SCHEMA_FILE} NAME_WE)
        string(TOUPPER ${SCHEMA_NAME} SCHEMA_NAME_UPPER)

        configure_file(
            "${TEMPLATE_FILE}"
            "${SCHEMA_BINARY_DIR}/schema/${SCHEMA_NAME}.hpp"
            @ONLY
        )
    endforeach()

    message(
        STATUS
        "[profiler-hub] Generating schema headers in ${SCHEMA_BINARY_DIR}/schema"
    )
endfunction()

set(SCHEMA_FILES
    "rocpd_tables.sql"
    "rocpd_views.sql"
    "data_views.sql"
    "summary_views.sql"
    "rocpd_metadata.sql"
    "rocpd_indexes.sql"
)

set(ROCPD_SCHEMA_VERSION
    "3.0.1"
    CACHE STRING
    "Version of the rocprofiler-sdk-rocpd schema to compile in"
)

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

# Shared by the in-tree lookup and the sparse-checkout below, so the two cannot
# drift onto different schema versions.
set(_ROCPD_SCHEMA_SUBDIR
    "projects/rocprofiler-sdk/source/share/rocprofiler-sdk-rocpd/versions/${ROCPD_SCHEMA_VERSION}"
)

# The schema ships in this repository, so it is read from the source tree rather
# than fetched: reading in-tree keeps the schema on the same commit as the rest
# of the build, and sub-projects may not reach the network at configure time.
set(_ROCPD_SCHEMA_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../${_ROCPD_SCHEMA_SUBDIR}"
)

if(EXISTS "${_ROCPD_SCHEMA_DIR}")
    message(
        STATUS
        "[profiler-hub] Using in-tree rocprofiler-sdk-rocpd schema at ${_ROCPD_SCHEMA_DIR}"
    )
elseif(NOT PROFILER_HUB_FETCH_DEPENDENCIES)
    message(
        FATAL_ERROR
        "[profiler-hub] rocprofiler-sdk-rocpd schema directory not found: ${_ROCPD_SCHEMA_DIR}. Configure with -DPROFILER_HUB_FETCH_DEPENDENCIES=ON to clone it instead."
    )
else()
    find_package(Git REQUIRED)

    set(_ROCPD_SCHEMA_CLONE_DIR
        "${PROJECT_BINARY_DIR}/external/rocprofiler-sdk-rocpd"
    )
    set(_ROCPD_SCHEMA_DIR "${_ROCPD_SCHEMA_CLONE_DIR}/${_ROCPD_SCHEMA_SUBDIR}")

    # reuse a clone left by an earlier configure rather than fetching again
    set(_ROCPD_SCHEMA_HAVE_ALL_FILES TRUE)
    foreach(SCHEMA_FILE ${SCHEMA_FILES})
        if(NOT EXISTS "${_ROCPD_SCHEMA_DIR}/${SCHEMA_FILE}")
            set(_ROCPD_SCHEMA_HAVE_ALL_FILES FALSE)
        endif()
    endforeach()

    if(NOT _ROCPD_SCHEMA_HAVE_ALL_FILES)
        if(EXISTS "${_ROCPD_SCHEMA_CLONE_DIR}")
            file(REMOVE_RECURSE "${_ROCPD_SCHEMA_CLONE_DIR}")
        endif()

        # ROCPD_SCHEMA_GIT_BRANCH tracks a moving branch, not this checkout's pin,
        # so a build taking this path compiles against whatever that branch holds
        # at configure time. Said here so the cost is visible in the log.
        message(
            STATUS
            "[profiler-hub] rocprofiler-sdk-rocpd schema not in the source tree; cloning from ${ROCPD_SCHEMA_GIT_URL} @ ${ROCPD_SCHEMA_GIT_BRANCH}, which is not this checkout's pin"
        )

        execute_process(
            COMMAND
                ${GIT_EXECUTABLE} clone --depth 1 --filter=blob:none --sparse
                --branch ${ROCPD_SCHEMA_GIT_BRANCH} ${ROCPD_SCHEMA_GIT_URL}
                ${_ROCPD_SCHEMA_CLONE_DIR}
            RESULT_VARIABLE _ROCPD_SCHEMA_CLONE_RESULT
        )

        if(_ROCPD_SCHEMA_CLONE_RESULT EQUAL 0)
            execute_process(
                COMMAND
                    ${GIT_EXECUTABLE} sparse-checkout set
                    ${_ROCPD_SCHEMA_SUBDIR}
                WORKING_DIRECTORY ${_ROCPD_SCHEMA_CLONE_DIR}
                RESULT_VARIABLE _ROCPD_SCHEMA_CLONE_RESULT
            )
        endif()

        if(NOT _ROCPD_SCHEMA_CLONE_RESULT EQUAL 0)
            message(
                FATAL_ERROR
                "[profiler-hub] Failed to clone rocpd schema files (return code=${_ROCPD_SCHEMA_CLONE_RESULT})"
            )
        endif()
    endif()

    # A sparse-checkout of a path the branch does not carry succeeds and produces
    # nothing, so the clone result above cannot catch it. Reported here because the
    # per-file check in rocpd_configure_rocpd_schema_files names the missing file
    # but neither the version nor the branch that caused it to be missing.
    if(NOT EXISTS "${_ROCPD_SCHEMA_DIR}")
        message(
            FATAL_ERROR
            "[profiler-hub] rocprofiler-sdk-rocpd schema version ${ROCPD_SCHEMA_VERSION} is not on branch ${ROCPD_SCHEMA_GIT_BRANCH} of ${ROCPD_SCHEMA_GIT_URL}: the sparse checkout of ${_ROCPD_SCHEMA_SUBDIR} produced no such directory. Set ROCPD_SCHEMA_VERSION to a version that branch carries, or point ROCPD_SCHEMA_GIT_BRANCH at a branch that has this one."
        )
    endif()

    message(
        STATUS
        "[profiler-hub] Using cloned rocprofiler-sdk-rocpd schema at ${_ROCPD_SCHEMA_DIR}"
    )
endif()

rocpd_configure_rocpd_schema_files(
    ${_ROCPD_SCHEMA_DIR}
    ${SQL_SCHEMA_BINARY_DIR}
)
