# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ----------------------------------------------------------------------------------------#
#
# ROCpd schema files
#
# ----------------------------------------------------------------------------------------#
function(ROCPD_CONFIGURE_ROCPD_SCHEMA_FILES SCHEMA_DIR SCHEMA_BINARY_DIR)
    set(TEMPLATE_FILE "${SCHEMA_DIR}/rocpd_schema.in")

    file(MAKE_DIRECTORY ${SCHEMA_BINARY_DIR}/schema)

    message(
        STATUS
        "[profiler-hub] Generating schema headers in ${SCHEMA_BINARY_DIR}/schema"
    )

    # Recursively discover every schema file, not just the ones in the top-level
    # directory: this also picks up per-version schemas nested under sub-directories,
    # e.g. versions/3.0.0/rocpd_tables.sql, versions/3.0.1/rocpd_tables.sql, etc.
    file(
        GLOB_RECURSE SCHEMA_FILES_ALL
        RELATIVE "${SCHEMA_DIR}"
        CONFIGURE_DEPENDS
        "${SCHEMA_DIR}/*.sql"
    )
    foreach(SCHEMA_FILE ${SCHEMA_FILES_ALL})
        file(READ "${SCHEMA_DIR}/${SCHEMA_FILE}" SQL_CONTENT)

        string(REPLACE "\\" "\\\\" SQL_CONTENT "${SQL_CONTENT}")
        string(REPLACE "\"" "\\\"" SQL_CONTENT "${SQL_CONTENT}")
        string(REPLACE "\n" "\\n\"\n\"" SQL_CONTENT "${SQL_CONTENT}")

        get_filename_component(SCHEMA_NAME ${SCHEMA_FILE} NAME_WE)
        get_filename_component(SCHEMA_SUBDIR ${SCHEMA_FILE} DIRECTORY)
        string(TOUPPER ${SCHEMA_NAME} SCHEMA_NAME_UPPER)

        if(SCHEMA_SUBDIR)
            string(TOUPPER "${SCHEMA_SUBDIR}" SCHEMA_SUBDIR_UPPER)
            string(
                REGEX REPLACE
                "[/.\\-]+"
                "_"
                SCHEMA_SUBDIR_UPPER
                "${SCHEMA_SUBDIR_UPPER}"
            )
            set(SCHEMA_NAME_UPPER "${SCHEMA_SUBDIR_UPPER}_${SCHEMA_NAME_UPPER}")
            set(SCHEMA_OUT_DIR "${SCHEMA_BINARY_DIR}/schema/${SCHEMA_SUBDIR}")
        else()
            set(SCHEMA_OUT_DIR "${SCHEMA_BINARY_DIR}/schema")
        endif()

        file(MAKE_DIRECTORY "${SCHEMA_OUT_DIR}")

        configure_file(
            "${TEMPLATE_FILE}"
            "${SCHEMA_OUT_DIR}/${SCHEMA_NAME}.hpp"
            @ONLY
        )
    endforeach()
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

    # Copy header template, so that rocpd_configure_rocpd_schema_files can find it
    # from the cloned schema directory
    if(NOT EXISTS "${SCHEMA_DIR}/rocpd_schema.in")
        file(
            COPY "${SQL_SCHEMA_DIR}/rocpd_schema.in"
            DESTINATION "${SCHEMA_DIR}"
        )
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

# the schema .sql files are not available in the installed library, so they must be
# obtained by cloning rocprofiler-sdk-rocpd library
rocpd_clone_rocpd_schema_files(_ROCPD_SCHEMA_DIR)

rocpd_configure_rocpd_schema_files(
    ${_ROCPD_SCHEMA_DIR}
    ${SQL_SCHEMA_BINARY_DIR}
)
