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

    set(TEMPLATE_FILE "${SCHEMA_DIR}/rocpd_schema.in")

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

# ----------------------------------------------------------------------------------------#
#
# Clone rocprofiler-sdk-rocpd to obtain the schema files if not found in the installed library
#
# ----------------------------------------------------------------------------------------#
function(ROCPD_CLONE_ROCPD_SCHEMA_FILES OUTPUT_SCHEMA_DIR)
    find_package(Git REQUIRED)

    set(CLONE_DIR "${PROJECT_BINARY_DIR}/external/rocprofiler-sdk-rocpd")
    set(SCHEMA_DIR "${CLONE_DIR}/${ROCPD_SCHEMA_SDK_SUBDIR}")

    # clone only when the schema files are not already present
    set(HAVE_ALL_FILES TRUE)
    foreach(FILE ${SCHEMA_FILES})
        if(NOT EXISTS "${SCHEMA_DIR}/${FILE}")
            set(HAVE_ALL_FILES FALSE)
        endif()
    endforeach()

    if(NOT HAVE_ALL_FILES)
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

set(SCHEMA_FILES
    "rocpd_tables.sql"
    "rocpd_views.sql"
    "data_views.sql"
    "summary_views.sql"
    # Adding for future, not using below files in current implementation
    "rocpd_metadata.sql"
    "rocpd_indexes.sql"
    "versions.yml"
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
set(ROCPD_SCHEMA_SDK_SUBDIR
    "projects/rocprofiler-sdk/source/share/rocprofiler-sdk-rocpd"
    CACHE STRING
    "Path (within the cloned repo) to the rocprofiler-sdk-rocpd schema files"
)

set(USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD OFF)
find_package(rocprofiler-sdk-rocpd QUIET)

if(rocprofiler-sdk-rocpd_FOUND)
    set(ROCPD_HAS_SQL_H FALSE)

    if(rocprofiler-sdk-rocpd_INCLUDE_DIR)
        set(_INCLUDE_PATH
            "${rocprofiler-sdk-rocpd_INCLUDE_DIR}/rocprofiler-sdk-rocpd"
        )
        message(STATUS "${_INCLUDE_PATH}/sql.h")
        if(EXISTS "${_INCLUDE_PATH}/sql.h")
            set(ROCPD_HAS_SQL_H TRUE)
        endif()
    endif()

    if(ROCPD_HAS_SQL_H)
        set(USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD ON)
        message(
            STATUS
            "[profiler-hub] rocprofiler-sdk-rocpd found with sql.h - using schema files from rocprofiler-sdk-rocpd library"
        )
    else()
        message(
            STATUS
            "[profiler-hub] rocprofiler-sdk-rocpd found but sql.h missing - cloning schema files from rocprofiler-sdk-rocpd library"
        )
    endif()
else()
    message(
        STATUS
        "[profiler-hub] rocprofiler-sdk-rocpd not found - cloning schema files from rocprofiler-sdk-rocpd library"
    )
endif()

if(NOT USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD)
    # the schema .sql files are not available in the installed library, so they must be
    # obtained by cloning rocprofiler-sdk-rocpd library
    rocpd_clone_rocpd_schema_files(_ROCPD_SCHEMA_DIR)

    rocpd_configure_rocpd_schema_files(
        ${_ROCPD_SCHEMA_DIR}
        ${SQL_SCHEMA_BINARY_DIR}
    )
endif()
