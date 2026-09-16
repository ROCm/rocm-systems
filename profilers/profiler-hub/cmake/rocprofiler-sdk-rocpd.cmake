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

# The schema ships in this repository, so it is read from the source tree rather
# than fetched: reading in-tree keeps the schema on the same commit as the rest
# of the build, and sub-projects may not reach the network at configure time.
set(_ROCPD_SCHEMA_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../projects/rocprofiler-sdk/source/share/rocprofiler-sdk-rocpd/versions/${ROCPD_SCHEMA_VERSION}"
)

if(NOT EXISTS "${_ROCPD_SCHEMA_DIR}")
    message(
        FATAL_ERROR
        "[profiler-hub] rocprofiler-sdk-rocpd schema directory not found: ${_ROCPD_SCHEMA_DIR}"
    )
endif()

message(
    STATUS
    "[profiler-hub] Using in-tree rocprofiler-sdk-rocpd schema at ${_ROCPD_SCHEMA_DIR}"
)

rocpd_configure_rocpd_schema_files(
    ${_ROCPD_SCHEMA_DIR}
    ${SQL_SCHEMA_BINARY_DIR}
)
