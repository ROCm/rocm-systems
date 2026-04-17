# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ----------------------------------------------------------------------------------------#
#
# ROCpd schema files
#
# ----------------------------------------------------------------------------------------#

function(
    ROCPD_CONFIGURE_SCHEMA_VERSION
    SCHEMA_DIR
    SCHEMA_BINARY_DIR
    VERSION_NAME
    NAMESPACE_NAME
)
    set(SCHEMA_FILES
        "rocpd_tables.sql"
        "rocpd_views.sql"
        "data_views.sql"
        "summary_views.sql"
    )

    # Add rocpd_metadata.sql only for v4 (it doesn't exist in v3)
    if(EXISTS "${SCHEMA_DIR}/rocpd_metadata.sql")
        list(APPEND SCHEMA_FILES "rocpd_metadata.sql")
    endif()

    foreach(SCHEMA_FILE ${SCHEMA_FILES})
        if(NOT EXISTS "${SCHEMA_DIR}/${SCHEMA_FILE}")
            message(
                FATAL_ERROR
                "Schema file ${SCHEMA_FILE} not found in ${SCHEMA_DIR}"
            )
        endif()
    endforeach()

    set(TEMPLATE_FILE "${SCHEMA_DIR}/rocpd_shema.in")

    file(MAKE_DIRECTORY "${SCHEMA_BINARY_DIR}/schema/${VERSION_NAME}")

    foreach(SCHEMA_FILE ${SCHEMA_FILES})
        file(READ "${SCHEMA_DIR}/${SCHEMA_FILE}" SQL_CONTENT)

        string(REPLACE "\\" "\\\\" SQL_CONTENT "${SQL_CONTENT}")
        string(REPLACE "\"" "\\\"" SQL_CONTENT "${SQL_CONTENT}")
        string(REPLACE "\n" "\\n\"\n\"" SQL_CONTENT "${SQL_CONTENT}")

        get_filename_component(SCHEMA_NAME ${SCHEMA_FILE} NAME_WE)
        string(TOUPPER ${SCHEMA_NAME} SCHEMA_NAME_UPPER)

        configure_file(
            "${TEMPLATE_FILE}"
            "${SCHEMA_BINARY_DIR}/schema/${VERSION_NAME}/${SCHEMA_NAME}.hpp"
            @ONLY
        )
    endforeach()

    message(
        STATUS
        "[rocpdsna] Generated ${VERSION_NAME} schema headers (namespace: ${NAMESPACE_NAME}) in ${SCHEMA_BINARY_DIR}/schema/${VERSION_NAME}"
    )
endfunction()

function(ROCPD_CONFIGURE_ROCPD_SCHEMA_FILES SCHEMA_DIR SCHEMA_BINARY_DIR)
    # Use PROJECT_SOURCE_DIR to get correct path
    set(BACKENDS_DIR "${PROJECT_SOURCE_DIR}/source/data_storage/backends")

    # Generate V3 schema headers
    set(V3_SCHEMA_DIR "${BACKENDS_DIR}/schema/3.0.0")
    if(EXISTS "${V3_SCHEMA_DIR}")
        rocpd_configure_schema_version("${V3_SCHEMA_DIR}" "${SCHEMA_BINARY_DIR}" "3.0.0" "schema_v3")
    else()
        message(
            WARNING
            "[rocpdsna] V3 schema directory not found: ${V3_SCHEMA_DIR}"
        )
    endif()

    # Generate V4 schema headers
    set(V4_SCHEMA_DIR "${BACKENDS_DIR}/schema/4.0.0")
    if(EXISTS "${V4_SCHEMA_DIR}")
        rocpd_configure_schema_version("${V4_SCHEMA_DIR}" "${SCHEMA_BINARY_DIR}" "4.0.0" "schema_v4")
    else()
        message(
            WARNING
            "[rocpdsna] V4 schema directory not found: ${V4_SCHEMA_DIR}"
        )
    endif()

    message(
        STATUS
        "[rocpdsna] Generating schema headers in ${SCHEMA_BINARY_DIR}/schema"
    )
endfunction()

set(USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD
    OFF
    CACHE BOOL
    "Use schema from rocprofiler-sdk-rocpd library"
    FORCE
)

find_package(rocprofiler-sdk-rocpd QUIET)

if(rocprofiler-sdk-rocpd_FOUND)
    set(ROCPD_HAS_SQL_H FALSE)
    #    if(rocprofiler-sdk-rocpd_INCLUDE_DIR)
    #        set(_INCLUDE_PATH
    #            "${rocprofiler-sdk-rocpd_INCLUDE_DIR}/rocprofiler-sdk-rocpd"
    #        )
    #        message(STATUS "${_INCLUDE_PATH}/sql.h")
    #        if(EXISTS "${_INCLUDE_PATH}/sql.h")

    # Temporary check will remove after new rocprofiler-sdk-rocpd is released.
    # rocprofiler-sdk-rocpd_INCLUDE_DIR may be a list (source + build include dirs).
    # Search each entry for sql.h and detect API features.
    set(ROCPD_SQL_HAS_SCHEMA_VERSION 0)
    foreach(_ROCPD_INC_DIR ${rocprofiler-sdk-rocpd_INCLUDE_DIR})
        set(_ROCPD_SQL_H "${_ROCPD_INC_DIR}/rocprofiler-sdk-rocpd/sql.h")
        if(EXISTS "${_ROCPD_SQL_H}")
            set(ROCPD_HAS_SQL_H TRUE)

            # Distinguish old vs new API by checking two markers in sql.h:
            #   1. ROCPD_SQL_SCHEMA_ROCPD_METADATA enum value (old SDK has MARKER_VIEWS instead)
            #   2. rocpd_version_triplet_t param in rocpd_sql_load_schema signature
            # The old SDK (rocm-7.2.2) has neither; the new SDK has both.
            file(READ "${_ROCPD_SQL_H}" _ROCPD_SQL_H_CONTENTS)
            string(
                FIND "${_ROCPD_SQL_H_CONTENTS}"
                "ROCPD_SQL_SCHEMA_ROCPD_METADATA"
                _ROCPD_METADATA_POS
            )
            string(
                FIND "${_ROCPD_SQL_H_CONTENTS}"
                "rocpd_version_triplet_t"
                _ROCPD_VERSION_PARAM_POS
            )
            if(
                NOT _ROCPD_METADATA_POS EQUAL -1
                AND NOT _ROCPD_VERSION_PARAM_POS EQUAL -1
            )
                set(ROCPD_SQL_HAS_SCHEMA_VERSION 1)
                message(
                    STATUS
                    "[rocpdsna] Detected new rocpd API (ROCPD_SQL_SCHEMA_ROCPD_METADATA + rocpd_version_triplet_t in ${_ROCPD_SQL_H})"
                )
            else()
                message(
                    STATUS
                    "[rocpdsna] Detected old rocpd API (${_ROCPD_SQL_H})"
                )
            endif()

            break()
        endif()
    endforeach()

    if(ROCPD_HAS_SQL_H)
        set(USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD
            ON
            CACHE BOOL
            "Use schema from rocprofiler-sdk-rocpd library"
            FORCE
        )

        message(
            STATUS
            "[rocpdsna] rocprofiler-sdk-rocpd found with sql.h - using latest schema files"
        )
    else()
        message(
            STATUS
            "[rocpdsna] rocprofiler-sdk-rocpd found but sql.h missing - using local schema files"
        )
    endif()
else()
    message(
        STATUS
        "[rocpdsna] rocprofiler-sdk-rocpd not found - using local schema files"
    )
endif()

if(NOT USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD)
    rocpd_configure_rocpd_schema_files(${SQL_SCHEMA_DIR} ${SQL_SCHEMA_BINARY_DIR})
endif()
