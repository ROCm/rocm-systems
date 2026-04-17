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

function(ROCPD_CONFIGURE_ROCPD_SCHEMA_FILES SCHEMA_DIR SCHEMA_BINARY_DIR)
    set(SCHEMA_FILES
        "rocpd_tables.sql"
        "rocpd_views.sql"
        "data_views.sql"
        "marker_views.sql"
        "summary_views.sql"
    )

    foreach(SCHEMA_FILE ${SCHEMA_FILES})
        if(NOT EXISTS "${SCHEMA_DIR}/${SCHEMA_FILE}")
            message(
                FATAL_ERROR
                "Schema file ${SCHEMA_FILE} not found in ${SCHEMA_DIR}"
            )
        endif()
    endforeach()

    set(TEMPLATE_FILE "${SCHEMA_DIR}/rocpd_shema.in")

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
