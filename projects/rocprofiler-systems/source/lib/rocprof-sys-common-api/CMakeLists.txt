# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ------------------------------------------------------------------------------#
#
# rocprofiler-systems common API headers: types shared by the core public API
# (rocprofsys_push_category_region, rocprofsys_progress, etc.) that used to live
# alongside the deprecated rocprof-sys-user library.
#
# ------------------------------------------------------------------------------#

set(_common_api_headers ${CMAKE_CURRENT_SOURCE_DIR}/rocprofiler-systems/categories.h)

foreach(_HEADER ${_common_api_headers})
    string(
        REPLACE
        "${CMAKE_CURRENT_SOURCE_DIR}/rocprofiler-systems/"
        "${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}/"
        _DEST
        ${_HEADER}
    )
    string(
        REPLACE
        "${CMAKE_CURRENT_SOURCE_DIR}/"
        "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}/"
        _DEST
        "${_DEST}"
    )

    configure_file(${_HEADER} ${_DEST} COPYONLY)
endforeach()

install(
    FILES ${_common_api_headers}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)
