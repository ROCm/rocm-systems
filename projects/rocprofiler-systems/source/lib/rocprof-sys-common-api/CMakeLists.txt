# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ------------------------------------------------------------------------------#
#
# rocprofiler-systems common API headers: types shared across rocprof-sys's
# internal category/tracing machinery and the causal API. Used to live alongside
# the deprecated rocprof-sys-user library.
#
# categories.h: rocprofsys_category_t, used internally by the category/trait
# system (source/lib/core/categories.hpp) and by the Python profiler bindings.
#
# annotation.h: rocprofsys_annotation_t, the payload type for annotated regions
# and progress points (rocprof-sys-causal-api, rocprofsys_push_category_region).
#
# ------------------------------------------------------------------------------#

set(_common_api_headers
    ${CMAKE_CURRENT_SOURCE_DIR}/rocprofiler-systems/categories.h
    ${CMAKE_CURRENT_SOURCE_DIR}/rocprofiler-systems/annotation.h
)

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
