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
# Not installed: rocprofsys_category_t is an implementation detail, not part of
# the public ABI.
#
# annotation.h: rocprofsys_annotation_t, the payload type for annotated regions
# and progress points (rocprof-sys-causal-api, rocprofsys_push_category_region).
#
# Header-only INTERFACE target so consumers link it instead of hard-coding this
# directory. It is exported because rocprofiler-systems-causal-api-library
# depends on it publicly and is itself exported.
#
# ------------------------------------------------------------------------------#

add_library(rocprofiler-systems-common-api-library INTERFACE)
add_library(
    rocprofiler-systems::rocprofiler-systems-common-api-library
    ALIAS rocprofiler-systems-common-api-library
)

set(_common_api_headers ${CMAKE_CURRENT_SOURCE_DIR}/rocprofiler-systems/annotation.h)

target_include_directories(
    rocprofiler-systems-common-api-library
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

rocprofiler_systems_install_public_headers(${_common_api_headers})

install(
    TARGETS rocprofiler-systems-common-api-library
    EXPORT rocprofiler-systems-library-targets
)
