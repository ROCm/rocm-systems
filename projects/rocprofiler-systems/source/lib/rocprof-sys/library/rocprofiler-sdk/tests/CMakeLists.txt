# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_path(GET CMAKE_CURRENT_LIST_DIR PARENT_PATH _rocprofiler_sdk_source_dir)

set(rocprofiler_sdk_components_unit_tests_sources
    test_rccl_unit.cpp
    test_roctx_client.cpp
    test_service_compatibility.cpp
    test_spm.cpp
    ${_rocprofiler_sdk_source_dir}/service_compatibility.cpp
    ${_rocprofiler_sdk_source_dir}/trace_control.cpp
    ${_rocprofiler_sdk_source_dir}/spm.cpp
)

add_library(
    rocprofiler-sdk-components-unit-tests
    OBJECT
    ${rocprofiler_sdk_components_unit_tests_sources}
)

# Unit tests intentionally compile only SDK-independent service policy, SPM
# configuration validation, and the no-SDK runtime fallback path here.
# The SDK runtime helpers are compile-checked by the normal SPM-enabled
# rocprofiler-systems-object-library target.
set_source_files_properties(
    ${_rocprofiler_sdk_source_dir}/spm.cpp
    PROPERTIES COMPILE_DEFINITIONS ROCPROFSYS_DISABLE_SPM_RUNTIME=1
)

unset(_rocprofiler_sdk_source_dir)

target_link_libraries(
    rocprofiler-sdk-components-unit-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems::rocprofiler-systems-core
        rocprofiler-systems::rocprofiler-systems-interface-library
        rocprofiler-systems::rocprofiler-systems-timemory
        rocprofiler-systems-common-library
        rocprofiler-systems-logger
)

target_include_directories(
    rocprofiler-sdk-components-unit-tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/source/lib
        ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys
        ${CMAKE_BINARY_DIR}/source/lib/
)
