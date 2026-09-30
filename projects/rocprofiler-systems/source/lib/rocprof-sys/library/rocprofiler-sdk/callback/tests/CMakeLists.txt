# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(
    callback-domains-unit-tests
    OBJECT
    test_code_object.cpp
    test_common_tracing_callbacks.cpp
    test_hipfile_api.cpp
    test_rocdecode_api.cpp
    test_rocjpeg_api.cpp
    test_rocshmem_api.cpp
    ${callback_domains_hsa_test_sources}
    ${callback_domains_hip_test_sources}
)

target_link_libraries(
    callback-domains-unit-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-common-library
        rocprofiler-systems-logger
)

target_include_directories(
    callback-domains-unit-tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/source/lib
        ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys
        ${CMAKE_BINARY_DIR}/source/lib/
)
