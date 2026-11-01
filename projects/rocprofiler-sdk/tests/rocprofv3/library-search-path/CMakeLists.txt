# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

find_package(rocprofiler-sdk REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

foreach(_location IN ITEMS local-copy rocm-root-libs)
    string(REPLACE "-" " " _label "${_location}")
    add_library(rocprofv3-library-origin-${_location} SHARED library_origin.c)
    target_compile_definitions(rocprofv3-library-origin-${_location}
                               PRIVATE LIBRARY_LOCATION="${_label}")
    set_target_properties(
        rocprofv3-library-origin-${_location}
        PROPERTIES OUTPUT_NAME library_origin LIBRARY_OUTPUT_DIRECTORY
                                              ${CMAKE_CURRENT_BINARY_DIR}/${_location})
endforeach()

add_executable(rocprofv3-library-origin-app app.c)
target_link_libraries(rocprofv3-library-origin-app
                      PRIVATE rocprofv3-library-origin-local-copy)
add_dependencies(rocprofv3-library-origin-app rocprofv3-library-origin-rocm-root-libs)
# This regression depends on RUNPATH, whose precedence differs from old-style RPATH.
target_link_options(rocprofv3-library-origin-app PRIVATE "LINKER:--enable-new-dtags")

configure_file(run_test.py run_test.py COPYONLY)
rocprofiler_add_integration_execute_test(
    rocprofv3-test-library-search-path
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/run_test.py --rocprofv3
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --library-dir
        $<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library> --app
        $<TARGET_FILE:rocprofv3-library-origin-app> --rocm-library
        $<TARGET_FILE:rocprofv3-library-origin-rocm-root-libs>
    DEPENDS rocprofv3-library-origin-app
    TIMEOUT 60
    LABELS "integration-tests")
