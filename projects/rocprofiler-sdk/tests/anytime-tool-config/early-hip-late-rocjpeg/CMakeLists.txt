# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

#
# Early (preloaded, HIP-only) tool + late (rocjpeg) tool.
#
# Proves a late tool captures the rocjpeg API service of a shared GPU JPEG-decode
# workload, disjoint from the early tool's HIP-API subset.
#
# Optional dependency: requires rocjpeg (libs + the python-rocjpeg workload module +
# sample media). Mirrors tests/rocjpeg gating: the test is always registered but DISABLED
# when the dependency is unavailable. At runtime, if the rocjpeg VA-API backend is
# unavailable, the driver prints "test skipped" (SKIP_REGULAR_EXPRESSION) and the test
# reports as skipped.
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-anytime-early-hip-late-rocjpeg
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(Python3 REQUIRED)

# rocjpeg_FOUND / rocjpeg_VERSION / rocjpeg_ROOT_DIR are inherited from the top-level
# find_package(rocjpeg). The version gate matches tests/bin/CMakeLists.txt (> 0.6.0).
set(rocjpeg_IMAGE_FILE "${rocjpeg_ROOT_DIR}/share/rocjpeg/images/mug_420.jpg")

set(IS_DISABLED OFF)
if(NOT TARGET python-rocjpeg)
    set(IS_DISABLED ON)
elseif(NOT EXISTS "${rocjpeg_IMAGE_FILE}")
    message(WARNING "rocjpeg image not found: ${rocjpeg_IMAGE_FILE}")
    set(IS_DISABLED ON)
endif()

set(early-hip-late-rocjpeg-env
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$<TARGET_FILE_DIR:rocprofiler-sdk-json-tool-late>:$ENV{LD_LIBRARY_PATH}"
    "ROCPROFILER_SDK_TEST_LIB_DIR=$<TARGET_FILE_DIR:rocprofiler-sdk-json-tool-late>"
    "ROCPROFILER_TOOL_CONTEXTS=HIP_API_BUFFERED"
    "ROCPROFILER_TOOL_OUTPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/early-hip-late-rocjpeg-tool-a.json"
    "ROCPROFILER_TOOL_LATE_OUTPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/early-hip-late-rocjpeg-tool-b.json"
    "ROCJPEG_IMAGE_FILE=${rocjpeg_IMAGE_FILE}"
    "ROCPROFILER_TOOL_DISABLE_PERFETTO=1")

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY conftest.py)

rocprofiler_add_integration_execute_test(
    anytime-early-hip-late-rocjpeg
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_early_hip_late_rocjpeg.py
    DEPENDS python-rocjpeg rocprofiler-sdk-json-tool rocprofiler-sdk-json-tool-late
            python-hip-kernels
    TIMEOUT 120
    LABELS "integration-tests;anytime-tool-config"
    DISABLED ${IS_DISABLED}
    PRELOAD "$<TARGET_FILE:rocprofiler-sdk-json-tool>"
    ENVIRONMENT "${early-hip-late-rocjpeg-env}"
    SKIP_REGULAR_EXPRESSION "test skipped"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_SETUP anytime-early-hip-late-rocjpeg)

rocprofiler_add_integration_validate_test(
    anytime-early-hip-late-rocjpeg
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    TIMEOUT 45
    LABELS "integration-tests;anytime-tool-config"
    DISABLED ${IS_DISABLED}
    SKIP_REGULAR_EXPRESSION "test skipped"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_REQUIRED anytime-early-hip-late-rocjpeg
    ARGS --tool-a-input ${CMAKE_CURRENT_BINARY_DIR}/early-hip-late-rocjpeg-tool-a.json
         --tool-b-input ${CMAKE_CURRENT_BINARY_DIR}/early-hip-late-rocjpeg-tool-b.json)
