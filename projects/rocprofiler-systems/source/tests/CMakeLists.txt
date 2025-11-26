# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

set(unit_test_sources dummy.cpp)

# Add AMD SMI unit tests when ROCm is enabled
if(ROCPROFSYS_USE_ROCM)
    list(APPEND unit_test_sources
        ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys/library/amd_smi/test/sampler_tests.cpp
    )
endif()

add_executable(rocprof-sys-unit-tests ${unit_test_sources})

target_link_libraries(
    rocprof-sys-unit-tests
    PRIVATE rocprofiler-systems-googletest-library
)

# Link against the main library for AMD SMI tests
if(ROCPROFSYS_USE_ROCM)
    target_link_libraries(
        rocprof-sys-unit-tests
        PRIVATE
            rocprofiler-systems::librocprofiler-systems-static
            rocprofiler-systems::rocprofiler-systems-interface-library
    )
    target_include_directories(
        rocprof-sys-unit-tests
        PRIVATE
            ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys
            ${PROJECT_SOURCE_DIR}/source/lib
    )
endif()

include(GoogleTest)
gtest_discover_tests(rocprof-sys-unit-tests)
