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

set(rocprof-sys-library-source 
    ${CMAKE_CURRENT_SOURCE_DIR}/amd_smi/service_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/amd_smi/processor_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/amd_smi/amd_smi_tests.cpp
)

add_library(unit-tests-rocprof-sys STATIC ${rocprof-sys-library-source})

add_library(rocprof::unit-tests::sys ALIAS unit-tests-rocprof-sys)

target_link_libraries(
    unit-tests-rocprof-sys
    rocprofiler-systems::rocprofiler-systems-common-library
    rocprofiler-systems::rocprofiler-systems-core
    rocprofiler-systems::rocprofiler-systems-headers
    rocprofiler-systems::rocprofiler-systems-interface-library
    rocprofiler-systems::rocprofiler-systems-binary
    amd-smi::amd-smi
)
