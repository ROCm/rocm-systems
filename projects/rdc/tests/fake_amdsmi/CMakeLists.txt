# Copyright (c) 2026 - present Advanced Micro Devices, Inc. All rights reserved.
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

# Fake amd-smi for the GPU-less tests in tests/ci/test_fake_gpu.py. Test-only:
# it is built with BUILD_TESTS and installed with the tests component next to
# the suite, never on a library search path. It replaces libamd_smi through
# LD_PRELOAD, so it must not link libamd_smi or any RDC library.
add_library(rdc_fake_amdsmi SHARED fake_amdsmi.cc)
target_include_directories(rdc_fake_amdsmi PRIVATE ${AMD_SMI_INCLUDE_DIR})
find_package(Threads REQUIRED)
target_link_libraries(rdc_fake_amdsmi PRIVATE Threads::Threads)
set_target_properties(rdc_fake_amdsmi PROPERTIES CXX_VISIBILITY_PRESET default INSTALL_RPATH "")

set(FAKE_GPU_TEST_DIR ${CMAKE_INSTALL_DATAROOTDIR}/${RDC}/rdctst_tests/fake_gpu)
install(
    TARGETS rdc_fake_amdsmi
    LIBRARY DESTINATION ${FAKE_GPU_TEST_DIR} COMPONENT ${TESTS_COMPONENT}
)
# Every tests/ci/test_fake_gpu*.py runs from this installed copy; CI's run of
# tests/ci skips them.
file(GLOB FAKE_GPU_TESTS CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/tests/ci/test_fake_gpu*.py)
install(
    FILES fake_values.inc ${FAKE_GPU_TESTS} ${PROJECT_SOURCE_DIR}/tests/ci/fake_gpu_golden.py
    DESTINATION ${FAKE_GPU_TEST_DIR}
    COMPONENT ${TESTS_COMPONENT}
)
