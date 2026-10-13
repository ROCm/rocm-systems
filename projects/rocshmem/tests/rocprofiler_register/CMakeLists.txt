###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################

# Regression test for rocprofiler-register integration (AIPROFSYST-787).
#
# Verifies that ROCSHMEM_ROCPROFILER_REGISTER=1 was set at compile time by:
#   (a) Link-time: fails if rocprofiler_register_import_rocshmem is missing
#       from librocshmem.a.  The executable is always built so this check
#       fires even when CTest registration is disabled (BUILD_CTESTS=OFF).
#   (b) Runtime:   calls the function and asserts the returned version equals
#       the version in rocshmem_config.h.
#
# No MPI or GPU required.

# Always build the executable: the link step itself is the first regression
# guard — a missing symbol causes a build failure regardless of BUILD_CTESTS.
add_executable(rocshmem-rocprofiler-register-import-test
    test_import.cpp
)

target_link_libraries(rocshmem-rocprofiler-register-import-test
    PRIVATE
        ${PROJECT_NAME}
        rocprofiler-register::rocprofiler-register
)

# Register with CTest and apply tier labels only when CTest is enabled.
# Mirrors the pattern used by functional_tests and unit_tests.
if(BUILD_CTESTS)
    add_test(
        NAME    rocshmem-rocprofiler-register-import
        COMMAND rocshmem-rocprofiler-register-import-test
    )

    set_tests_properties(rocshmem-rocprofiler-register-import
        PROPERTIES
            LABELS "rocprofiler-register;quick"
    )

    # Apply YAML-based tier labels so this test appears in CI tier filters
    # (e.g. ctest -L pre-commit, ctest -L nightly).
    set(_rocp_reg_categories_yaml "${CMAKE_SOURCE_DIR}/tests/test_categories.yaml")
    include(${ROCM_SYSTEMS_ROOT}/shared/ctest/TestCategories.cmake OPTIONAL)
    if(COMMAND apply_ctest_category_labels AND EXISTS "${_rocp_reg_categories_yaml}")
        apply_ctest_category_labels("${_rocp_reg_categories_yaml}")
    endif()
endif()
