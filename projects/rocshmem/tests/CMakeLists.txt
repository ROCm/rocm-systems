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

# Enable testing at the top level
if(BUILD_CTESTS)
    enable_testing()
endif()

IF (BUILD_FUNCTIONAL_TESTS)
    add_subdirectory(functional_tests)
ENDIF()

IF (BUILD_UNIT_TESTS)
    add_subdirectory(unit_tests)
ENDIF()

# Print top-level test summary if CTest is enabled
if(BUILD_CTESTS AND (BUILD_FUNCTIONAL_TESTS OR BUILD_UNIT_TESTS))
    # Get all tests from this directory and subdirectories
    get_directory_property(all_tests TESTS)
    list(LENGTH all_tests total_tests)

    message(STATUS "")
    message(STATUS "====================================================================")
    message(STATUS "ROCshmem CTest Summary")
    message(STATUS "====================================================================")
    message(STATUS "Total tests registered: ${total_tests}")
    message(STATUS "")
    message(STATUS "CI Test Tiers:")
    message(STATUS "  ctest -L quick           # Quick smoke test")
    message(STATUS "  ctest -L standard        # Standard suite")
    message(STATUS "  ctest -L comprehensive   # Comprehensive suite")
    message(STATUS "  ctest -L full            # Full suite")
    message(STATUS "")
    message(STATUS "Test Category Filters:")
    message(STATUS "  ctest -L functional      # Functional tests")
    message(STATUS "  ctest -L unit            # Unit tests")
    message(STATUS "  ctest -L RMA             # RMA functional tests")
    message(STATUS "  ctest -L AMO             # Atomic operations")
    message(STATUS "  ctest -L COLLECTIVE      # Collective operations")
    message(STATUS "  ctest -L HEATMAP         # Performance/heatmap tests")
    message(STATUS "")
    message(STATUS "Backend-Specific Testing:")
    message(STATUS "  ROCSHMEM_BACKEND_TYPE=ipc ctest -L quick")
    message(STATUS "  ROCSHMEM_BACKEND_TYPE=ro ctest -L standard")
    message(STATUS "  ROCSHMEM_BACKEND_TYPE=gda ctest -L comprehensive")
    message(STATUS "")
    message(STATUS "Or run from subdirectories:")
    if(BUILD_FUNCTIONAL_TESTS)
        message(STATUS "  cd tests/functional_tests && ctest -L functional")
    endif()
    if(BUILD_UNIT_TESTS)
        message(STATUS "  cd tests/unit_tests && ctest -L unit")
    endif()
    message(STATUS "====================================================================")
    message(STATUS "")
endif()

# Install CTest infrastructure for running tests from installation directory
if(BUILD_CTESTS AND (BUILD_FUNCTIONAL_TESTS OR BUILD_UNIT_TESTS))
    # Set subdirs() calls based on what's built
    if(BUILD_FUNCTIONAL_TESTS)
        set(BUILD_FUNCTIONAL_TESTS_SUBDIRS "subdirs(\"tests/functional\")")
    else()
        set(BUILD_FUNCTIONAL_TESTS_SUBDIRS "# Functional tests not built")
    endif()

    if(BUILD_UNIT_TESTS)
        set(BUILD_UNIT_TESTS_SUBDIRS "subdirs(\"tests/unit\")")
    else()
        set(BUILD_UNIT_TESTS_SUBDIRS "# Unit tests not built")
    endif()

    # Generate and install top-level CTestTestfile.cmake
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/CTestInstallRoot.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/CTestInstallRoot.cmake
        @ONLY)

    rocm_install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CTestInstallRoot.cmake
                 COMPONENT tests
                 DESTINATION ${CMAKE_INSTALL_BINDIR}/rocshmem
                 RENAME CTestTestfile.cmake)
endif()

IF (BUILD_PYTHON_TESTS)
    # Canonical source of truth is projects/rocshmem/python/tests.
    # This test CMake path only installs those assets into the test package.
    configure_file(../scripts/python_tests/driver.sh rocshmem_python_driver.sh COPYONLY)
    rocm_install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/rocshmem_python_driver.sh
                 COMPONENT tests DESTINATION ${CMAKE_INSTALL_DATADIR}/rocshmem)

    # Install all Python test files so adding new tests does not require
    # touching this CMake list.
    file(GLOB ROCSHMEM_PYTHON_TEST_FILES CONFIGURE_DEPENDS
        ${CMAKE_SOURCE_DIR}/python/tests/*.py)
    list(SORT ROCSHMEM_PYTHON_TEST_FILES)

    rocm_install(FILES
        ${ROCSHMEM_PYTHON_TEST_FILES}
        COMPONENT tests
        DESTINATION ${CMAKE_INSTALL_DATADIR}/rocshmem/python_tests)

    rocm_install(PROGRAMS
        ${CMAKE_SOURCE_DIR}/python/launch_test.sh
        COMPONENT tests
        DESTINATION ${CMAKE_INSTALL_DATADIR}/rocshmem)
ENDIF()
