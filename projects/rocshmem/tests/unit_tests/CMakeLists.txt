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

cmake_minimum_required(VERSION 3.16.3 FATAL_ERROR)

###############################################################################
# PROJECT
###############################################################################
include(${CMAKE_SOURCE_DIR}/cmake/setup_project.cmake)
project(rocshmem_unit_tests VERSION 1.0.0 LANGUAGES CXX)

# ASAN option (if not already defined by parent project)
option(ASAN "Enable AddressSanitizer" ${ASAN})

###############################################################################
# SOURCES
###############################################################################
add_executable(${PROJECT_NAME} "")

target_include_directories(
  ${PROJECT_NAME}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(
  ${PROJECT_NAME}
  PRIVATE
    shmem_gtest.cpp
    heap_memory_gtest.cpp
    hipmalloc_gtest.cpp
    #bitwise_gtest.cpp # Test is disabled because of compilation errors
    index_strategy_gtest.cpp
    single_heap_gtest.cpp
    symmetric_heap_gtest.cpp
    dlmalloc_gtest.cpp
    remote_heap_info_gtest.cpp
    mpi_instance_gtest.cpp
    abql_block_mutex_gtest.cpp
    notifier_gtest.cpp
    free_list_gtest.cpp
    wavefront_size_gtest.cpp
    atomic_wf_queue_gtest.cpp
    envvar_gtest.cpp
    backend_bc_buffer_register_gtest.cpp
)

if (USE_IPC)
  target_sources(
    ${PROJECT_NAME}
    PRIVATE
      ipc_impl_simple_coarse_gtest.cpp
      ipc_impl_simple_fine_gtest.cpp
      ipc_impl_tiled_fine_gtest.cpp
  )
endif()


if (USE_GDA)
  target_sources(
    ${PROJECT_NAME}
    PRIVATE
      topology_gtest.cpp
  )
endif()

###############################################################################
# ROCSHMEM DEPENDENCY
###############################################################################
find_package(hip REQUIRED PATHS /opt/rocm)
find_package(MPI REQUIRED)

if (BUILD_TESTS_ONLY)
  find_package(rocshmem REQUIRED PATHS /opt/rocm)

  target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
      $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  )
endif()

target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE
    MPI::MPI_CXX
    roc::rocshmem
)

###############################################################################
# GTEST DEPENDENCY
###############################################################################
include(FetchContent)

FetchContent_Declare(
    googletest
    PREFIX extern/googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.0
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(BUILD_GTEST ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

configure_file(../../scripts/unit_tests/driver.sh rocshmem_unit_driver.sh COPYONLY)
rocm_install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/rocshmem_unit_driver.sh COMPONENT tests DESTINATION ${CMAKE_INSTALL_DATADIR}/rocshmem)
install(TARGETS rocshmem_unit_tests COMPONENT tests RUNTIME DESTINATION ${CMAKE_INSTALL_DATADIR}/rocshmem)

target_link_libraries(
  ${PROJECT_NAME}
  PRIVATE
    gtest
    gtest_main
)

###############################################################################
# CTEST INTEGRATION (replaces driver.sh with native CTest tests)
###############################################################################
# Enable testing
enable_testing()

# Install the wrapper script
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/unit_test_wrapper.sh
               ${CMAKE_CURRENT_BINARY_DIR}/unit_test_wrapper.sh COPYONLY)
rocm_install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/unit_test_wrapper.sh
             COMPONENT tests DESTINATION ${CMAKE_INSTALL_DATADIR}/rocshmem)

# Include CTest-based unit tests
# Set BUILD_CTESTS=ON to register unit tests
if(BUILD_CTESTS)
    message(STATUS "Registering CTest-based unit tests")
    include(${CMAKE_CURRENT_SOURCE_DIR}/CTestUnitTests.cmake OPTIONAL)

    # Check if CTest generation functions are available
    if(COMMAND init_install_ctest_generation AND COMMAND register_all_unit_tests)
        # Initialize install-time test generation
        set(INSTALL_CTEST_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/CTestUnitInstall.cmake")
        init_install_ctest_generation("${INSTALL_CTEST_OUTPUT}")

        # Configure template header
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/CTestUnitInstall.cmake.in
            ${INSTALL_CTEST_OUTPUT}
            @ONLY)

        # Register all tests (populates both build-time and install-time)
        register_all_unit_tests()

        # Apply YAML-based tier labels (build-time only)
        set(_test_categories_yaml "${CMAKE_SOURCE_DIR}/tests/test_categories.yaml")
        include(${ROCM_SYSTEMS_ROOT}/shared/ctest/TestCategories.cmake OPTIONAL)
        if(COMMAND apply_ctest_category_labels AND EXISTS "${_test_categories_yaml}")
            # Apply tier labels to build-time tests
            apply_ctest_category_labels("${_test_categories_yaml}")
        endif()

        # Apply tier labels to install-time CTest file
        include(${CMAKE_SOURCE_DIR}/tests/ApplyInstallTierLabels.cmake OPTIONAL)
        if(COMMAND apply_install_tier_labels AND EXISTS "${_test_categories_yaml}")
            apply_install_tier_labels("${INSTALL_CTEST_OUTPUT}" "${_test_categories_yaml}")
        endif()

        # Install the generated install-time CTest file
        rocm_install(FILES "${INSTALL_CTEST_OUTPUT}"
                     COMPONENT tests
                     DESTINATION ${CMAKE_INSTALL_BINDIR}/rocshmem/tests/unit
                     RENAME CTestTestfile.cmake)

        # Print summary
        get_directory_property(all_tests TESTS)
        list(LENGTH all_tests num_tests)
        message(STATUS "Registered ${num_tests} unit tests")
        message(STATUS "Run with: ctest -L unit")
        message(STATUS "Install-time CTest file: ${INSTALL_CTEST_OUTPUT}")
    else()
        message(WARNING "CTest generation files not found in sparse checkout - skipping CTest registration")
        set(BUILD_CTESTS OFF CACHE BOOL "CTests disabled due to missing files" FORCE)
    endif()
endif()

# For backward compatibility, the shell script is still available
# Usage: rocshmem_unit_driver.sh <binary> all
# Usage: rocshmem_unit_driver.sh <binary> custom <ranks> <filter>
# ASAN SETTINGS
###############################################################################
target_compile_options(
  ${PROJECT_NAME}
  PRIVATE
    $<$<BOOL:${ASAN}>:-fsanitize=address -g>
)

target_link_options(
  ${PROJECT_NAME}
  PRIVATE
    $<$<BOOL:${ASAN}>:-fsanitize=address -g -shared-libsan>
)
