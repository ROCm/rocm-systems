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

include_guard(GLOBAL)

if(ROCPROFSYS_BUILD_GTEST)
    message(STATUS "Setting up GTEST to build from source!")

    rocprofiler_systems_checkout_git_submodule(
        RELATIVE_PATH external/googletest
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        TEST_FILE CMakeLists.txt
        REPO_URL https://github.com/google/googletest.git
        REPO_BRANCH "v1.17.0"
    )

    set(_GTEST_INSTALL_DIR ${PROJECT_BINARY_DIR}/external/googletest/install)

    # Configure GoogleTest
    execute_process(
        COMMAND
            ${CMAKE_COMMAND} -B ${PROJECT_BINARY_DIR}/external/googletest/build -G
            ${CMAKE_GENERATOR} -S ${PROJECT_SOURCE_DIR}/external/googletest
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_INSTALL_PREFIX=${_GTEST_INSTALL_DIR}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DBUILD_GMOCK=ON -DINSTALL_GTEST=ON
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/external/googletest
        RESULT_VARIABLE _GTEST_CONFIGURE_RET
        ERROR_VARIABLE _GTEST_CONFIGURE_ERR
        OUTPUT_QUIET
    )

    if(NOT _GTEST_CONFIGURE_RET EQUAL 0)
        message(FATAL_ERROR "Failed to configure GoogleTest: ${_GTEST_CONFIGURE_ERR}")
    endif()
    message(STATUS "GoogleTest configured successfully!")

    set(_GTEST_LIB_DIR "${_GTEST_INSTALL_DIR}/lib")

    # Build and install GoogleTest
    add_custom_target(
        rocprofiler-systems-googletest-library-build
        ALL
        COMMAND
            ${CMAKE_COMMAND} --build ${PROJECT_BINARY_DIR}/external/googletest/build
            --config ${CMAKE_BUILD_TYPE}
        COMMAND
            ${CMAKE_COMMAND} --install ${PROJECT_BINARY_DIR}/external/googletest/build
            --config ${CMAKE_BUILD_TYPE}
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/external/googletest/build
        COMMENT "Building and installing GoogleTest"
        BYPRODUCTS
            ${PROJECT_BINARY_DIR}/external/googletest/install/lib/libgtest.a
            ${PROJECT_BINARY_DIR}/external/googletest/install/lib/libgtest_main.a
            ${PROJECT_BINARY_DIR}/external/googletest/install/lib/libgmock.a
            ${PROJECT_BINARY_DIR}/external/googletest/install/lib/libgmock_main.a
    )

    add_library(rocprofiler-systems-gtest STATIC IMPORTED GLOBAL)
    add_library(rocprofiler-systems-gtest_main STATIC IMPORTED GLOBAL)
    add_library(rocprofiler-systems-gmock STATIC IMPORTED GLOBAL)

    set_target_properties(
        rocprofiler-systems-gtest
        PROPERTIES
            IMPORTED_LOCATION
                ${PROJECT_BINARY_DIR}/external/googletest/install/lib/libgtest.a
            INTERFACE_INCLUDE_DIRECTORIES
                ${PROJECT_BINARY_DIR}/external/googletest/install/include
    )

    set_target_properties(
        rocprofiler-systems-gtest_main
        PROPERTIES
            IMPORTED_LOCATION
                ${PROJECT_BINARY_DIR}/external/googletest/install/lib/libgtest_main.a
            INTERFACE_INCLUDE_DIRECTORIES
                ${PROJECT_BINARY_DIR}/external/googletest/install/include
    )

    set_target_properties(
        rocprofiler-systems-gmock
        PROPERTIES
            IMPORTED_LOCATION
                ${PROJECT_BINARY_DIR}/external/googletest/install/lib/libgmock.a
            INTERFACE_INCLUDE_DIRECTORIES
                ${PROJECT_BINARY_DIR}/external/googletest/install/include
    )

    add_dependencies(
        rocprofiler-systems-gtest
        rocprofiler-systems-googletest-library-build
    )
    add_dependencies(
        rocprofiler-systems-gtest_main
        rocprofiler-systems-googletest-library-build
    )
    add_dependencies(
        rocprofiler-systems-gmock
        rocprofiler-systems-googletest-library-build
    )

    # Create interface library
    add_library(rocprofiler-systems-googletest-library INTERFACE)

    target_link_libraries(
        rocprofiler-systems-googletest-library
        INTERFACE
            rocprofiler-systems-gtest
            rocprofiler-systems-gtest_main
            rocprofiler-systems-gmock
    )

    # GoogleTest requires threading support
    find_package(Threads REQUIRED)
    target_link_libraries(
        rocprofiler-systems-googletest-library
        INTERFACE Threads::Threads
    )

    message(STATUS "GoogleTest install directory: ${_GTEST_INSTALL_DIR}")
else()
    message(STATUS "Using system GTest library")
    find_package(GTest REQUIRED)
    add_library(rocprofiler-systems-googletest-library INTERFACE)

    # Link against system GTest targets
    target_link_libraries(
        rocprofiler-systems-googletest-library
        INTERFACE GTest::gtest GTest::gtest_main
    )

    # Also link gmock if available
    if(TARGET GTest::gmock)
        target_link_libraries(
            rocprofiler-systems-googletest-library
            INTERFACE GTest::gmock
        )
    endif()
endif()
