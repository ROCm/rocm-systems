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

include(gtest)

# ------------------------------------------------------------ #
# Helper function to add unit tests
# ------------------------------------------------------------ #

function(rocstorage_add_unit_test)
    cmake_parse_arguments(ARG "" "NAME" "SOURCES;LIBRARIES" ${ARGN})

    if(NOT ARG_NAME)
        message(FATAL_ERROR "rocstorage_add_unit_test: NAME is required")
    endif()

    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "rocstorage_add_unit_test: SOURCES is required")
    endif()

    set(TEST_TARGET "rocstorage_test_${ARG_NAME}")

    add_executable(${TEST_TARGET} ${ARG_SOURCES})

    target_link_libraries(
        ${TEST_TARGET}
        PRIVATE
            rocstorage
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
            ${ARG_LIBRARIES}
    )

    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)

    gtest_discover_tests(
        ${TEST_TARGET}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        PROPERTIES LABELS "unit"
    )
endfunction()

# ------------------------------------------------------------ #
# Unit Tests
# ------------------------------------------------------------ #

rocstorage_add_unit_test(
    NAME storage
    SOURCES storage_test.cpp
)

rocstorage_add_unit_test(
    NAME writer
    SOURCES writer_test.cpp
)

rocstorage_add_unit_test(
    NAME reader
    SOURCES reader_test.cpp
)
