# ##############################################################################
# Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ##############################################################################

# Setup unit testing env
# Handle HSA Interception Tool Tests
find_package(hsa-runtime64 REQUIRED CONFIG PATHS ${ROCM_PATH})

find_package(
    Clang REQUIRED CONFI
    PATHS "${ROCM_PATH}"
    PATH_SUFFIXES "llvm/lib/cmake/clang")

file(GLOB TEST_HSATOOl_SRC_FILES ${PROJECT_SOURCE_DIR}/tests-v2/HSAToolLibrary/*.cpp)

add_library(test_hsatool_library SHARED ${TEST_HSATOOl_SRC_FILES})

target_include_directories(test_hsatool_library PRIVATE ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(
    test_hsatool_library
    PROPERTIES CXX_VISIBILITY_PRESET hidden
    DEFINE_SYMBOL TEST_HSA_TOOL_EXPORTS
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests-v2
    INSTALL_RPATH "${ROCM_APPEND_PRIVLIB_RPATH}")

install(
    TARGETS test_hsatool_library LIBRARY
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests
    COMPONENT tests)

target_link_libraries(test_hsatool_library PRIVATE hsa-runtime64::hsa-runtime64)
