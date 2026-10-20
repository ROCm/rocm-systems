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

# rdc_race_tests: tests that drive RDC sources from several threads under
# ThreadSanitizer. Each race test compiles the RDC sources it drives into this
# target with -fsanitize=thread (the sanity test drives no RDC source);
# GoogleTest is not instrumented, and no RDC library, gRPC or protobuf is
# linked. Built with RDC_BUILD_RACE_TESTS=ON, run by CI, never installed.

# GoogleTest comes from tests/unit, which the top-level CMakeLists.txt adds
# first.
if(NOT TARGET GTest::gtest)
    message(FATAL_ERROR "tests/race needs GoogleTest from tests/unit")
endif()
find_package(Threads REQUIRED)

add_executable(rdc_race_tests ${CMAKE_CURRENT_SOURCE_DIR}/test_race_sanity.cc)
# RDC's headers, for the sources the race tests compile in.
target_include_directories(rdc_race_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_options(rdc_race_tests PRIVATE -fsanitize=thread -fno-omit-frame-pointer)
target_link_options(rdc_race_tests PRIVATE -fsanitize=thread)
target_link_libraries(rdc_race_tests PRIVATE GTest::gtest Threads::Threads)
add_test(NAME rdc_race_tests COMMAND rdc_race_tests)

# Each race test adds its test file and the RDC sources it drives below, in a
# target_sources() block of its own.

# RdcModuleMgrImpl makes its telemetry and diagnostic modules on first use, and
# rdcd's clients can ask for them from several threads at once. Compiles the
# module manager, the two modules it makes, the library loader and the logger,
# with stand-ins for the AMD SMI, ROCr and ROCp modules in the test file; CMake
# compiles a source that another block lists too only once.
target_sources(
    rdc_race_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/test_module_mgr_race.cc
        ${PROJECT_SOURCE_DIR}/rdc_libs/rdc/src/RdcModuleMgrImpl.cc
        ${PROJECT_SOURCE_DIR}/rdc_libs/rdc/src/RdcTelemetryModule.cc
        ${PROJECT_SOURCE_DIR}/rdc_libs/rdc/src/RdcDiagnosticModule.cc
        ${PROJECT_SOURCE_DIR}/rdc_libs/bootstrap/src/RdcLibraryLoader.cc
        ${PROJECT_SOURCE_DIR}/rdc_libs/bootstrap/src/RdcLogger.cc
)
target_include_directories(rdc_race_tests PRIVATE ${PROJECT_SOURCE_DIR} ${AMD_SMI_INCLUDE_DIR})
target_link_libraries(rdc_race_tests PRIVATE ${CMAKE_DL_LIBS})
