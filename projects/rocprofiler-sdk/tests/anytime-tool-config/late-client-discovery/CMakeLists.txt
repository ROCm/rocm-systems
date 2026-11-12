# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#
# Regression test: a tool library which only becomes visible after rocprofiler-sdk was
# loaded must still be configured, without the tool calling rocprofiler_force_configure().
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

if(NOT CMAKE_HIP_COMPILER)
    find_program(
        amdclangpp_EXECUTABLE
        NAMES amdclang++
        HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATH_SUFFIXES bin llvm/bin NO_CACHE)
    mark_as_advanced(amdclangpp_EXECUTABLE)

    if(amdclangpp_EXECUTABLE)
        set(CMAKE_HIP_COMPILER "${amdclangpp_EXECUTABLE}")
    endif()
endif()

project(rocprofiler-sdk-tests-late-client-discovery LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

find_package(rocprofiler-sdk REQUIRED)

# Tool library the test app dlopens. It is never preloaded and never force-configures
# itself, so it is only ever reached through rocprofiler-sdk's own client discovery.
add_library(late-client-discovery-client SHARED)
target_sources(late-client-discovery-client PRIVATE client.cpp)
target_link_libraries(
    late-client-discovery-client PRIVATE rocprofiler-sdk::rocprofiler-sdk
                                         rocprofiler-sdk::tests-build-flags)

# The app links HIP but issues no HIP call until the tool is loaded, which is what leaves
# the runtime and HSA API table registrations on the far side of the dlopen.
set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(late-client-discovery-testapp)
target_sources(late-client-discovery-testapp PRIVATE main.cpp)
target_link_libraries(late-client-discovery-testapp
                      PRIVATE ${CMAKE_DL_LIBS} rocprofiler-sdk::tests-build-flags)

# An inherited ROCPROFILER_REGISTER_LIBRARY is the reported trigger: it makes
# rocprofiler-register load rocprofiler-sdk without first checking that a client is
# visible, so the library initializes with zero clients before main() runs. The app
# asserts that state rather than assuming it, so this fails loudly if the variable ever
# stops implying a forced load.
rocprofiler_add_integration_execute_test(
    late-client-discovery-forced-load
    COMMAND $<TARGET_FILE:late-client-discovery-testapp>
            $<TARGET_FILE:late-client-discovery-client> 1
    DEPENDS late-client-discovery-client
    TIMEOUT 120
    ENVIRONMENT
        "ROCPROFILER_REGISTER_LIBRARY=$<TARGET_FILE:rocprofiler-sdk::rocprofiler-sdk-shared-library>"
    PASS_REGULAR_EXPRESSION "Test PASSED")

# Control: the same binary with nothing forcing an early load, so rocprofiler-sdk is
# brought in only once the dlopen'd tool has made rocprofiler_configure visible and client
# discovery succeeds the ordinary way. Distinguishes "recovered a late client" from "the
# app would have been traced regardless".
rocprofiler_add_integration_execute_test(
    late-client-discovery-natural-load
    COMMAND $<TARGET_FILE:late-client-discovery-testapp>
            $<TARGET_FILE:late-client-discovery-client> 0
    DEPENDS late-client-discovery-client
    TIMEOUT 120
    PASS_REGULAR_EXPRESSION "Test PASSED")
