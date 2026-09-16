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

#
# A mock "second" OMPT tool, standing in for TAU or Score-P, for the OMPT keep-or-defer
# coexistence tests. It contains no OpenMP code, so it builds with the default compiler
# rather than OMP_TARGET_COMPILER. The tests that drive it are registered in
# tests/rocprofv3/ompt-coexist.
#
add_library(openmp-tools-coexist-mock SHARED coexist_ompt_tool.cpp)
# The mock stands in for a third-party tool, so it compiles against the public headers
# alone and resolves rocprofiler_ompt_start_tool with dlsym at run time. Taking the
# include directories rather than linking rocprofiler-sdk-headers keeps the SDK's internal
# build definitions off it, so it compiles the same way in-tree as it does against an
# installed SDK.
target_include_directories(
    openmp-tools-coexist-mock SYSTEM
    PRIVATE
        $<TARGET_PROPERTY:rocprofiler-sdk::rocprofiler-sdk,INTERFACE_INCLUDE_DIRECTORIES>)
# links dl for dlsym(3); CMAKE_DL_LIBS is empty where libdl is folded into libc
target_link_libraries(openmp-tools-coexist-mock PRIVATE rocprofiler-sdk::tests-build-flags
                                                        ${CMAKE_DL_LIBS})
set_target_properties(openmp-tools-coexist-mock PROPERTIES CXX_VISIBILITY_PRESET hidden)
