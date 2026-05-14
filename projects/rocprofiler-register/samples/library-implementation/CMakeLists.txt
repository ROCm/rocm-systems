#
#
#
cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

# WINDOWS-DIVERGENCE: -W -Wall -Wextra are GCC/Clang flags. MSVC's cl.exe
# rejects them with D9002 ("ignoring unknown option"); clang-cl accepts them
# silently but only as a partial fallback. CMAKE_CXX_FLAGS_INIT is consulted
# before the CXX language is enabled, so it must be set without using the
# CXX_COMPILER_ID generator-expression machinery -- gate on platform here and
# defer the MSVC equivalent (/W4) to per-target compile options if/when it
# becomes desired.
if(NOT WIN32)
    set(CMAKE_CXX_FLAGS_INIT "-W -Wall -Wextra")
endif()

project(rocprofiler-register-library-implementation LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(rocprofiler-register REQUIRED)

add_library(rocp-reg-hip-demo SHARED)
target_sources(rocp-reg-hip-demo PRIVATE demo.cpp demo.hpp)
target_link_libraries(rocp-reg-hip-demo
                      PRIVATE rocprofiler-register::rocprofiler-register)
