################################################################################
# Copyright (c) 2023 - 2026 Advanced Micro Devices, Inc.
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
################################################################################

cmake_minimum_required(VERSION 3.10)

# Include common sample configuration (must be before project())
include(${CMAKE_CURRENT_SOURCE_DIR}/../SampleCommon.cmake)

project(videodecoderaw)

set (HIP_PLATFORM amd CACHE STRING "HIP platform")
find_package(HIP QUIET)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads QUIET)
find_package(rocdecode QUIET)
find_package(rocprofiler-register QUIET)

# Check dependencies based on platform
if(IS_WINDOWS)
    # Windows: rocprofiler-register is optional
    set(DEPENDENCIES_MET FALSE)
    if(HIP_FOUND AND rocdecode_FOUND)
        set(DEPENDENCIES_MET TRUE)
    endif()
else()
    # Linux: rocprofiler-register is required
    set(DEPENDENCIES_MET FALSE)
    if(HIP_FOUND AND rocdecode_FOUND AND rocprofiler-register_FOUND)
        set(DEPENDENCIES_MET TRUE)
    endif()
endif()

if(DEPENDENCIES_MET)
    # HIP
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} hip::host)
    # rocdecode and utils
    include_directories (${rocdecode_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../../utils ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/rocvideodecode)
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} rocdecode::rocdecode)

    # rocprofiler-register (required on Linux, optional on Windows)
    if(rocprofiler-register_FOUND)
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} rocprofiler-register::rocprofiler-register)
        message("-- rocprofiler-register found, profiling support enabled")
    else()
        if(IS_WINDOWS)
            message("-- rocprofiler-register not found, profiling support disabled (optional on Windows)")
        endif()
    endif()

    # sample app exe
    list(APPEND SOURCES ${PROJECT_SOURCE_DIR} videodecoderaw.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/rocvideodecode/roc_video_dec.cpp)
    add_executable(${PROJECT_NAME} ${SOURCES})
    target_link_libraries(${PROJECT_NAME} ${LINK_LIBRARY_LIST})

    message("-- ${PROJECT_NAME} configured successfully")
else()
    message("-- ${PROJECT_NAME} excluded due to unmet dependencies:")
    if (NOT HIP_FOUND)
        message("   - HIP not found")
    endif()
    if (NOT rocdecode_FOUND)
        message("   - rocdecode not found (run 'ninja install' in main build directory)")
    endif()
    if (IS_LINUX AND NOT rocprofiler-register_FOUND)
        message("   - rocprofiler-register not found (required on Linux)")
    endif()
endif()
