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

project(rocdecdecode)

set (HIP_PLATFORM amd CACHE STRING "HIP platform")
find_package(HIP QUIET)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads QUIET)
find_package(FFmpeg QUIET)
find_package(rocdecode QUIET)

if(HIP_FOUND AND rocdecode_FOUND)
    # HIP
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} hip::host)

    # FFmpeg and rocdecode-host are optional
    # rocdecode-host depends on FFmpeg runtime libraries, so
    # find_package for rocdecode-host is enabled when FFmpeg is available
    if(FFMPEG_FOUND AND NOT USING_THE_ROCK)
      find_package(rocdecode-host 1.0.0 QUIET)
      # FFMPEG
      include_directories(${AVUTIL_INCLUDE_DIR} ${AVCODEC_INCLUDE_DIR}
                          ${AVFORMAT_INCLUDE_DIR})
      set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${FFMPEG_LIBRARIES})
    endif()
    if(rocdecode-host_FOUND AND NOT USING_THE_ROCK)
      # rocdecode-host
      include_directories(${rocdecode-host_INCLUDE_DIR})
      set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} rocdecode::rocdecode-host)
    endif()

    # STD Filesystem (Linux only - Windows has it built-in with C++17)
    if(IS_LINUX)
      set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} stdc++fs)
    endif()

    # rocdecode
    include_directories(${rocdecode_INCLUDE_DIR} ${ROCM_PATH}/lib)
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} rocdecode::rocdecode)

    # sample app exe
    list(APPEND SOURCES ${PROJECT_SOURCE_DIR} rocdecdecode.cpp)
    add_executable(${PROJECT_NAME} ${SOURCES})

    if(FFMPEG_FOUND AND rocdecode-host_FOUND AND NOT USING_THE_ROCK)
      target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_HOST_DECODE=1)
      message("-- rocdecode-host support enabled")
    else()
      target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_HOST_DECODE=0)
      message("-- rocdecode-host support disabled (FFmpeg or rocdecode-host not found)")
    endif()

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
endif()
