################################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc.
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

cmake_minimum_required(VERSION 3.18.4 FATAL_ERROR)

# This example requires hip and rocjpeg.
find_package(HIP QUIET)

if(NOT HIP_FOUND)
    message(WARNING "hip is not found. Skip jpegdecode example.")
    return()
endif()

# Set AMD Clang as default compiler
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_C_COMPILER ${ROCmVersion_DIR}/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCmVersion_DIR}/bin/amdclang++)
endif()

project(rocprofiler-systems-jpegdecode-example)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../cmake)
list(APPEND CMAKE_PREFIX_PATH ${ROCmVersion_DIR}/lib/cmake ${ROCmVersion_DIR})
list(APPEND CMAKE_PREFIX_PATH ${ROCmVersion_DIR}/hip ${ROCmVersion_DIR})
list(APPEND CMAKE_MODULE_PATH ${ROCmVersion_DIR}/share/rocjpeg/cmake)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")
string(REPLACE " " ";" _FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")

if(ROCPROFSYS_DISABLE_EXAMPLES)
    get_filename_component(_DIR ${CMAKE_CURRENT_LIST_DIR} NAME)

    if(
        ${PROJECT_NAME} IN_LIST ROCPROFSYS_DISABLE_EXAMPLES
        OR ${_DIR} IN_LIST ROCPROFSYS_DISABLE_EXAMPLES
    )
        return()
    endif()
endif()

find_package(rocjpeg QUIET)
find_package(rocprofiler-register QUIET)

# Copy image files to build directory
function(copy_image_files_and_make_copies)
    if(EXISTS "${ROCmVersion_DIR}/share/rocjpeg/images")
        if(NOT EXISTS "${CMAKE_BINARY_DIR}/images")
            file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/images")
        endif()

        file(GLOB_RECURSE image_files "${ROCmVersion_DIR}/share/rocjpeg/images/*")
        file(COPY ${image_files} DESTINATION ${CMAKE_BINARY_DIR}/images)
        set(NUM_COPIES 20)

        # Loop over each file and make additional copies
        foreach(file ${image_files})
            get_filename_component(filename ${file} NAME)
            foreach(i RANGE 1 ${NUM_COPIES})
                file(
                    COPY ${file}
                    DESTINATION ${CMAKE_BINARY_DIR}/images/${filename}_copy${i}.jpg
                )
            endforeach()
        endforeach()
    else()
        message(
            AUTHOR_WARNING
            "Source directory ${ROCmVersion_DIR}/share/rocjpeg/images does not exist"
        )
    endif()
endfunction()

# threads
find_package(Threads REQUIRED)

if(HIP_FOUND AND rocjpeg_FOUND AND Threads_FOUND AND rocprofiler-register_FOUND)
    # HIP
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} hip::host)
    # threads
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} Threads::Threads)
    # std filesystem
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} stdc++fs)
    # rocprofiler-register
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} rocprofiler-register::rocprofiler-register)

    # rocJPEG
    message(STATUS "RocJPEG library found: ${rocjpeg_LIBRARIES}")
    include_directories(${rocjpeg_INCLUDE_DIR})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} rocjpeg::rocjpeg)
    list(APPEND SOURCES ${PROJECT_SOURCE_DIR} jpegdecodeperf.cpp)
    add_executable(jpegdecode ${SOURCES})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17")
    target_link_libraries(jpegdecode ${LINK_LIBRARY_LIST})
    target_compile_options(jpegdecode PRIVATE ${_FLAGS})
    copy_image_files_and_make_copies()

    if(ROCPROFSYS_INSTALL_EXAMPLES)
        install(TARGETS jpegdecode DESTINATION bin COMPONENT rocprofiler-systems-examples)
        install(
            FILES ${CMAKE_BINARY_DIR}/images
            DESTINATION share/rocprofiler-systems/tests/images
            COMPONENT rocprofiler-systems-examples
        )
    endif()
else()
    message(
        "-- ERROR!: ${PROJECT_NAME} excluded! please install all the dependencies and try again!"
    )
    if(NOT HIP_FOUND)
        message(FATAL_ERROR "-- ERROR!: HIP Not Found! - please install ROCm and HIP!")
    endif()
    if(NOT rocjpeg_FOUND)
        message(WARNING "-- ERROR!: rocJPEG Not Found! - please install rocJPEG!")
    endif()
    if(NOT Threads_FOUND)
        message(FATAL_ERROR "-- ERROR!: Threads Not Found! - please insatll Threads!")
    endif()
    if(NOT rocprofiler-register_FOUND)
        message(
            FATAL_ERROR
            "-- ERROR!: rocprofiler-register Not Found! - please install rocprofiler-register!"
        )
    endif()
endif()
