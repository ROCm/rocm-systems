# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

# Set AMD Clang as default compiler for standalone builds
if(NOT DEFINED CMAKE_CXX_COMPILER)
    find_program(
        _AMDCLANG
        NAMES amdclang++
        HINTS ${ROCM_PATH} $ENV{ROCM_PATH} /opt/rocm
        PATH_SUFFIXES bin
    )
    if(_AMDCLANG)
        set(CMAKE_CXX_COMPILER ${_AMDCLANG})
    endif()
    unset(_AMDCLANG)
endif()

project(rocprofiler-systems-jpegdecode-example LANGUAGES CXX)

# Support standalone builds
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(${CMAKE_CURRENT_LIST_DIR}/../cmake/standalone-helpers.cmake OPTIONAL)
endif()

# This example requires hip and rocjpeg.
find_package(HIP QUIET)

if(NOT HIP_FOUND)
    message(WARNING "hip is not found. Skip jpegdecode example.")
    return()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../cmake)
list(APPEND CMAKE_PREFIX_PATH ${ROCmVersion_DIR}/lib/cmake ${ROCmVersion_DIR})
list(APPEND CMAKE_PREFIX_PATH ${ROCmVersion_DIR}/hip ${ROCmVersion_DIR})
list(APPEND CMAKE_MODULE_PATH ${ROCmVersion_DIR}/share/rocjpeg/cmake)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")
string(REPLACE " " ";" _FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")

# find rocJPEG - library and headers
find_path(
    rocjpeg_ROOT_DIR
    NAMES include/rocjpeg/rocjpeg.h
    HINTS ${ROCmVersion_DIR} ${ROCM_PATH}
    PATHS ${ROCmVersion_DIR} ${ROCM_PATH}
)

mark_as_advanced(rocjpeg_ROOT_DIR)

find_path(
    rocjpeg_INCLUDE_DIR
    NAMES rocjpeg/rocjpeg.h
    HINTS ${rocjpeg_ROOT_DIR}
    PATHS ${rocjpeg_ROOT_DIR}
    PATH_SUFFIXES include
)

find_library(
    rocjpeg_LIBRARY
    NAMES rocjpeg
    HINTS ${rocjpeg_ROOT_DIR}
    PATHS ${rocjpeg_ROOT_DIR}
    PATH_SUFFIXES lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    rocjpeg
    FOUND_VAR rocjpeg_FOUND
    REQUIRED_VARS rocjpeg_INCLUDE_DIR rocjpeg_LIBRARY
)

if(rocjpeg_FOUND)
    if(NOT TARGET rocjpeg::rocjpeg)
        add_library(rocjpeg::rocjpeg INTERFACE IMPORTED)
        target_link_libraries(rocjpeg::rocjpeg INTERFACE ${rocjpeg_LIBRARY})
        target_include_directories(rocjpeg::rocjpeg INTERFACE ${rocjpeg_INCLUDE_DIR})
    endif()
endif()

find_package(rocprofiler-register QUIET)

# Copy image files to build directory
function(copy_image_files_and_make_copies)
    if(EXISTS "${rocjpeg_ROOT_DIR}/share/rocjpeg/images")
        if(NOT EXISTS "${CMAKE_BINARY_DIR}/images")
            file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/images")
        endif()

        file(GLOB_RECURSE image_files "${rocjpeg_ROOT_DIR}/share/rocjpeg/images/*")
        file(COPY ${image_files} DESTINATION ${CMAKE_BINARY_DIR}/images)
        set(NUM_COPIES 20)

        # Loop over each file and make additional copies.
        # configure_file with COPYONLY treats the destination as a file path,
        # unlike file(COPY ... DESTINATION dir) which always creates a directory.
        foreach(file ${image_files})
            get_filename_component(filename ${file} NAME_WE)
            foreach(i RANGE 1 ${NUM_COPIES})
                configure_file(
                    "${file}"
                    "${CMAKE_BINARY_DIR}/images/${filename}_copy${i}.jpg"
                    COPYONLY
                )
            endforeach()
        endforeach()
    else()
        message(
            AUTHOR_WARNING
            "Source directory ${rocjpeg_ROOT_DIR}/share/rocjpeg/images does not exist; "
            "falling back to bundled sample images"
        )
        # Fall back to the sample images bundled with rocprofiler-systems so that
        # the jpeg-decode tests can run even on non-test ROCm builds that do not
        # ship the rocjpeg image dataset.
        set(_fallback_dir "${CMAKE_CURRENT_LIST_DIR}/sample_images")
        if(EXISTS "${_fallback_dir}")
            file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/images")
            file(GLOB_RECURSE _fallback_files "${_fallback_dir}/*")
            file(COPY ${_fallback_files} DESTINATION "${CMAKE_BINARY_DIR}/images")
            set(NUM_COPIES 20)
            foreach(_file ${_fallback_files})
                get_filename_component(_fname ${_file} NAME_WE)
                foreach(i RANGE 1 ${NUM_COPIES})
                    configure_file(
                        "${_file}"
                        "${CMAKE_BINARY_DIR}/images/${_fname}_copy${i}.jpg"
                        COPYONLY
                    )
                endforeach()
            endforeach()
        else()
            message(
                AUTHOR_WARNING
                "Bundled sample images directory ${_fallback_dir} also does not exist; "
                "jpeg-decode tests will be skipped due to missing image data"
            )
        endif()
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
    message(STATUS "RocJPEG library found: ${rocjpeg_LIBRARY}")
    include_directories(${rocjpeg_INCLUDE_DIR})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} rocjpeg::rocjpeg)
    list(APPEND SOURCES ${PROJECT_SOURCE_DIR} jpegdecodeperf.cpp)
    add_executable(jpegdecode ${SOURCES})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17")
    target_link_libraries(jpegdecode ${LINK_LIBRARY_LIST})
    target_compile_options(jpegdecode PRIVATE ${_FLAGS})
    copy_image_files_and_make_copies()

    if(ROCPROFSYS_INSTALL_EXAMPLES AND TARGET jpegdecode)
        install(
            TARGETS jpegdecode
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples
            COMPONENT rocprofiler-systems-examples
        )
        install(
            DIRECTORY ${CMAKE_BINARY_DIR}/images/
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples/images
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
        message(WARNING "-- ERROR!: Threads Not Found! - please install Threads!")
    endif()
    if(NOT rocprofiler-register_FOUND)
        message(
            WARNING
            "-- ERROR!: rocprofiler-register Not Found! - please install rocprofiler-register!"
        )
    endif()
endif()
