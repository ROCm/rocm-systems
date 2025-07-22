cmake_minimum_required(VERSION 3.18.4 FATAL_ERROR)

project(rocprofiler-systems-videodecode-example LANGUAGES CXX)

# This example requires hip and rocdecode.
find_package(HIP QUIET)

if(NOT HIP_FOUND)
    message(WARNING "hip is not found. Skip videodecode example.")
    return()
endif()

message(STATUS "hip found: ${hip_DIR}")

# Set AMD Clang as default compiler
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER ${ROCmVersion_DIR}/bin/amdclang++)
endif()

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../cmake)
list(APPEND CMAKE_PREFIX_PATH ${ROCmVersion_DIR}/lib/cmake ${ROCmVersion_DIR})
list(APPEND CMAKE_PREFIX_PATH ${ROCmVersion_DIR}/hip ${ROCmVersion_DIR})
list(APPEND CMAKE_MODULE_PATH ${ROCmVersion_DIR}/share/rocdecode/cmake)

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

function(videodecode_message _MSG_TYPE)
    if(
        "${CMAKE_PROJECT_NAME}" STREQUAL "rocprofiler-systems"
        AND "$ENV{ROCPROFSYS_CI}"
        AND "${_MSG_TYPE}" MATCHES "WARNING"
    )
        set(_MSG_TYPE STATUS) # don't generate warnings during CI
    endif()
    if("${CMAKE_PROJECT_NAME}" STREQUAL "rocprofiler-systems")
        rocprofiler_systems_message(${_MSG_TYPE} ${ARGN})
    else()
        message(${_MSG_TYPE} ${ARGN})
    endif()
endfunction()

# Find RocDecode
find_package(rocdecode QUIET)
if(NOT rocdecode_FOUND)
    videodecode_message(AUTHOR_WARNING "${PROJECT_NAME} skipped. Missing RocDecode...")
    return()
endif()

# Copy video files to build directory
if(EXISTS "${ROCmVersion_DIR}/share/rocdecode/video")
    if(NOT EXISTS "${CMAKE_BINARY_DIR}/videos")
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/videos")
    endif()

    file(GLOB_RECURSE video_files "${ROCmVersion_DIR}/share/rocdecode/video/*H26*.mp4")
    file(COPY ${video_files} DESTINATION ${CMAKE_BINARY_DIR}/videos)
else()
    videodecode_message(
        AUTHOR_WARNING
        "Source directory ${ROCmVersion_DIR}/share/rocdecode/video does not exist"
    )
endif()

# Find FFMPEG
find_package(FFmpeg)
if(NOT FFMPEG_FOUND)
    videodecode_message(AUTHOR_WARNING "${PROJECT_NAME} skipped. Missing FFMPEG...")
    return()
endif()

message(STATUS "FFMPEG libraries: ${FFMPEG_LIBRARIES}")
message(STATUS "FFMPEG AVFORMAT version: ${_FFMPEG_AVFORMAT_VERSION}")
message(STATUS "FFMPEG AVCODEC version: ${_FFMPEG_AVCODEC_VERSION}")
message(STATUS "FFMPEG AVUTIL version: ${_FFMPEG_AVUTIL_VERSION}")

find_path(
    FFMPEG_INCLUDE_DIR
    NAMES libavcodec/avcodec.h libavformat/avformat.h libavutil/avutil.h
    PATHS ${FFMPEG_INCLUDE_DIRS}
    PATH_SUFFIXES ffmpeg libav
)
find_library(AVCODEC_LIBRARY NAMES avcodec PATHS ${FFMPEG_LIBRARY_DIRS})
find_library(AVFORMAT_LIBRARY NAMES avformat PATHS ${FFMPEG_LIBRARY_DIRS})
find_library(AVUTIL_LIBRARY NAMES avutil PATHS ${FFMPEG_LIBRARY_DIRS})

set(FFMPEG_LIBRARIES ${AVCODEC_LIBRARY} ${AVFORMAT_LIBRARY} ${AVUTIL_LIBRARY})
set(FFMPEG_INCLUDE_DIRS ${FFMPEG_INCLUDE_DIR})

mark_as_advanced(FFMPEG_INCLUDE_DIR AVCODEC_LIBRARY AVFORMAT_LIBRARY AVUTIL_LIBRARY)

if(FFMPEG_FOUND AND rocdecode_FOUND)
    # HIP
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} hip::host)

    # FFMPEG
    include_directories(${FFMPEG_INCLUDE_DIR})
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${FFMPEG_LIBRARIES})

    # filesystem: c++ compilers less than equal to 8.5 need explicit link with stdc++fs
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS_EQUAL "8.5")
        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} stdc++fs)
    endif()

    # rocDecode
    include_directories(${rocdecode_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} rocdecode::rocdecode)

    # Threads
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} Threads::Threads)

    add_executable(videodecode videodecodebatch.cpp roc_video_dec.cpp)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17")
    target_link_libraries(videodecode ${LINK_LIBRARY_LIST})
    target_compile_options(videodecode PRIVATE ${_FLAGS})

    # FFMPEG multi-version support
    if(_FFMPEG_AVCODEC_VERSION VERSION_LESS_EQUAL 58.134.100)
        target_compile_definitions(videodecode PUBLIC USE_AVCODEC_GREATER_THAN_58_134=0)
    else()
        target_compile_definitions(videodecode PUBLIC USE_AVCODEC_GREATER_THAN_58_134=1)
    endif()

    if(ROCPROFSYS_INSTALL_EXAMPLES)
        install(
            TARGETS videodecode
            DESTINATION bin
            COMPONENT rocprofiler-systems-examples
        )
        install(
            FILES ${CMAKE_BINARY_DIR}/videos
            DESTINATION share/rocprofiler-systems/tests/videos
            COMPONENT rocprofiler-systems-examples
        )
    endif()
else()
    message(
        "-- ERROR!: videodecode excluded! please install all the dependencies and try again!"
    )
    if(NOT FFMPEG_FOUND)
        message(WARNING "-- ERROR!: FFMPEG Not Found! - please install FFMPEG!")
    endif()
    if(NOT rocdecode_FOUND)
        message(WARNING "-- ERROR!: rocDecode Not Found! - please install rocDecode!")
    endif()
endif()
