cmake_minimum_required(VERSION 3.18.4 FATAL_ERROR)

project(rocprofiler-systems-streams-example LANGUAGES CXX)

# This example requires hip and rocdecode.
find_package(HIP QUIET HINTS ${ROCmVersion_DIR} PATHS ${ROCmVersion_DIR})

if(NOT HIP_FOUND)
    message(WARNING "hip is not found. Skip streams example.")
    return()
endif()

message(STATUS "streams example: hip found: ${hip_DIR}")

if(NOT CMAKE_CXX_COMPILER_IS_HIPCC AND HIPCC_EXECUTABLE)
    if(
        CMAKE_CXX_COMPILER STREQUAL HIPCC_EXECUTABLE
        OR "${CMAKE_CXX_COMPILER}" MATCHES "hipcc"
    )
        set(CMAKE_CXX_COMPILER_IS_HIPCC 1 CACHE BOOL "HIP compiler")
    endif()
endif()

if(
    (
        NOT CMAKE_CXX_COMPILER_IS_HIPCC
        OR (NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT hip_FOUND)
    )
    AND (NOT COMMAND rocprofiler_systems_custom_compilation AND NOT HIPCC_EXECUTABLE)
)
    message(AUTHOR_WARNING "streams target could not be built")
    return()
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../cmake)
list(APPEND CMAKE_PREFIX_PATH ${ROCmVersion_DIR}/lib/cmake ${ROCmVersion_DIR})
list(APPEND CMAKE_PREFIX_PATH ${ROCmVersion_DIR}/hip ${ROCmVersion_DIR})

if(ROCPROFSYS_DISABLE_EXAMPLES)
    get_filename_component(_DIR ${CMAKE_CURRENT_LIST_DIR} NAME)

    if(
        ${PROJECT_NAME} IN_LIST ROCPROFSYS_DISABLE_EXAMPLES
        OR ${_DIR} IN_LIST ROCPROFSYS_DISABLE_EXAMPLES
    )
        return()
    endif()
endif()

find_program(
    HIPCC_EXECUTABLE
    NAMES hipcc
    HINTS ${ROCmVersion_DIR} ${ROCM_PATH}
    ENV ROCM_PATH
    /opt/rocm
    PATHS ${ROCmVersion_DIR} ${ROCM_PATH}
    ENV ROCM_PATH
    /opt/rocm
    NO_CACHE
)
mark_as_advanced(HIPCC_EXECUTABLE)

if(NOT HIPCC_EXECUTABLE)
    message(AUTHOR_WARNING "hipcc could not be found. Cannot build streams target")
    return()
endif()

if(NOT CMAKE_CXX_COMPILER_IS_HIPCC AND HIPCC_EXECUTABLE)
    if(
        CMAKE_CXX_COMPILER STREQUAL HIPCC_EXECUTABLE
        OR "${CMAKE_CXX_COMPILER}" MATCHES "hipcc"
    )
        set(CMAKE_CXX_COMPILER_IS_HIPCC 1 CACHE BOOL "HIP compiler")
    endif()
endif()

if(
    (
        NOT CMAKE_CXX_COMPILER_IS_HIPCC
        OR (NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT hip_FOUND)
    )
    AND (NOT COMMAND rocprofiler_systems_custom_compilation AND NOT HIPCC_EXECUTABLE)
)
    message(AUTHOR_WARNING "streams target could not be built")
    return()
endif()

find_package(Threads REQUIRED)

add_executable(streams streams.cpp)
target_link_libraries(streams PRIVATE Threads::Threads)

if(
    CMAKE_CXX_COMPILER_ID MATCHES "Clang"
    AND NOT CMAKE_CXX_COMPILER_IS_HIPCC
    AND NOT HIPCC_EXECUTABLE
)
    target_link_libraries(
        streams
        PRIVATE
            $<TARGET_NAME_IF_EXISTS:rocprofiler-systems::rocprofiler-systems-compile-options>
            $<TARGET_NAME_IF_EXISTS:hip::host>
            $<TARGET_NAME_IF_EXISTS:hip::device>
    )
else()
    target_compile_options(streams PRIVATE -W -Wall)
endif()

if("${CMAKE_BUILD_TYPE}" MATCHES "Release")
    target_compile_options(streams PRIVATE -g1)
endif()

if(NOT CMAKE_CXX_COMPILER_IS_HIPCC AND HIPCC_EXECUTABLE)
    # defined in MacroUtilities.cmake
    rocprofiler_systems_custom_compilation(COMPILER ${HIPCC_EXECUTABLE} TARGET streams)
endif()

if(ROCPROFSYS_INSTALL_EXAMPLES)
    install(TARGETS streams DESTINATION bin COMPONENT rocprofiler-systems-examples)
endif()
