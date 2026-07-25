# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ------------------------------------------------------------------------------#
#
# rocprofiler-systems common headers
#
# ------------------------------------------------------------------------------#

add_library(rocprofiler-systems-common-library INTERFACE)
add_library(rocprofiler-systems::common-library ALIAS rocprofiler-systems-common-library)
add_library(
    rocprofiler-systems::rocprofiler-systems-common-library
    ALIAS rocprofiler-systems-common-library
)

# Query clang for host target triple for defines.h
# This is useful when LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON and
# the openmp libraries are installed one level deeper in lib/<host-target-triple>.
find_program(
    amdclang_EXECUTABLE
    NAMES amdclang
    HINTS ${ROCmVersion_DIR} ${ROCM_PATH}
    ENV ROCM_PATH
    /opt/rocm
    PATH_SUFFIXES bin llvm/bin lib/llvm/bin
)
mark_as_advanced(amdclang_EXECUTABLE)
set(LLVM_HOST_TRIPLE "")
if(amdclang_EXECUTABLE)
    execute_process(
        COMMAND "${amdclang_EXECUTABLE}" --print-target-triple
        OUTPUT_VARIABLE LLVM_HOST_TRIPLE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _llvm_host_triple_result
    )
    if(NOT _llvm_host_triple_result EQUAL 0)
        set(LLVM_HOST_TRIPLE "")
        rocprofiler_systems_message(
            FATAL_ERROR
            "'${amdclang_EXECUTABLE} --print-target-triple' failed (exit code ${_llvm_host_triple_result})"
            "This toolchain may be broken."
        )
    endif()
else()
    rocprofiler_systems_message(
        WARNING
        "Could not find amdclang to determine the LLVM host triple; "
        "libomptarget discovery for LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON may not work."
    )
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/defines.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/defines.h
    @ONLY
)

target_sources(
    rocprofiler-systems-common-library
    INTERFACE
        ${CMAKE_CURRENT_BINARY_DIR}/defines.h
        ${CMAKE_CURRENT_SOURCE_DIR}/delimit.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/environment.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/invoke.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/join.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/md5sum.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/path.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/setup.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/span.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/static_object.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/synchronized.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/traits.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/units.hpp
)

get_filename_component(COMMON_SOURCE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
get_filename_component(COMMON_BINARY_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}" DIRECTORY)

target_include_directories(
    rocprofiler-systems-common-library
    INTERFACE
        $<BUILD_INTERFACE:${COMMON_SOURCE_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${COMMON_BINARY_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/timemory/source>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/external/timemory/source>
)

target_compile_definitions(
    rocprofiler-systems-common-library
    INTERFACE $<BUILD_INTERFACE:ROCPROFSYS_INTERNAL_BUILD=1>
)

target_link_libraries(
    rocprofiler-systems-common-library
    INTERFACE $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-logger>
)

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()
