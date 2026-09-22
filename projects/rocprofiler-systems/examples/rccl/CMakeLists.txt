# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
list(APPEND CMAKE_MESSAGE_CONTEXT "rccl-tests")

project(rocprofiler-systems-rccl-example LANGUAGES CXX)

# Support standalone builds
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(${CMAKE_CURRENT_LIST_DIR}/../cmake/standalone-helpers.cmake OPTIONAL)
endif()

# Find HIP
find_package(
    hip
    QUIET
    HINTS ${ROCmVersion_DIR} ${ROCM_PATH} /opt/rocm
    PATHS ${ROCmVersion_DIR} ${ROCM_PATH} /opt/rocm
)
if(NOT hip_FOUND)
    message(AUTHOR_WARNING "rccl-tests skipped. Missing HIP...")
    return()
endif()

# Find RCCL
find_package(
    rccl
    QUIET
    HINTS ${ROCmVersion_DIR} ${ROCM_PATH} /opt/rocm
    PATHS ${ROCmVersion_DIR} ${ROCM_PATH} /opt/rocm
)
if(NOT rccl_FOUND)
    message(AUTHOR_WARNING "rccl-tests skipped. Missing RCCL...")
    return()
endif()

# rccl-tests sources are written against the CUDA/NCCL API and must be converted
# to HIP with hipify-perl before they can be compiled.
find_program(
    HIPIFY_PERL_EXECUTABLE
    NAMES hipify-perl
    HINTS ${ROCmVersion_DIR} ${ROCM_PATH} $ENV{ROCM_PATH} /opt/rocm
    PATH_SUFFIXES bin
)
mark_as_advanced(HIPIFY_PERL_EXECUTABLE)
if(NOT HIPIFY_PERL_EXECUTABLE)
    message(AUTHOR_WARNING "rccl-tests skipped. hipify-perl could not be found.")
    return()
endif()

# HIP compiler detection (mirrors the other GPU examples, e.g. transpose). We
# avoid enable_language(HIP) and instead compile the HIP sources through hipcc
# via rocprofiler_systems_custom_compilation().
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
    message(AUTHOR_WARNING "hipcc could not be found. Cannot build rccl-tests")
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

find_package(Threads REQUIRED)

# ----------------------------------------------------------------------------
# Source layout
#
set(_rccl_src_dir "${CMAKE_CURRENT_SOURCE_DIR}/rccl-tests/src")
set(_rccl_verifiable_dir "${CMAKE_CURRENT_SOURCE_DIR}/rccl-tests/verifiable")
set(_rccl_hipify_dir "${CMAKE_CURRENT_BINARY_DIR}/hipify")
file(MAKE_DIRECTORY ${_rccl_hipify_dir})

# rccl-tests perf binaries. This is trimmed to the minimal set that still
# exercises every distinct RCCL (nccl*) API entry point: scatter/gather/sendrecv
# only add ncclSend/ncclRecv, which is already covered by alltoallv_perf, so they
# are omitted for API-tracing coverage purposes.
set(_rccl_tests
    all_reduce
    all_gather
    broadcast
    reduce_scatter
    reduce
    alltoall
    alltoallv
)

# Sources compiled into every perf binary (shared object library below)
set(_rccl_common_sources
    ${_rccl_src_dir}/common.cpp
    ${_rccl_src_dir}/timer.cc
    ${_rccl_verifiable_dir}/verifiable.cpp
)

# Headers needed by the sources (hipified so quoted includes resolve locally)
set(_rccl_headers
    ${_rccl_src_dir}/common.h
    ${_rccl_src_dir}/timer.h
    ${_rccl_src_dir}/git_version.h
    ${_rccl_src_dir}/rccl_compat.h
    ${_rccl_src_dir}/nccl1_compat.h
    ${_rccl_src_dir}/rccl_float8.h
    ${_rccl_verifiable_dir}/verifiable.h
)

# ----------------------------------------------------------------------------
# hipify step: convert each source/header to HIP and place the result (by base
# name) in a single directory so quoted includes resolve against the hipified
# copies.
#
set(_rccl_hipify_inputs ${_rccl_common_sources} ${_rccl_headers})
foreach(_test IN LISTS _rccl_tests)
    list(APPEND _rccl_hipify_inputs ${_rccl_src_dir}/${_test}.cpp)
endforeach()

set(_rccl_hipified_outputs)
foreach(_input IN LISTS _rccl_hipify_inputs)
    get_filename_component(_name ${_input} NAME)
    set(_output "${_rccl_hipify_dir}/${_name}")
    add_custom_command(
        OUTPUT ${_output}
        COMMAND ${HIPIFY_PERL_EXECUTABLE} -quiet-warnings ${_input} -o ${_output}
        MAIN_DEPENDENCY ${_input}
        COMMENT "Hipifying ${_name}"
        VERBATIM
    )
    list(APPEND _rccl_hipified_outputs ${_output})
endforeach()

add_custom_target(rccl-tests-hipify DEPENDS ${_rccl_hipified_outputs})

# ----------------------------------------------------------------------------
# git_version.cpp defines rcclTestsGitHash (referenced by common.cpp). It is
# plain host C++, so it does not go through hipify.
#
if(NOT GIT_EXECUTABLE)
    find_package(Git QUIET)
endif()
set(_rccl_git_rev "unknown")
if(GIT_EXECUTABLE)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} log --pretty=format:%h -n 1
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE _rccl_git_rev_out
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _rccl_git_ret
        ERROR_QUIET
    )
    if(_rccl_git_ret EQUAL 0 AND NOT _rccl_git_rev_out STREQUAL "")
        set(_rccl_git_rev "${_rccl_git_rev_out}")
    endif()
endif()
set(_rccl_git_version_cpp "${CMAKE_CURRENT_BINARY_DIR}/git_version.cpp")
file(
    WRITE ${_rccl_git_version_cpp}
    "const char* rcclTestsGitHash = \"rccl-tests:${_rccl_git_rev}\";\n"
)

# ----------------------------------------------------------------------------
# Apply the HIP compile/link settings used by the other GPU examples.
#
function(rccl_tests_setup_hip_target _target)
    foreach(_arch IN LISTS ROCPROFSYS_GFX_TARGETS)
        target_compile_options(${_target} PRIVATE --offload-arch=${_arch})
        target_link_options(${_target} PRIVATE --offload-arch=${_arch})
    endforeach()

    # HIPCC_EXECUTABLE is guaranteed to be set (early return otherwise), so the
    # sources are compiled through hipcc unless the main C++ compiler already is
    # hipcc.
    if(NOT CMAKE_CXX_COMPILER_IS_HIPCC)
        # defined in MacroUtilities.cmake (or standalone-helpers.cmake)
        rocprofiler_systems_custom_compilation(COMPILER ${HIPCC_EXECUTABLE} TARGET ${_target})
    endif()
endfunction()

# ----------------------------------------------------------------------------
# Object library with the sources shared by every perf binary.
#
add_library(
    rccl-tests-common
    OBJECT
    ${_rccl_hipify_dir}/common.cpp
    ${_rccl_hipify_dir}/timer.cc
    ${_rccl_hipify_dir}/verifiable.cpp
    ${_rccl_git_version_cpp}
)
add_dependencies(rccl-tests-common rccl-tests-hipify)
target_include_directories(rccl-tests-common PRIVATE ${_rccl_hipify_dir})
target_link_libraries(
    rccl-tests-common
    PRIVATE Threads::Threads roc::rccl ${CMAKE_DL_LIBS}
)
rccl_tests_setup_hip_target(rccl-tests-common)

# ----------------------------------------------------------------------------
# One executable per collective, linking the shared object library.
#
foreach(_test IN LISTS _rccl_tests)
    set(_target "${_test}_perf")
    add_executable(${_target} ${_rccl_hipify_dir}/${_test}.cpp)
    add_dependencies(${_target} rccl-tests-hipify)
    target_include_directories(${_target} PRIVATE ${_rccl_hipify_dir})
    target_link_libraries(
        ${_target}
        PRIVATE rccl-tests-common Threads::Threads roc::rccl ${CMAKE_DL_LIBS}
    )
    rccl_tests_setup_hip_target(${_target})

    if(ROCPROFSYS_INSTALL_EXAMPLES)
        install(
            TARGETS ${_target}
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples
            COMPONENT rocprofiler-systems-examples
        )
    endif()
endforeach()
