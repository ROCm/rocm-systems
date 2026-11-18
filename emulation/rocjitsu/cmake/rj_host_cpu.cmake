# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Use the target platform, including when cross-compiling. Never infer the
# minimum supported CPU from the machine running CMake.
set(_rj_x86_64_linux OFF)
if(
    CMAKE_SYSTEM_NAME STREQUAL "Linux"
    AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$"
    AND CMAKE_SIZEOF_VOID_P EQUAL 8
)
    set(_rj_x86_64_linux ON)
endif()

set(_rj_host_cpu_default default)
if(
    _rj_x86_64_linux
    AND CMAKE_C_COMPILER_ID MATCHES "^(GNU|Clang)$"
    AND CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$"
)
    set(_rj_host_cpu_default x86-64-v3)
endif()
set(ROCJITSU_HOST_CPU_BASELINE
    "${_rj_host_cpu_default}"
    CACHE STRING
    "rocjitsu host CPU baseline (default, x86-64, x86-64-v3, x86-64-v4)"
)
set(_rj_host_cpu_values default x86-64 x86-64-v3 x86-64-v4)
set_property(
    CACHE ROCJITSU_HOST_CPU_BASELINE
    PROPERTY STRINGS ${_rj_host_cpu_values}
)
if(NOT ROCJITSU_HOST_CPU_BASELINE IN_LIST _rj_host_cpu_values)
    message(
        FATAL_ERROR
        "Unsupported ROCJITSU_HOST_CPU_BASELINE='${ROCJITSU_HOST_CPU_BASELINE}'; "
        "choose one of: ${_rj_host_cpu_values}"
    )
endif()

if(NOT ROCJITSU_HOST_CPU_BASELINE STREQUAL "default")
    if(NOT _rj_x86_64_linux)
        message(
            FATAL_ERROR
            "An explicit ROCJITSU_HOST_CPU_BASELINE requires a Linux x86-64 target."
        )
    endif()
    include(CheckCCompilerFlag)
    include(CheckCXXCompilerFlag)
    set(_rj_host_cpu_flag "-march=${ROCJITSU_HOST_CPU_BASELINE}")
    string(MAKE_C_IDENTIFIER "${ROCJITSU_HOST_CPU_BASELINE}" _rj_host_cpu_key)
    check_c_compiler_flag(
        "${_rj_host_cpu_flag}"
        _RJ_C_HOST_CPU_${_rj_host_cpu_key}
    )
    check_cxx_compiler_flag(
        "${_rj_host_cpu_flag}"
        _RJ_CXX_HOST_CPU_${_rj_host_cpu_key}
    )
    if(
        NOT _RJ_C_HOST_CPU_${_rj_host_cpu_key}
        OR NOT _RJ_CXX_HOST_CPU_${_rj_host_cpu_key}
    )
        message(
            FATAL_ERROR
            "The C and C++ compilers must support ${_rj_host_cpu_flag}; "
            "use ROCJITSU_HOST_CPU_BASELINE=default to retain toolchain flags."
        )
    endif()
    # Directory properties cover object libraries and header-only SIMD users,
    # including tests, without changing parent projects or fetched dependencies.
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:${_rj_host_cpu_flag}>")
    add_link_options("$<$<LINK_LANGUAGE:C,CXX>:${_rj_host_cpu_flag}>")
endif()
message(STATUS "rocjitsu host CPU baseline: ${ROCJITSU_HOST_CPU_BASELINE}")

unset(_rj_x86_64_linux)
unset(_rj_host_cpu_default)
unset(_rj_host_cpu_values)
unset(_rj_host_cpu_flag)
unset(_rj_host_cpu_key)
