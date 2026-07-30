# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Auto-detect ROCm LLVM compiler if user hasn't specified one.
# Usage: include(rj_default_compiler) before project().
if(NOT DEFINED CMAKE_C_COMPILER AND NOT DEFINED ENV{CC})
    find_program(
        _RJ_CC
        NAMES amdclang
        PATHS ${ROCM_PATH}
        PATH_SUFFIXES lib/llvm/bin bin
        NO_DEFAULT_PATH
    )
    if(_RJ_CC)
        set(CMAKE_C_COMPILER "${_RJ_CC}" CACHE FILEPATH "C compiler" FORCE)
        message(STATUS "rocjitsu: defaulting C compiler to ${_RJ_CC}")
    endif()
    unset(_RJ_CC CACHE)
endif()

if(NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
    find_program(
        _RJ_CXX
        NAMES amdclang++
        PATHS ${ROCM_PATH}
        PATH_SUFFIXES lib/llvm/bin bin
        NO_DEFAULT_PATH
    )
    if(_RJ_CXX)
        set(CMAKE_CXX_COMPILER "${_RJ_CXX}" CACHE FILEPATH "C++ compiler" FORCE)
        message(STATUS "rocjitsu: defaulting C++ compiler to ${_RJ_CXX}")
    endif()
    unset(_RJ_CXX CACHE)
endif()
