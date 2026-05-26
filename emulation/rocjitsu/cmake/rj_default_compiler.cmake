# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Default to the ROCm LLVM clang/clang++ if available and the user hasn't
# specified a compiler via -DCMAKE_CXX_COMPILER or the CC/CXX env vars.
#
# The ROCm install provides two compiler entry points:
#   /opt/rocm/bin/amdclang++        → amdllvm wrapper binary
#   /opt/rocm/lib/llvm/bin/clang++  → direct symlink to clang-NN
#
# CMake's clang-scan-deps (C++20 module dependency scanning) is incompatible
# with the amdllvm wrapper — it cannot resolve resource headers through it.
# This module resolves amdclang++ to the sibling clang++ in the same LLVM
# directory so that both the compiler and the scanner work correctly.
#
# Usage: include(rj_default_compiler) before project().

# Resolve amdclang/amdclang++ to their LLVM directory siblings.
# Given /opt/rocm/bin/amdclang++ (symlink to ../lib/llvm/bin/amdllvm),
# find /opt/rocm/lib/llvm/bin/clang++ in the same directory.
function(_rj_resolve_rocm_compiler LANG COMPILER_VAR)
    if(NOT DEFINED ${COMPILER_VAR})
        return()
    endif()
    get_filename_component(_name "${${COMPILER_VAR}}" NAME)
    if(NOT _name MATCHES "^amd")
        return() # Not an amdclang variant; nothing to resolve.
    endif()
    # If the user passed just a name (e.g. "amdclang++"), resolve via PATH first.
    set(_compiler "${${COMPILER_VAR}}")
    if(NOT IS_ABSOLUTE "${_compiler}")
        find_program(_resolved "${_compiler}")
        if(_resolved)
            set(_compiler "${_resolved}")
        endif()
        unset(_resolved CACHE)
    endif()
    # Follow the symlink to find the real directory containing clang/clang++.
    get_filename_component(_real "${_compiler}" REALPATH)
    get_filename_component(_dir "${_real}" DIRECTORY)
    if(LANG STREQUAL "C")
        set(_target "clang")
    else()
        set(_target "clang++")
    endif()
    if(EXISTS "${_dir}/${_target}")
        set(${COMPILER_VAR}
            "${_dir}/${_target}"
            CACHE FILEPATH
            "${LANG} compiler"
            FORCE
        )
        message(STATUS "rocjitsu: resolved ${_name} → ${_dir}/${_target}")
    endif()
endfunction()

# Auto-detect ROCm LLVM compiler if user hasn't specified one.
if(NOT DEFINED CMAKE_C_COMPILER AND NOT DEFINED ENV{CC})
    find_program(
        _RJ_CC
        NAMES amdclang clang
        PATHS /opt/rocm/bin /opt/rocm/lib/llvm/bin
        ENV ROCM_PATH
        PATH_SUFFIXES bin lib/llvm/bin
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
        NAMES amdclang++ clang++
        PATHS /opt/rocm/bin /opt/rocm/lib/llvm/bin
        ENV ROCM_PATH
        PATH_SUFFIXES bin lib/llvm/bin
        NO_DEFAULT_PATH
    )
    if(_RJ_CXX)
        set(CMAKE_CXX_COMPILER "${_RJ_CXX}" CACHE FILEPATH "C++ compiler" FORCE)
        message(STATUS "rocjitsu: defaulting C++ compiler to ${_RJ_CXX}")
    endif()
    unset(_RJ_CXX CACHE)
endif()

# If the user explicitly passed amdclang/amdclang++, resolve to clang/clang++.
_rj_resolve_rocm_compiler(C CMAKE_C_COMPILER)
_rj_resolve_rocm_compiler(CXX CMAKE_CXX_COMPILER)
