# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

#-----------------------------------------------------------------------------
# Option to use the clang-tidy tool
#
# When this option is enabled, compilation will emit clang-tidy suggestions.
#
# AIS_CLANG_TIDY_EXECUTABLE selects which clang-tidy to run. It is deliberately
# decoupled from CMAKE_CXX_COMPILER: the ROCm LLVM toolchain is built with
# CLANG_TIDY_ENABLE_STATIC_ANALYZER=OFF (see ROCm/TheRock compiler/CMakeLists.txt),
# so its clang-tidy ships with zero clang-analyzer-* checks and cannot run the
# Clang Static Analyzer at all. An upstream clang-tidy of a matching LLVM major
# version is used instead. Keep that version aligned with the ROCm LLVM major
# version -- mismatched clang-tidy versions report conflicting diagnostics.
#
# The check set is defined explicitly in projects/hipfile/.clang-tidy rather
# than relying on clang-tidy's built-in defaults, because those defaults are not
# stable across LLVM versions (see the comment in that file).
#-----------------------------------------------------------------------------
option(AIS_USE_CLANG_TIDY "Run clang-tidy when compiling" OFF)
set(AIS_CLANG_TIDY_EXECUTABLE "" CACHE STRING
    "clang-tidy executable to use (name or absolute path); empty searches PATH for clang-tidy")

if(AIS_USE_CLANG_TIDY)
    if(AIS_CLANG_TIDY_EXECUTABLE)
        find_program(CLANG_TIDY_EXE NAMES "${AIS_CLANG_TIDY_EXECUTABLE}" REQUIRED)
    else()
        find_program(CLANG_TIDY_EXE NAMES clang-tidy REQUIRED)
    endif()
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_EXE};-warnings-as-errors=*)
endif()
