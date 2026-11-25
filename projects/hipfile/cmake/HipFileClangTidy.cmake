# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

#-----------------------------------------------------------------------------
# Option to use the clang-tidy tool
#
# When this option is enabled, compilation will emit clang-tidy suggestions.
#
# HIPFILE_CLANG_TIDY_EXECUTABLE selects which clang-tidy to run. It is
# deliberately decoupled from CMAKE_CXX_COMPILER: the ROCm LLVM toolchain is
# built with CLANG_TIDY_ENABLE_STATIC_ANALYZER=OFF (see ROCm/TheRock
# compiler/CMakeLists.txt), so its clang-tidy ships with zero clang-analyzer-*
# checks and cannot run the Clang Static Analyzer at all. An upstream clang-tidy of a matching LLVM major
# version is used instead. Keep that version aligned with the ROCm LLVM major
# version -- mismatched clang-tidy versions report conflicting diagnostics.
#
# The check set is defined explicitly in projects/hipfile/.clang-tidy rather
# than relying on clang-tidy's built-in defaults, because those defaults are not
# stable across LLVM versions (LLVM 22 dropped clang-analyzer-*, which was
# the whole default set).
#-----------------------------------------------------------------------------
option(HIPFILE_USE_CLANG_TIDY "Run clang-tidy when compiling" OFF)
set(HIPFILE_CLANG_TIDY_EXECUTABLE "" CACHE STRING
    "clang-tidy executable to use (name or absolute path); empty searches PATH for clang-tidy")

# .clang-tidy uses ExcludeHeaderFilterRegex, which clang-tidy 19 added
set(HIPFILE_CLANG_TIDY_MIN_VERSION 19)

if(HIPFILE_USE_CLANG_TIDY)
    if(HIPFILE_CLANG_TIDY_EXECUTABLE)
        find_program(CLANG_TIDY_EXE NAMES "${HIPFILE_CLANG_TIDY_EXECUTABLE}" REQUIRED)
    else()
        find_program(CLANG_TIDY_EXE NAMES clang-tidy REQUIRED)
    endif()

    # Checked on every configure, since CLANG_TIDY_EXE is cached and can
    # be set by hand
    execute_process(
        COMMAND "${CLANG_TIDY_EXE}" --version
        RESULT_VARIABLE _hipfile_clang_tidy_result
        OUTPUT_VARIABLE _hipfile_clang_tidy_output
        ERROR_QUIET)
    set(_hipfile_clang_tidy_version "")
    if(_hipfile_clang_tidy_result EQUAL 0 AND _hipfile_clang_tidy_output MATCHES "LLVM version ([0-9]+)")
        set(_hipfile_clang_tidy_version "${CMAKE_MATCH_1}")
    endif()
    if(NOT _hipfile_clang_tidy_version)
        message(FATAL_ERROR
            "Can't get the version of clang-tidy from CLANG_TIDY_EXE (${CLANG_TIDY_EXE})")
    elseif(_hipfile_clang_tidy_version VERSION_LESS HIPFILE_CLANG_TIDY_MIN_VERSION)
        message(FATAL_ERROR
            "CLANG_TIDY_EXE (${CLANG_TIDY_EXE}) is clang-tidy ${_hipfile_clang_tidy_version}, "
            "but HIPFILE_USE_CLANG_TIDY needs ${HIPFILE_CLANG_TIDY_MIN_VERSION} or later. "
            "Set CLANG_TIDY_EXE to a newer one.")
    endif()
    unset(_hipfile_clang_tidy_result)
    unset(_hipfile_clang_tidy_output)
    unset(_hipfile_clang_tidy_version)

    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_EXE};-warnings-as-errors=*)
endif()
