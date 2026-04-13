# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

#-----------------------------------------------------------------------------
# Option to use the clang-tidy tool
#
# When this option is enabled, compilation will emit clang-tidy suggestions.
#-----------------------------------------------------------------------------
option(AIS_USE_CLANG_TIDY "Run clang-tidy when compiling" OFF)

if(AIS_USE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES clang-tidy REQUIRED)
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_EXE};-warnings-as-errors=*)
endif()
