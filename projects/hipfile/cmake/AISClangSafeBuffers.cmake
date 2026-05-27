# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

#-----------------------------------------------------------------------------
# Warn about unsafe buffer operations (llvm C++ only)
#
# See: https://clang.llvm.org/docs/SafeBuffers.html for more info
#
# Fixing this cleanly requires C++20, so it's an option for now
#-----------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(USE_UBO ON)
else()
    set(USE_UBO OFF)
endif()
option(AIS_WARN_UNSAFE_BUFFER_OPS "Warn about unsafe buffer operations (llvm C++ only)" ${USE_UBO})

if(AIS_WARN_UNSAFE_BUFFER_OPS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(AIS_CLANG_WARNING_FLAGS
            -Wunsafe-buffer-usage
            -fsafe-buffer-usage-suggestions
            ${AIS_CLANG_WARNING_FLAGS}
        )
    else()
        message(FATAL_ERROR "Unsafe buffer warnings are only useful for clang/llvm")
    endif()
endif()
