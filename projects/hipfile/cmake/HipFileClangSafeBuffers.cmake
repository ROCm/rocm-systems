# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

#-----------------------------------------------------------------------------
# Warn about unsafe buffer operations (llvm C++ only)
#
# See: https://clang.llvm.org/docs/SafeBuffers.html for more info
#
# Fixing this cleanly requires C++20, so it's an option for now
#
# The flags are applied to C++ sources in hipfile_set_compiler_flags()
#
# This is OFF by default until the existing warnings are fixed
#-----------------------------------------------------------------------------
option(HIPFILE_WARN_UNSAFE_BUFFER_OPS "Warn about unsafe buffer operations (llvm C++ only)" OFF)

if(HIPFILE_WARN_UNSAFE_BUFFER_OPS AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "Unsafe buffer warnings are only useful for clang/llvm")
endif()
