# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

#-----------------------------------------------------------------------------
# Warn about unsafe buffer operations and harden the C++ standard library
# (llvm C++ only)
#
# See: https://clang.llvm.org/docs/SafeBuffers.html for more info
#
# The warnings steer code toward bounds-aware types like std::span, and the
# standard library hardening (_GLIBCXX_ASSERTIONS for libstdc++,
# _LIBCPP_HARDENING_MODE for libc++) makes those types check their bounds
# at run time. Each library ignores the other's macro.
#
# Fixing this cleanly requires C++20, so it's an option for now
#
# The flags are applied to C++ sources in hipfile_set_compiler_flags(),
# except for targets created with NO_SAFE_BUFFERS (the examples, which are
# also built as C++17 when installed and can't use std::span)
#
# This is OFF by default until the existing warnings are fixed
#-----------------------------------------------------------------------------
option(HIPFILE_WARN_UNSAFE_BUFFER_OPS
    "Warn about unsafe buffer operations and harden the C++ standard library (llvm C++ only)" OFF)

if(HIPFILE_WARN_UNSAFE_BUFFER_OPS AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "Unsafe buffer warnings are only useful for clang/llvm")
endif()
