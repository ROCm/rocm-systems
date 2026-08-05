# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Generate a version header from the project version so C++ code and the CLI
# can report the version at runtime.  The template lives next to this module;
# the generated header goes into the build tree's include directory.

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/rocjitsu/version.h
    @ONLY
)

# Expose the generated header path for targets that need it.
set(RJ_VERSION_INCLUDE_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/include
    CACHE INTERNAL
    ""
)
