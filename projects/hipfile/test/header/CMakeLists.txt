# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Compile hipfile.h as C11 and as C++11 with strict warnings, so a
# construct in the header that isn't valid in both languages fails the
# build. These are the oldest standards the header supports. The objects
# are never linked.
#
# -Wundef catches a preprocessor test of a macro that only one language
# defines, such as __STDC_VERSION__ (C only) or __cplusplus (C++ only),
# that isn't guarded by defined().
#
# header-check.c is copied to a .cpp file in the build directory to
# compile it as C++. configure_file() reruns CMake when header-check.c
# changes, so the copy stays up to date.
#
# The target links hipfile, as a consumer would, and gets the hipFile
# include directory from it. That also checks that hipfile doesn't
# impose a language standard on its consumers: CMake compiles a target
# with the highest standard any of its dependencies require, so a
# cxx_std_* or c_std_* usage requirement on hipfile would raise this
# target above C11 or C++11, and header-check.c fails with #error if
# it isn't compiled as exactly those.
#
# The hipFile include directory isn't SYSTEM, since hipfile is a target
# in this build rather than an imported one, so warnings from the header
# are reported. The HIP include directory comes from hip::host and is
# SYSTEM, since it's an imported target, so warnings from the HIP
# headers are not reported.
#
# clang-tidy is disabled for this target. It finds its config by searching
# upward from the source file, and the generated .cpp file is in the build
# directory, which may be outside the source tree. Without a config, recent
# clang-tidy enables no checks and fails. hipfile.h is analyzed via the
# library sources anyway.

configure_file(header-check.c ${CMAKE_CURRENT_BINARY_DIR}/header-check.cpp COPYONLY)

add_library(hipfile_header_check OBJECT
    header-check.c
    ${CMAKE_CURRENT_BINARY_DIR}/header-check.cpp
)

set_target_properties(hipfile_header_check PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CXX_CLANG_TIDY ""
)

target_link_libraries(hipfile_header_check
    PRIVATE hipfile hip::host
)

target_compile_options(hipfile_header_check
    PRIVATE -Wall -Wextra -Wundef -Werror -pedantic
)
