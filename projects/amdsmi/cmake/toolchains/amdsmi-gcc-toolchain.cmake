# SPDX-License-Identifier: MIT
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# Example toolchain: build AMD SMI with GCC instead of the default
# amdclang++/clang++ auto-detection.
#
# Usage:
#   cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/amdsmi-gcc-toolchain.cmake
#
# To use a specific GCC version:
#   1. Edit this file and change the compiler paths below
#   2. Or copy this file and create your own custom toolchain
#
# Example with GCC 13:
#   set(CMAKE_C_COMPILER /usr/bin/gcc-13 CACHE FILEPATH "C compiler" FORCE)
#   set(CMAKE_CXX_COMPILER /usr/bin/g++-13 CACHE FILEPATH "C++ compiler" FORCE)
#

# Set the C and C++ compilers to GCC
set(CMAKE_C_COMPILER gcc CACHE FILEPATH "C compiler" FORCE)
set(CMAKE_CXX_COMPILER g++ CACHE FILEPATH "C++ compiler" FORCE)

# Optional: Set compiler flags for GCC
# Uncomment and customize as needed
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall" CACHE STRING "C flags" FORCE)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall" CACHE STRING "C++ flags" FORCE)
