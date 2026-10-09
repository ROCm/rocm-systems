# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# GNU toolchain (gfortran).
# Usage, from projects/hip/fortran:
#   cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/gnu.cmake

set(CMAKE_Fortran_COMPILER gfortran CACHE FILEPATH "Fortran compiler")
set(CMAKE_C_COMPILER       gcc CACHE FILEPATH "C compiler")
