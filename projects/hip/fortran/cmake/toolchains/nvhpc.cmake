# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# NVIDIA HPC SDK toolchain (nvfortran).
# Usage, from projects/hip/fortran:
#   cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/nvhpc.cmake

set(CMAKE_Fortran_COMPILER nvfortran CACHE FILEPATH "Fortran compiler")
set(CMAKE_C_COMPILER       nvc CACHE FILEPATH "C compiler")
