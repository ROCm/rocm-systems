# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Intel oneAPI toolchain (ifx).
# Usage, from projects/hip/fortran:
#   cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/intel.cmake

set(CMAKE_Fortran_COMPILER ifx CACHE FILEPATH "Fortran compiler")
set(CMAKE_C_COMPILER       icx CACHE FILEPATH "C compiler")
