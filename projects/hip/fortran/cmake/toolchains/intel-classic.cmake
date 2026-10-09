# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Classic Intel toolchain (ifort), deprecated by Intel in favour of ifx.
# Usage, from projects/hip/fortran:
#   cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/intel-classic.cmake

set(CMAKE_Fortran_COMPILER ifort CACHE FILEPATH "Fortran compiler")
set(CMAKE_C_COMPILER       icx CACHE FILEPATH "C compiler")
