# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# AMD ROCm toolchain (amdflang), the recommended default.
# Usage, from projects/hip/fortran:
#   cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/amdflang.cmake

set(CMAKE_Fortran_COMPILER amdflang CACHE FILEPATH "Fortran compiler")
set(CMAKE_C_COMPILER       amdclang CACHE FILEPATH "C compiler")
