# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Cray toolchain: the ftn and cc wrappers follow the loaded PrgEnv module.
# Usage, from projects/hip/fortran:
#   cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/cray.cmake

set(CMAKE_Fortran_COMPILER ftn CACHE FILEPATH "Fortran compiler")
set(CMAKE_C_COMPILER       cc CACHE FILEPATH "C compiler")
