# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC
    hipMemcpy.cc
    hipMemcpyAsync.cc
    hipMemcpyWithStream.cc
    hipMemcpyAtoH.cc
    hipMemcpyHtoA.cc
    hipMemcpyDtoD.cc
    hipMemcpyDtoDAsync.cc
    hipMemcpyDtoH.cc
    hipMemcpyDtoHAsync.cc
    hipMemcpyHtoD.cc
    hipMemcpyHtoDAsync.cc
    hipMemcpyWithKernel.cc
    hipMemcpyToSymbol.cc
    hipMemcpyToSymbolAsync.cc
    hipMemcpyFromSymbol.cc
    hipMemcpyFromSymbolAsync.cc
    hipMemcpy2D.cc
    hipMemcpy2DAsync.cc
    hipMemcpy2DToArray.cc
    hipMemcpy2DToArrayAsync.cc
    hipMemcpy2DFromArray.cc
    hipMemcpy2DFromArrayAsync.cc
    hipMemcpyParam2D.cc
    hipMemcpyParam2DAsync.cc
    hipMemcpy3D.cc
    hipMemcpy3DAsync.cc
)

hip_add_exe_to_target(NAME MemcpyPerformance
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
