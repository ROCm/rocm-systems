/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Compiles the real src/sym_kernels.cc for rccl-UnitTestsMicroTuningSymModel: the symmetric cost model
// calls its kernel masks and Tma gates. Same prelude as sym-kernels-test.cc, which owns this unit's tests.

#include <functional>

#include "fakes/devcomm_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"
#include "sym_kernels.h"

// Neuter __global__: a real definition needs hipLaunchKernel/__hipRegisterFunction, unavailable under -no-hip-rt.
#undef __global__
#define __global__

#include "fakes/sym_kernels_test_stubs.h"

#include SYM_KERNELS_HOST_CC_PATH

#undef __global__

#include SYM_KERNELS_CC_PATH

// sym_kernels.cc asks for the railed GIN backend; gin_host.cc is not linked into this binary.
ncclResult_t ncclGetRailedGinType(struct ncclComm*, ncclGinType_t* ginType) {
  *ginType = NCCL_GIN_TYPE_NONE;
  return ncclSuccess;
}
