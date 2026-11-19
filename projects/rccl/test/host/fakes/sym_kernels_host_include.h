/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// #includes the generated sym_kernels_host.cc with __global__ neutered: a real definition needs
// hipLaunchKernel/__hipRegisterFunction, unavailable under -no-hip-rt. Include after sym_kernels.h.

#ifndef RCCL_TEST_HOST_FAKES_SYM_KERNELS_HOST_INCLUDE_H_
#define RCCL_TEST_HOST_FAKES_SYM_KERNELS_HOST_INCLUDE_H_

#undef __global__
#define __global__

#include "fakes/sym_kernels_test_stubs.h"

#include SYM_KERNELS_HOST_CC_PATH

#undef __global__

#endif  // RCCL_TEST_HOST_FAKES_SYM_KERNELS_HOST_INCLUDE_H_
