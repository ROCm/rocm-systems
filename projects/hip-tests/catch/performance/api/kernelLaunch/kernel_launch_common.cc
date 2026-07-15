/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "kernel_launch_common.hh"

#define DO_NOT_OPTIMIZE_AWAY                                                                       \
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;                                              \
  if (out) *out = args.args[i];

__global__ void NullKernel() {}

__global__ void KernelWithSmallArgs(SmallKernelArgs args, char* out) { DO_NOT_OPTIMIZE_AWAY; }

__global__ void KernelWithMediumArgs(MediumKernelArgs args, char* out) { DO_NOT_OPTIMIZE_AWAY; }

__global__ void KernelWithLargeArgs(LargeKernelArgs args, char* out) { DO_NOT_OPTIMIZE_AWAY; }

SmallKernelArgs small_kernel_args;
MediumKernelArgs medium_kernel_args;
LargeKernelArgs large_kernel_args;
