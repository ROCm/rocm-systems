/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Simple test kernel for rocshmem_hipmodule_init testing
 * This file is compiled to HSACO/bitcode at build time
 *
 * This kernel includes actual rocshmem headers to use the real
 * ROCSHMEM_CTX_DEFAULT definition rather than a hardcoded copy.
 *****************************************************************************/

#include <hip/hip_runtime.h>

// Include rocshmem common header to get the actual ROCSHMEM_CTX_DEFAULT definition
// This ensures we're using the real rocshmem context structure
#include <rocshmem/rocshmem_common.hpp>

// Simple test kernel
// Note: We reference ROCSHMEM_CTX_DEFAULT here to ensure the symbol is included
// in the compiled module, even though we don't actually use it
extern "C" __global__ void simple_test_kernel(int *result, int *shmem_buf) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *result = 42;

    // Reference ROCSHMEM_CTX_DEFAULT to force the symbol to be included
    // This is a dummy reference - we're not actually using the context
    // but the linker will include the symbol in the module
    (void)&rocshmem::ROCSHMEM_CTX_DEFAULT;
  }
}
