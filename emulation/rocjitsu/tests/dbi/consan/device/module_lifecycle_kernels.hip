// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Caller-supplied LDS publication kernels for the HIP module lifecycle
/// fixture. CMake compiles this source twice so a single process can exercise
/// simultaneous, independently instrumented code objects.

#include <hip/hip_runtime.h>

#ifndef RJ_CONSAN_MODULE_VARIANT
#error "RJ_CONSAN_MODULE_VARIANT must select the module fixture"
#endif

#if RJ_CONSAN_MODULE_VARIANT == 1
#define RJ_CONSAN_MODULE_CORRECT module_lifecycle_correct_a
#define RJ_CONSAN_MODULE_INCORRECT module_lifecycle_incorrect_a
constexpr uint32_t kModuleTag = 0xa1000000u;
#elif RJ_CONSAN_MODULE_VARIANT == 2
#define RJ_CONSAN_MODULE_CORRECT module_lifecycle_correct_b
#define RJ_CONSAN_MODULE_INCORRECT module_lifecycle_incorrect_b
constexpr uint32_t kModuleTag = 0xb2000000u;
#else
#error "unsupported RJ_CONSAN_MODULE_VARIANT"
#endif

template <bool Synchronize>
__device__ void module_lifecycle_body(uint32_t *output, uint32_t marker) {
  __shared__ uint32_t published;
  const uint32_t wave = threadIdx.x / warpSize;
  const uint32_t lane = threadIdx.x % warpSize;
  if (wave == 0u && lane == 0u)
    published = marker ^ kModuleTag;
  if constexpr (Synchronize)
    __syncthreads();
  if (wave == 1u && lane == 0u)
    output[0] = published;
}

extern "C" __global__ __launch_bounds__(128) void RJ_CONSAN_MODULE_CORRECT(uint32_t *output,
                                                                           uint32_t marker) {
  module_lifecycle_body<true>(output, marker);
}

extern "C" __global__ __launch_bounds__(128) void RJ_CONSAN_MODULE_INCORRECT(uint32_t *output,
                                                                             uint32_t marker) {
  module_lifecycle_body<false>(output, marker);
}
