// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file device_function_direct_call.hip
/// @brief HIP fixture for non-kernel device functions shared by two kernels.

#include <hip/hip_runtime.h>

/// @brief Leaf device function called from more than one kernel.
///
/// @details Out of line and not address-taken, so the linked object contains a
/// plain `STT_FUNC` body reached by a call rather than a relocation-backed
/// table. Clang materializes the callee address with
/// `s_get_pc_i64` + `s_add_nc_u64 lit64` and transfers with `s_swap_pc_i64`,
/// so this exercises the recovered-indirect call path rather than a direct
/// `s_call_i64`.
__device__ __attribute__((noinline)) float device_leaf(float a, float b) {
  return a * b + 1.0f;
}

/// @brief Non-leaf device function that calls the leaf.
///
/// @details A callee that itself calls must preserve its own return address
/// across the inner transfer. Clang stashes it in VGPR lanes with
/// `v_writelane_b32` / `v_readlane_b32` and still returns through `s[30:31]`,
/// which is the shape translation has to classify as returning.
__device__ __attribute__((noinline)) float device_nonleaf(float a, float b) {
  return device_leaf(a, b) + b;
}

/// @brief Device function whose only statement is a call, so clang tail-calls it.
///
/// @details A tail call lowers to `s_set_pc_i64` on the *callee* address in an
/// ordinary SGPR pair rather than `s[30:31]`, so it is an inter-function
/// transfer that happens to share its encoding shape with a return. Both bodies
/// relocate together, so the PC-relative literal stays correct only if they keep
/// their relative distance.
__device__ __attribute__((noinline)) float device_tail_caller(float a, float b) {
  return device_leaf(a, b);
}

/// @brief First caller of both device functions.
extern "C" __global__ void device_function_caller_a(float *__restrict__ output,
                                                    const float *__restrict__ input,
                                                    unsigned count) {
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  output[index] = device_nonleaf(input[index], 2.0f);
}

/// @brief Second caller, so both bodies are reachable from more than one scope.
extern "C" __global__ void device_function_caller_b(float *__restrict__ output,
                                                    const float *__restrict__ input,
                                                    unsigned count) {
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  output[index] = device_leaf(input[index], 3.0f) + device_tail_caller(input[index], 4.0f);
}
