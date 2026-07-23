// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file relocation_function_table_dispatch.hip
/// @brief HIP fixture for a relocation-backed eight-function device dispatch table.

#include <hip/hip_runtime.h>

using RelocationFunction = unsigned (*)(unsigned);

/// @brief Return a distinct result for function-table slot zero.
///
/// All eight targets remain address-taken and out of line so the linked code
/// object contains eight distinct `.text` destinations instead of folding the
/// dispatch into arithmetic or a switch.
__device__ __attribute__((noinline)) unsigned relocation_function_0(unsigned value) {
  return value + 1u;
}

/// @brief Return a distinct result for function-table slot one.
__device__ __attribute__((noinline)) unsigned relocation_function_1(unsigned value) {
  return value * 3u + 2u;
}

/// @brief Return a distinct result for function-table slot two.
__device__ __attribute__((noinline)) unsigned relocation_function_2(unsigned value) {
  return (value ^ 0x13579bdfu) + 3u;
}

/// @brief Return a distinct result for function-table slot three.
__device__ __attribute__((noinline)) unsigned relocation_function_3(unsigned value) {
  return (value << 3) | (value >> 29);
}

/// @brief Return a distinct result for function-table slot four.
__device__ __attribute__((noinline)) unsigned relocation_function_4(unsigned value) {
  return value - 5u;
}

/// @brief Return a distinct result for function-table slot five.
__device__ __attribute__((noinline)) unsigned relocation_function_5(unsigned value) {
  return (value ^ (value >> 7)) + 6u;
}

/// @brief Return a distinct result for function-table slot six.
__device__ __attribute__((noinline)) unsigned relocation_function_6(unsigned value) {
  return value * 9u - 7u;
}

/// @brief Return a distinct result for function-table slot seven.
__device__ __attribute__((noinline)) unsigned relocation_function_7(unsigned value) {
  return ~value + 8u;
}

/// External visibility and `used` make the table observable in the linked ELF.
/// Each initializer becomes a loader relocation whose addend identifies one of
/// the eight function entry points in `.text`.
extern "C" __device__ __attribute__((used)) RelocationFunction relocation_function_table[8] = {
    relocation_function_0, relocation_function_1, relocation_function_2, relocation_function_3,
    relocation_function_4, relocation_function_5, relocation_function_6, relocation_function_7,
};

/// Dispatch every work item through the table entry selected by an input integer.
///
/// The selector is a uniform kernel argument so the compiler can load the
/// selected function pointer through the scalar path modeled by DBT. Masking to
/// three bits proves the runtime index is within the eight populated slots
/// while keeping the target unknown at device-link time.
extern "C" __global__ void relocation_function_table_dispatch(unsigned selector,
                                                              const unsigned *__restrict__ input,
                                                              unsigned *__restrict__ output,
                                                              unsigned count) {
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;

  const unsigned table_index = selector & 7u;
  output[index] = relocation_function_table[table_index](input[index]);
}
