// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file atomic_op.h
/// @brief The atomic read-modify-write operations a memory instruction can name.

#ifndef ROCJITSU_VM_AMDGPU_ATOMIC_OP_H_
#define ROCJITSU_VM_AMDGPU_ATOMIC_OP_H_

#include <cstdint>

namespace rocjitsu {
namespace amdgpu {

/// @brief Atomic read-modify-write operation type.
enum class AtomicOp : uint8_t {
  NONE = 0,       ///< Not an atomic operation.
  SWAP,           ///< Exchange.
  CONDXCHG32,     ///< Two conditional dword exchanges, controlled by each source sign bit.
  CMPSWAP,        ///< Compare-and-swap (data[0] = src, data[1] = cmp).
  MSKOR,          ///< Masked OR (data[0] = mask, data[1] = src).
  ADD,            ///< Atomic add.
  SUB,            ///< Atomic subtract (mem - data).
  SUB_CLAMP,      ///< Unsigned subtraction, clamped to zero on underflow.
  COND_SUB,       ///< Unsigned subtraction, retaining memory on underflow.
  RSUB,           ///< Atomic reverse subtract (data - mem).
  SMIN,           ///< Signed minimum.
  UMIN,           ///< Unsigned minimum.
  SMAX,           ///< Signed maximum.
  UMAX,           ///< Unsigned maximum.
  AND,            ///< Bitwise AND.
  OR,             ///< Bitwise OR.
  XOR,            ///< Bitwise XOR.
  INC,            ///< Increment (wrapping).
  DEC,            ///< Decrement (wrapping).
  FCMPSWAP,       ///< Floating comparison, with replacement followed by comparison.
  FADD,           ///< Floating-point add.
  PK_ADD_F16,     ///< Two independent packed IEEE half additions.
  PK_ADD_BF16,    ///< Two independent packed BFloat16 additions.
  FMIN,           ///< Floating-point minimum.
  FMAX,           ///< Floating-point maximum.
  APPEND,         ///< LDS append counter.
  CONSUME,        ///< LDS consume counter.
  BARRIER_ARRIVE, ///< LDS barrier-arrive state update.
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_ATOMIC_OP_H_
