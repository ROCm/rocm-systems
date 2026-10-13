// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file atomic_op.h
/// @brief Atomic operations and shared integer read-modify-write semantics.

#ifndef ROCJITSU_VM_AMDGPU_ATOMIC_OP_H_
#define ROCJITSU_VM_AMDGPU_ATOMIC_OP_H_

#include <algorithm>
#include <cstdint>
#include <type_traits>

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
  WRAP,           ///< Subtract if nonnegative; otherwise add the second operand.
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

/// @brief Apply an integer atomic RMW operation (32-bit or 64-bit).
template <typename T> T apply_int_atomic(AtomicOp op, T old_val, T src_val, T cmp_val = 0) {
  using S = std::make_signed_t<T>;
  switch (op) {
  case AtomicOp::SWAP:
    return src_val;
  case AtomicOp::CONDXCHG32: {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8);
    T result = old_val;
    for (uint32_t shift = 0; shift < sizeof(T) * 8; shift += 32) {
      const T half_mask = T{0xffffffffu} << shift;
      const T store_mask = T{0x7fffffffu} << shift;
      if ((src_val >> shift) & 0x80000000u)
        result = (result & ~half_mask) | (src_val & store_mask);
    }
    return result;
  }
  case AtomicOp::CMPSWAP:
    return (old_val == cmp_val) ? src_val : old_val;
  case AtomicOp::MSKOR:
    return (old_val & ~src_val) | cmp_val;
  case AtomicOp::ADD:
    return old_val + src_val;
  case AtomicOp::SUB:
    return old_val - src_val;
  case AtomicOp::SUB_CLAMP:
    return old_val >= src_val ? old_val - src_val : T{0};
  case AtomicOp::COND_SUB:
    return old_val >= src_val ? old_val - src_val : old_val;
  case AtomicOp::WRAP:
    return old_val >= src_val ? old_val - src_val : old_val + cmp_val;
  case AtomicOp::RSUB:
    return src_val - old_val;
  case AtomicOp::SMIN:
    return static_cast<T>(std::min(static_cast<S>(old_val), static_cast<S>(src_val)));
  case AtomicOp::UMIN:
    return std::min(old_val, src_val);
  case AtomicOp::SMAX:
    return static_cast<T>(std::max(static_cast<S>(old_val), static_cast<S>(src_val)));
  case AtomicOp::UMAX:
    return std::max(old_val, src_val);
  case AtomicOp::AND:
    return old_val & src_val;
  case AtomicOp::OR:
    return old_val | src_val;
  case AtomicOp::XOR:
    return old_val ^ src_val;
  case AtomicOp::INC:
    return (old_val >= src_val) ? T{0} : old_val + 1;
  case AtomicOp::DEC:
    return (old_val == 0 || old_val > src_val) ? src_val : old_val - 1;
  default:
    return old_val;
  }
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_ATOMIC_OP_H_
