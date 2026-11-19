// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file atomic_op.h
/// @brief Atomic operations and shared integer read-modify-write semantics.

#ifndef ROCJITSU_VM_AMDGPU_ATOMIC_OP_H_
#define ROCJITSU_VM_AMDGPU_ATOMIC_OP_H_

#include "util/result.h"
#include <algorithm>
#include <atomic>
#include <cassert>
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
  ORDERED_ADD,    ///< Conditionally advance the streamout ID and add to its value.
};

/// @brief Whether each lane supplies a second operand after its source (data[1]).
constexpr bool atomic_uses_two_sources(AtomicOp op) {
  return op == AtomicOp::CMPSWAP || op == AtomicOp::FCMPSWAP || op == AtomicOp::MSKOR ||
         op == AtomicOp::WRAP;
}

constexpr bool atomic_is_floating_point(AtomicOp op) {
  return op == AtomicOp::FADD || op == AtomicOp::FMIN || op == AtomicOp::FMAX ||
         op == AtomicOp::FCMPSWAP || op == AtomicOp::PK_ADD_F16 || op == AtomicOp::PK_ADD_BF16;
}

/// @brief Apply an integer atomic RMW operation (32-bit or 64-bit).
/// @details Returns failure for operations outside the supported integer set.
template <typename T>
util::FailureOr<T> apply_int_atomic(AtomicOp op, T old_val, T src_val, T cmp_val = 0) {
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
  case AtomicOp::ORDERED_ADD: {
    if constexpr (sizeof(T) == sizeof(uint64_t)) {
      const auto id = static_cast<uint32_t>(old_val);
      if (id != static_cast<uint32_t>(src_val))
        return old_val;
      // The compute queue on gfx1201 uses the full 12-bit ordered wave ID.
      const uint32_t value =
          static_cast<uint32_t>(old_val >> 32) + static_cast<uint32_t>(src_val >> 32);
      return (T{value} << 32) | ((id + 1u) & 0xfffu);
    }
    return util::Result::failure();
  }
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
    return util::Result::failure();
  }
}

/// @brief Replace the T at @p target with compute(old) as one hardware read-modify-write.
/// @details A compare-exchange loop, so the update is atomic against host threads that use
/// their own atomics on the same word, not only against holders of a simulator lock.
/// @p compute runs once per attempt and must have no side effects. The target must be
/// naturally aligned; the GPU VM, L2Cache and PM4 paths refuse misaligned atomics first.
/// @returns The value the successful exchange replaced.
template <typename T, typename Compute> T atomic_update(void *target, Compute &&compute) {
  static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>,
                "backing atomics are 4 or 8 bytes");
  assert(reinterpret_cast<uintptr_t>(target) % std::atomic_ref<T>::required_alignment == 0 &&
         "backing atomics must be naturally aligned");
  std::atomic_ref<T> word(*static_cast<T *>(target));
  T old = word.load(std::memory_order_relaxed);
  while (!word.compare_exchange_weak(old, compute(old), std::memory_order_acq_rel,
                                     std::memory_order_acquire)) {
  }
  return old;
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_ATOMIC_OP_H_
