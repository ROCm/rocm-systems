/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Accelerated Blit Copy Engine (ABCE) — compiler configuration and status.

#ifndef ABCE_CONFIG_H_
#define ABCE_CONFIG_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Compiler configuration
//===----------------------------------------------------------------------===//

// C11 spells it _Static_assert; C++ and C23 spell it static_assert.
#if defined(__cplusplus)
#define ABCE_STATIC_ASSERT(expr, msg) static_assert(expr, msg)
#else
#define ABCE_STATIC_ASSERT(expr, msg) _Static_assert(expr, msg)
#endif  // __cplusplus

#if defined(__cplusplus)
#define ABCE_ALIGNOF(type) alignof(type)
#define ABCE_ALIGNAS(bytes) alignas(bytes)
#define ABCE_THREAD_LOCAL thread_local
#else
#define ABCE_ALIGNOF(type) _Alignof(type)
#define ABCE_ALIGNAS(bytes) _Alignas(bytes)
#define ABCE_THREAD_LOCAL _Thread_local
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Limits
//
// Here rather than beside their main users so the host planner and the device
// submitter, which a kernel includes without the planner, size their arrays from
// one definition.
//===----------------------------------------------------------------------===//

// Maximum number of SDMA engines the orchestrator tracks (also the
// engine-bitmask width).
#define ABCE_MAX_ENGINES 16u

// Device ids (KFD node ids) the topology snapshot, and so the engine policy,
// can describe.
#define ABCE_MAX_TOPOLOGY_DEVICES 16

//===----------------------------------------------------------------------===//
// Math / bit helpers
//===----------------------------------------------------------------------===//

// Macros rather than functions so they work on any integer type without the
// C++ templates they replace. Arguments are evaluated more than once.
#define ABCE_MIN(a, b) (((a) < (b)) ? (a) : (b))
#define ABCE_MAX(a, b) (((a) > (b)) ? (a) : (b))
#define ABCE_ARRAYSIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

// Rounds |value| up to a power-of-two |alignment|.
static inline uint64_t abce_align_up(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

// |value| / |divisor| rounded up, exact for every |value|: there is no rounding
// addition to overflow. |divisor| MUST be non-zero.
static inline uint64_t abce_div_round_up_u64(uint64_t value, uint64_t divisor) {
  return value / divisor + (value % divisor != 0);
}

static inline bool abce_is_power_of_two(uint64_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

// Index of the lowest set bit. |value| MUST be non-zero. GCC/Clang get the
// builtin; anything else gets a portable loop. The masks involved are at most
// ABCE_MAX_ENGINES bits wide and no caller is on a hot path, so the fallback
// costs nothing worth an ISA-specific intrinsic.
static inline int abce_count_trailing_zeros_u64(uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_ctzll(value);
#else
  int count = 0;
  while ((value & 1u) == 0) {
    value >>= 1;
    ++count;
  }
  return count;
#endif  // __GNUC__ || __clang__
}

static inline int abce_popcount_u64(uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_popcountll(value);
#else
  int count = 0;
  for (; value != 0; value >>= 1) count += (int)(value & 1u);
  return count;
#endif  // __GNUC__ || __clang__
}

//===----------------------------------------------------------------------===//
// abce_status_t
//===----------------------------------------------------------------------===//

// Every fallible ABCE entry point returns one of these. It replaces the
// per-phase RingStatus/MapStatus/SubmitStatus enums and the one place the C++
// headers threw (out-of-range rect pitch/slice, now
// ABCE_STATUS_RECT_OUT_OF_RANGE), so a caller has exactly one thing to check.
//
// The leading values line up with the usual status vocabulary; the tail are
// ABCE-specific failures worth telling apart in a log, which is the only reason
// they are not collapsed into INVALID_ARGUMENT.
typedef enum abce_status_t {
  ABCE_STATUS_OK = 0,
  // Malformed request: null pointer, zero size, a host-to-host copy.
  ABCE_STATUS_INVALID_ARGUMENT,
  // The request can never fit: a payload at or above the ring size, or a frame
  // larger than the ring can hold.
  ABCE_STATUS_OUT_OF_RANGE,
  // A required allocation failed.
  ABCE_STATUS_RESOURCE_EXHAUSTED,
  // The operation is not available on this hardware (for example an indirect
  // copy below gfx125).
  ABCE_STATUS_UNIMPLEMENTED,
  // Rect copy pitch or slice exceeds what the packet can encode. The C++
  // builder threw std::invalid_argument here.
  ABCE_STATUS_RECT_OUT_OF_RANGE,
  // No engine is registered, or the batch's engine mask selected none.
  ABCE_STATUS_NO_ENGINE,
  // Engines are registered but policy ruled every one of them out for this
  // transfer.
  ABCE_STATUS_NO_LEGAL_ENGINE,
  // The batch exceeds the per-batch entry limit, or a frame exceeds UINT32_MAX
  // bytes.
  ABCE_STATUS_TOO_MANY_OPERATIONS,
  // The batch fans out across engines and needs a coordination word to sequence
  // them, but abce_signal_ref_t::coordination_scratch is null. Distinct from
  // INVALID_ARGUMENT because it depends on batch size and engine count, so the
  // same caller can map smaller batches successfully and only trip here once a
  // copy is large enough to fan out.
  ABCE_STATUS_MISSING_COORDINATION_SCRATCH,
  // A ring selected by the plan is not registered, or a reservation on it
  // failed.
  ABCE_STATUS_RING_UNAVAILABLE,
  // Submit was handed a plan that did not map, or one already submitted.
  ABCE_STATUS_INVALID_PLAN,
} abce_status_t;

static inline bool abce_status_is_ok(abce_status_t status) {
  return status == ABCE_STATUS_OK;
}

// Stable short name for logs. Never null.
static inline const char* abce_status_name(abce_status_t status) {
  switch (status) {
    case ABCE_STATUS_OK:
      return "OK";
    case ABCE_STATUS_INVALID_ARGUMENT:
      return "INVALID_ARGUMENT";
    case ABCE_STATUS_OUT_OF_RANGE:
      return "OUT_OF_RANGE";
    case ABCE_STATUS_RESOURCE_EXHAUSTED:
      return "RESOURCE_EXHAUSTED";
    case ABCE_STATUS_UNIMPLEMENTED:
      return "UNIMPLEMENTED";
    case ABCE_STATUS_RECT_OUT_OF_RANGE:
      return "RECT_OUT_OF_RANGE";
    case ABCE_STATUS_NO_ENGINE:
      return "NO_ENGINE";
    case ABCE_STATUS_NO_LEGAL_ENGINE:
      return "NO_LEGAL_ENGINE";
    case ABCE_STATUS_TOO_MANY_OPERATIONS:
      return "TOO_MANY_OPERATIONS";
    case ABCE_STATUS_MISSING_COORDINATION_SCRATCH:
      return "MISSING_COORDINATION_SCRATCH";
    case ABCE_STATUS_RING_UNAVAILABLE:
      return "RING_UNAVAILABLE";
    case ABCE_STATUS_INVALID_PLAN:
      return "INVALID_PLAN";
  }
  return "UNKNOWN";
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ABCE_CONFIG_H_
