// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMDF_XDNA_RESULT_H_
#define AMDF_XDNA_RESULT_H_

#include "amdf/base.h"
#include "amdf/observation.h"

/// An `amdf_xdna_result_buffer_info_t` output structure.
#define AMDF_STRUCTURE_TYPE_XDNA_RESULT_BUFFER_INFO \
  ((amdf_structure_type_t)0x0001000Fu)

/// Extent accepted by the native context's firmware-result association.
typedef uint32_t amdf_xdna_result_buffer_binding_mode_t;
enum amdf_xdna_result_buffer_binding_mode_e {
  /// Offset zero and the entire allocation, with no native rounding surplus.
  AMDF_XDNA_RESULT_BUFFER_BINDING_WHOLE_MEMORY = 1,
  /// One aligned range within an allocation shared with other context uses.
  AMDF_XDNA_RESULT_BUFFER_BINDING_RANGE = 2,
};

/// Cached requirements for a context's finite firmware-result destination.
/// The caller queries these before choosing its allocation/layout. This query
/// performs no native mutation and reserves nothing; attachment checks native
/// admission. The protocol supplies neither a live cursor nor an overflow
/// count.
typedef struct amdf_xdna_result_buffer_info_t {
  /// Must be AMDF_STRUCTURE_TYPE_XDNA_RESULT_BUFFER_INFO.
  amdf_structure_type_t type;
  /// Must be at least sizeof(amdf_xdna_result_buffer_info_t).
  uint32_t structure_size;
  /// Optional output extension chain. No extensions are defined.
  void* next;
  /// Context-private scope borrowed through the context's lifetime.
  amdf_memory_scope_t* memory_scope;
  /// Memory profile in memory_scope supplying the admitted native backing.
  uint32_t memory_profile_ordinal;
  /// Whether the destination occupies a whole allocation or an aligned range.
  amdf_xdna_result_buffer_binding_mode_t binding_mode;
  /// Smallest admitted destination capacity in bytes.
  uint64_t minimum_byte_length;
  /// Largest admitted destination capacity; allocation limits also apply.
  uint64_t maximum_byte_length;
  /// Required power-of-two alignment of the logical byte offset.
  uint64_t byte_offset_alignment;
  /// Required nonzero multiple for the destination capacity.
  uint64_t byte_length_granularity;
  /// RECORD_TIMER width; frequency is zero in this cached geometry query.
  /// device_query_timer_clock_info samples the separate operating rate.
  amdf_observation_clock_info_t timer_clock;
} amdf_xdna_result_buffer_info_t;

/// Caller-owned result backing borrowed by firmware until explicit detachment.
/// The descriptor itself is borrowed only during each control call. All calls
/// for one attachment supply the same memory, access and range. This range is
/// disjoint from live instructions and every other application memory use.
typedef struct amdf_xdna_result_buffer_binding_t {
  /// Memory created from the queried context-private scope and profile.
  amdf_memory_t* memory;
  /// Access granting the context device read and write permission.
  uint32_t access_ordinal;
  /// Reserved; zero.
  uint32_t reserved;
  /// Offset from logical byte zero, excluding any provider-owned prefix.
  uint64_t byte_offset;
  /// Finite capacity satisfying the queried geometry and planned record bound.
  uint64_t byte_length;
} amdf_xdna_result_buffer_binding_t;

/// One transaction-interpreter RECORD_TIMER result, in native word order.
/// The compiler/caller establishes record layout, count and timer epoch. Other
/// result-producing instructions can insert different records into this buffer.
typedef struct amdf_xdna_timer_record_t {
  /// Caller-selected identifier supplied by the RECORD_TIMER instruction.
  uint32_t id;
  /// Partition-shim timer words; the high word precedes the low word in memory.
  struct {
    /// Most significant 32 timestamp bits.
    uint32_t high;
    /// Least significant 32 timestamp bits.
    uint32_t low;
  } timestamp;
} amdf_xdna_timer_record_t;

/// Combines a completed marker's native high/low words without clock
/// conversion. Samples are comparable only within a program-established common
/// timer epoch. Context, attachment and submission identity do not establish
/// that epoch.
static inline uint64_t amdf_xdna_timer_record_timestamp(
    const amdf_xdna_timer_record_t* record) {
  return ((uint64_t)record->timestamp.high << 32) | record->timestamp.low;
}

#endif  // AMDF_XDNA_RESULT_H_
