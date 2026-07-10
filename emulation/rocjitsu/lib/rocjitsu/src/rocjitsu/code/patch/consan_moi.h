// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi.h
/// @brief MOI instrumentation mode entry points for ConSan DBI patching.

#pragma once

#include "rocjitsu/code/patch/consan.h"
#include "rocjitsu/code/rj_code.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>

namespace rocjitsu {

enum class ConSanMoiShadowAccessKind : uint32_t {
  Empty = 0,
  Read = 1,
  Write = 2,
  ReadWrite = 3,
  Atomic = 4,
};

enum class ConSanMoiDiagnosticKind : uint32_t {
  None = 0,
  AccessConflict = 1,
  MetadataFull = 2,
  BarrierDivergence = 3,
};

enum class ConSanMoiAtomicEventKind : uint32_t {
  Release = 1,
  Acquire = 2,
};

inline constexpr uint32_t kConSanMoiReportMagic = 0x494f4d43u; // "CMOI" as little-endian bytes.
inline constexpr uint32_t kConSanMoiReportAbiVersion = 1;

struct alignas(8) ConSanMoiReportHeader {
  uint32_t magic = kConSanMoiReportMagic;
  uint32_t abi_version = kConSanMoiReportAbiVersion;
  uint32_t header_size = 0;
  uint32_t flags = 0;
  uint64_t generation = 0;
  uint64_t dispatch_id = 0;
  uint32_t access_record_capacity = 0;
  uint32_t access_record_count = 0;
  uint32_t barrier_record_capacity = 0;
  uint32_t barrier_record_count = 0;
  uint32_t atomic_record_capacity = 0;
  uint32_t atomic_record_count = 0;
  uint32_t diagnostic_capacity = 0;
  uint32_t diagnostic_count = 0;
  uint32_t exact_shadow_entry_capacity = 0;
  uint32_t sampled_watchpoint_capacity = 0;
  uint32_t event_counter = 0;
  uint32_t reserved = 0;
};

struct alignas(8) ConSanMoiAccessRecord {
  uint64_t generation = 0;
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  uint32_t wave_id = 0;
  uint64_t lane_mask = 0;
  uint32_t instruction_offset = 0;
  uint32_t access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Empty);
  uint32_t lds_byte_offset = 0;
  uint32_t lds_byte_count = 0;
  uint32_t start_cell = 0;
  uint32_t cell_count = 0;
  uint32_t epoch = 0;
  uint32_t event_index = 0;
};

struct alignas(8) ConSanMoiBarrierRecord {
  uint64_t generation = 0;
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  uint32_t wave_id = 0;
  uint64_t lane_mask = 0;
  uint32_t instruction_offset = 0;
  uint32_t event_index = 0;
};

struct alignas(8) ConSanMoiAtomicRecord {
  uint64_t generation = 0;
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  uint32_t owner_id = 0;
  uint64_t atomic_address = 0;
  uint32_t instruction_offset = 0;
  uint32_t event_index = 0;
  uint32_t epoch = 0;
  ConSanMoiAtomicEventKind kind = ConSanMoiAtomicEventKind::Release;
  uint32_t scope = 0;
  uint32_t semantics = 0;
};

struct alignas(8) ConSanMoiInlineAtomicReleaseSlot {
  uint32_t valid = 0;
  uint32_t owner_id = 0;
  uint32_t epoch = 0;
  uint32_t reserved = 0;
  uint64_t atomic_address = 0;
};

struct alignas(8) ConSanMoiDiagnosticRecord {
  uint32_t kind = static_cast<uint32_t>(ConSanMoiDiagnosticKind::None);
  uint32_t backend = static_cast<uint32_t>(ConSanMoiEngine::RecordReplay);
  uint64_t generation = 0;
  uint32_t epoch = 0;
  uint32_t first_owner_id = 0;
  uint32_t second_owner_id = 0;
  uint32_t reserved = 0;
  uint64_t first_lane_mask = 0;
  uint64_t second_lane_mask = 0;
  uint32_t first_instruction_offset = 0;
  uint32_t second_instruction_offset = 0;
  uint32_t first_lds_byte_offset = 0;
  uint32_t first_lds_byte_count = 0;
  uint32_t second_lds_byte_offset = 0;
  uint32_t second_lds_byte_count = 0;
  uint32_t first_access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Empty);
  uint32_t second_access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Empty);
};

struct ConSanMoiReportBufferLayout {
  uint32_t access_record_capacity = 0;
  uint32_t barrier_record_capacity = 0;
  uint32_t atomic_record_capacity = 0;
  uint32_t diagnostic_capacity = 0;
  uint32_t exact_shadow_entry_capacity = 0;
  uint32_t sampled_watchpoint_capacity = 0;
  size_t access_records_offset = sizeof(ConSanMoiReportHeader);
  size_t barrier_records_offset = sizeof(ConSanMoiReportHeader);
  size_t atomic_records_offset = sizeof(ConSanMoiReportHeader);
  size_t diagnostic_records_offset = sizeof(ConSanMoiReportHeader);
  size_t exact_shadow_entries_offset = sizeof(ConSanMoiReportHeader);
  size_t inline_atomic_release_slots_offset = sizeof(ConSanMoiReportHeader);
  size_t sampled_watchpoints_offset = sizeof(ConSanMoiReportHeader);
};

struct ConSanMoiRecordReplayAccess {
  uint64_t generation = 0;
  uint32_t owner_id = 0;
  uint32_t epoch = 0;
  ConSanMoiShadowAccessKind kind = ConSanMoiShadowAccessKind::Empty;
  uint32_t lds_byte_offset = 0;
  uint32_t lds_byte_count = 0;
  uint32_t start_cell = 0;
  uint32_t cell_count = 0;
  uint32_t instruction_offset = 0;
  uint64_t lane_mask = 0;
};

struct ConSanMoiRecordReplayAccessResult {
  bool metadata_full = false;
  bool conflict = false;
  ConSanMoiDiagnosticRecord diagnostic;
};

struct ConSanMoiRecordReplayResult {
  uint32_t processed_access_count = 0;
  uint32_t processed_barrier_count = 0;
  uint32_t processed_atomic_count = 0;
  uint32_t dropped_access_count = 0;
  uint32_t dropped_barrier_count = 0;
  uint32_t unsupported_access_count = 0;
  uint32_t unsupported_atomic_count = 0;
  uint32_t emitted_diagnostic_count = 0;
  bool diagnostic_capacity_exhausted = false;
  bool metadata_full = false;
  bool conflict = false;
};

struct ConSanMoiSampledPublishResult {
  uint32_t processed_access_count = 0;
  uint32_t published_entry_count = 0;
  bool sampled_capacity_exhausted = false;
};

struct ConSanMoiSampledReplayResult {
  uint32_t processed_entry_count = 0;
  uint32_t emitted_diagnostic_count = 0;
  bool diagnostic_capacity_exhausted = false;
  bool conflict = false;
};

struct ConSanMoiAtomicReleaseRecord {
  bool valid = false;
  uint64_t generation = 0;
  uint64_t atomic_address = 0;
  uint32_t producer_owner_id = 0;
  uint32_t producer_epoch = 0;
  uint32_t release_instruction_offset = 0;
};

struct ConSanMoiAcquiredEpochToken {
  bool valid = false;
  uint64_t generation = 0;
  uint32_t consumer_owner_id = 0;
  uint32_t producer_owner_id = 0;
  uint32_t producer_epoch_plus_one = 0;
  uint32_t acquire_instruction_offset = 0;
};

struct ConSanMoiAtomicSyncResult {
  bool metadata_full = false;
  uint32_t updated_record_count = 0;
};

using ConSanMoiRecordReplayAtomicEvent = ConSanMoiAtomicRecord;

[[nodiscard]] constexpr ConSanMoiReportHeader make_consan_moi_report_header(
    uint64_t generation, uint64_t dispatch_id, uint32_t access_record_capacity,
    uint32_t diagnostic_capacity, uint32_t exact_shadow_entry_capacity,
    uint32_t sampled_watchpoint_capacity, uint32_t barrier_record_capacity = 0,
    uint32_t atomic_record_capacity = 0) {
  ConSanMoiReportHeader header;
  header.header_size = sizeof(ConSanMoiReportHeader);
  header.generation = generation;
  header.dispatch_id = dispatch_id;
  header.access_record_capacity = access_record_capacity;
  header.barrier_record_capacity = barrier_record_capacity;
  header.atomic_record_capacity = atomic_record_capacity;
  header.diagnostic_capacity = diagnostic_capacity;
  header.exact_shadow_entry_capacity = exact_shadow_entry_capacity;
  header.sampled_watchpoint_capacity = sampled_watchpoint_capacity;
  return header;
}

[[nodiscard]] constexpr size_t consan_moi_report_buffer_min_bytes(
    uint32_t access_record_capacity, uint32_t diagnostic_capacity,
    uint32_t exact_shadow_entry_capacity, uint32_t sampled_watchpoint_capacity,
    uint32_t barrier_record_capacity = 0, uint32_t atomic_record_capacity = 0) {
  return sizeof(ConSanMoiReportHeader) +
         static_cast<size_t>(access_record_capacity) * sizeof(ConSanMoiAccessRecord) +
         static_cast<size_t>(barrier_record_capacity) * sizeof(ConSanMoiBarrierRecord) +
         static_cast<size_t>(atomic_record_capacity) * sizeof(ConSanMoiAtomicRecord) +
         static_cast<size_t>(diagnostic_capacity) * sizeof(ConSanMoiDiagnosticRecord) +
         static_cast<size_t>(exact_shadow_entry_capacity) * sizeof(uint64_t) +
         static_cast<size_t>(sampled_watchpoint_capacity) * sizeof(uint64_t);
}

[[nodiscard]] constexpr uint32_t consan_moi_clamp_u32_capacity(uint64_t capacity) {
  return capacity > std::numeric_limits<uint32_t>::max() ? std::numeric_limits<uint32_t>::max()
                                                         : static_cast<uint32_t>(capacity);
}

[[nodiscard]] constexpr ConSanMoiReportBufferLayout
consan_moi_report_buffer_layout_for_bytes(uint64_t report_buffer_size, bool include_barriers,
                                          bool include_atomics = false) {
  ConSanMoiReportBufferLayout layout;
  if (report_buffer_size < sizeof(ConSanMoiReportHeader))
    return layout;

  const uint64_t payload_bytes = report_buffer_size - sizeof(ConSanMoiReportHeader);
  if (!include_barriers && !include_atomics) {
    layout.access_record_capacity =
        consan_moi_clamp_u32_capacity(payload_bytes / sizeof(ConSanMoiAccessRecord));
    layout.barrier_records_offset =
        layout.access_records_offset +
        static_cast<size_t>(layout.access_record_capacity) * sizeof(ConSanMoiAccessRecord);
    layout.atomic_records_offset = layout.barrier_records_offset;
    layout.diagnostic_records_offset = layout.atomic_records_offset;
    layout.exact_shadow_entries_offset = layout.diagnostic_records_offset;
    layout.inline_atomic_release_slots_offset = layout.exact_shadow_entries_offset;
    layout.sampled_watchpoints_offset = layout.exact_shadow_entries_offset;
    return layout;
  }

  if (!include_barriers && include_atomics) {
    const uint64_t paired_capacity =
        payload_bytes / (sizeof(ConSanMoiAccessRecord) + sizeof(ConSanMoiAtomicRecord));
    const uint32_t capacity = consan_moi_clamp_u32_capacity(paired_capacity);
    layout.access_record_capacity = capacity;
    layout.atomic_record_capacity = capacity;
    layout.barrier_records_offset =
        layout.access_records_offset +
        static_cast<size_t>(layout.access_record_capacity) * sizeof(ConSanMoiAccessRecord);
    layout.atomic_records_offset = layout.barrier_records_offset;
    layout.diagnostic_records_offset =
        layout.atomic_records_offset +
        static_cast<size_t>(layout.atomic_record_capacity) * sizeof(ConSanMoiAtomicRecord);
    layout.exact_shadow_entries_offset = layout.diagnostic_records_offset;
    layout.inline_atomic_release_slots_offset = layout.exact_shadow_entries_offset;
    layout.sampled_watchpoints_offset = layout.exact_shadow_entries_offset;
    return layout;
  }

  if (include_barriers && !include_atomics) {
    const uint64_t paired_capacity =
        payload_bytes / (sizeof(ConSanMoiAccessRecord) + sizeof(ConSanMoiBarrierRecord));
    const uint32_t capacity = consan_moi_clamp_u32_capacity(paired_capacity);
    layout.access_record_capacity = capacity;
    layout.barrier_record_capacity = capacity;
    layout.barrier_records_offset =
        layout.access_records_offset +
        static_cast<size_t>(layout.access_record_capacity) * sizeof(ConSanMoiAccessRecord);
    layout.atomic_records_offset =
        layout.barrier_records_offset +
        static_cast<size_t>(layout.barrier_record_capacity) * sizeof(ConSanMoiBarrierRecord);
    layout.diagnostic_records_offset = layout.atomic_records_offset;
    layout.exact_shadow_entries_offset = layout.diagnostic_records_offset;
    layout.inline_atomic_release_slots_offset = layout.exact_shadow_entries_offset;
    layout.sampled_watchpoints_offset = layout.exact_shadow_entries_offset;
    return layout;
  }

  const uint64_t paired_capacity =
      payload_bytes / (sizeof(ConSanMoiAccessRecord) + sizeof(ConSanMoiBarrierRecord) +
                       sizeof(ConSanMoiAtomicRecord));
  const uint32_t capacity = consan_moi_clamp_u32_capacity(paired_capacity);
  layout.access_record_capacity = capacity;
  layout.barrier_record_capacity = capacity;
  layout.atomic_record_capacity = capacity;
  layout.barrier_records_offset =
      layout.access_records_offset +
      static_cast<size_t>(layout.access_record_capacity) * sizeof(ConSanMoiAccessRecord);
  layout.atomic_records_offset =
      layout.barrier_records_offset +
      static_cast<size_t>(layout.barrier_record_capacity) * sizeof(ConSanMoiBarrierRecord);
  layout.diagnostic_records_offset =
      layout.atomic_records_offset +
      static_cast<size_t>(layout.atomic_record_capacity) * sizeof(ConSanMoiAtomicRecord);
  layout.exact_shadow_entries_offset = layout.diagnostic_records_offset;
  layout.inline_atomic_release_slots_offset = layout.exact_shadow_entries_offset;
  layout.sampled_watchpoints_offset = layout.exact_shadow_entries_offset;
  return layout;
}

[[nodiscard]] constexpr ConSanMoiReportBufferLayout
consan_moi_direct_sampled_report_buffer_layout_for_bytes(uint64_t report_buffer_size) {
  ConSanMoiReportBufferLayout layout;
  if (report_buffer_size < sizeof(ConSanMoiReportHeader))
    return layout;

  const uint64_t payload_bytes = report_buffer_size - sizeof(ConSanMoiReportHeader);
  layout.sampled_watchpoint_capacity =
      consan_moi_clamp_u32_capacity(payload_bytes / sizeof(uint64_t));
  layout.barrier_records_offset = layout.access_records_offset;
  layout.atomic_records_offset = layout.access_records_offset;
  layout.diagnostic_records_offset = layout.access_records_offset;
  layout.exact_shadow_entries_offset = layout.access_records_offset;
  layout.inline_atomic_release_slots_offset = layout.access_records_offset;
  layout.sampled_watchpoints_offset = layout.access_records_offset;
  return layout;
}

inline constexpr uint32_t kConSanMoiInlineShadowDefaultDiagnosticCapacity = 4;
inline constexpr uint32_t kConSanMoiInlineShadowConservativeExactShadowEntries = 16u * 1024u;
inline constexpr uint32_t kConSanMoiInlineShadowAtomicReleaseSlotCapacity = 1;

[[nodiscard]] constexpr ConSanMoiReportBufferLayout
consan_moi_inline_shadow_report_buffer_layout_for_bytes(
    uint64_t report_buffer_size,
    uint32_t requested_diagnostic_capacity = kConSanMoiInlineShadowDefaultDiagnosticCapacity) {
  ConSanMoiReportBufferLayout layout;
  if (report_buffer_size < sizeof(ConSanMoiReportHeader))
    return layout;

  const uint64_t payload_bytes = report_buffer_size - sizeof(ConSanMoiReportHeader);
  uint32_t diagnostic_capacity = consan_moi_clamp_u32_capacity(std::min<uint64_t>(
      requested_diagnostic_capacity, payload_bytes / sizeof(ConSanMoiDiagnosticRecord)));
  while (diagnostic_capacity != 0 &&
         payload_bytes -
                 static_cast<uint64_t>(diagnostic_capacity) * sizeof(ConSanMoiDiagnosticRecord) <
             sizeof(ConSanMoiInlineAtomicReleaseSlot) + sizeof(uint64_t)) {
    --diagnostic_capacity;
  }

  const uint64_t diagnostic_bytes =
      static_cast<uint64_t>(diagnostic_capacity) * sizeof(ConSanMoiDiagnosticRecord);
  const uint64_t inline_atomic_release_slot_bytes =
      diagnostic_capacity == 0
          ? 0
          : static_cast<uint64_t>(kConSanMoiInlineShadowAtomicReleaseSlotCapacity) *
                sizeof(ConSanMoiInlineAtomicReleaseSlot);
  layout.diagnostic_capacity = diagnostic_capacity;
  layout.exact_shadow_entry_capacity = consan_moi_clamp_u32_capacity(
      (payload_bytes - diagnostic_bytes - inline_atomic_release_slot_bytes) / sizeof(uint64_t));
  layout.diagnostic_records_offset = sizeof(ConSanMoiReportHeader);
  layout.exact_shadow_entries_offset =
      layout.diagnostic_records_offset +
      static_cast<size_t>(layout.diagnostic_capacity) * sizeof(ConSanMoiDiagnosticRecord);
  layout.inline_atomic_release_slots_offset =
      layout.exact_shadow_entries_offset +
      static_cast<size_t>(layout.exact_shadow_entry_capacity) * sizeof(uint64_t);
  layout.sampled_watchpoints_offset =
      layout.inline_atomic_release_slots_offset + inline_atomic_release_slot_bytes;
  layout.barrier_records_offset = layout.access_records_offset;
  layout.atomic_records_offset = layout.access_records_offset;
  return layout;
}

struct ConSanMoiLdsCellRange {
  uint32_t start_cell = 0;
  uint32_t cell_count = 0;
};

[[nodiscard]] constexpr uint64_t consan_moi_low_bit_mask(uint32_t bits) {
  return bits >= 64 ? ~uint64_t{0} : ((uint64_t{1} << bits) - 1u);
}

[[nodiscard]] constexpr uint64_t consan_moi_bit_field_mask(uint32_t shift, uint32_t bits) {
  return consan_moi_low_bit_mask(bits) << shift;
}

[[nodiscard]] constexpr ConSanMoiShadowAccessKind
consan_moi_shadow_kind_from_access_kind(ConSanLdsAccessKind kind) {
  switch (kind) {
  case ConSanLdsAccessKind::Read:
    return ConSanMoiShadowAccessKind::Read;
  case ConSanLdsAccessKind::Write:
    return ConSanMoiShadowAccessKind::Write;
  case ConSanLdsAccessKind::Atomic:
    return ConSanMoiShadowAccessKind::Atomic;
  case ConSanLdsAccessKind::Other:
    return ConSanMoiShadowAccessKind::Empty;
  }
  return ConSanMoiShadowAccessKind::Empty;
}

[[nodiscard]] constexpr bool consan_moi_shadow_kind_is_empty(ConSanMoiShadowAccessKind kind) {
  return kind == ConSanMoiShadowAccessKind::Empty;
}

[[nodiscard]] constexpr bool consan_moi_shadow_kind_conflicts(ConSanMoiShadowAccessKind current,
                                                              ConSanMoiShadowAccessKind prior) {
  if (consan_moi_shadow_kind_is_empty(current) || consan_moi_shadow_kind_is_empty(prior))
    return false;
  if (current == ConSanMoiShadowAccessKind::Read)
    return prior != ConSanMoiShadowAccessKind::Read;
  if (current == ConSanMoiShadowAccessKind::Atomic)
    return prior != ConSanMoiShadowAccessKind::Atomic;
  return true;
}

namespace consan_moi_exact_shadow {

inline constexpr uint32_t granule_shift = 2;
inline constexpr uint32_t granule_bytes = 1u << granule_shift;

inline constexpr uint32_t access_kind_shift = 0;
inline constexpr uint32_t access_kind_bits = 3;
inline constexpr uint32_t owner_shift = 3;
inline constexpr uint32_t owner_bits = 10;
inline constexpr uint32_t epoch_shift = 13;
inline constexpr uint32_t epoch_bits = 10;
inline constexpr uint32_t generation_shift = 23;
inline constexpr uint32_t generation_bits = 20;
inline constexpr uint32_t instruction_offset_shift = 43;
inline constexpr uint32_t instruction_offset_bits = 21;

inline constexpr uint64_t access_kind_mask =
    consan_moi_bit_field_mask(access_kind_shift, access_kind_bits);
inline constexpr uint64_t owner_mask = consan_moi_bit_field_mask(owner_shift, owner_bits);
inline constexpr uint64_t epoch_mask = consan_moi_bit_field_mask(epoch_shift, epoch_bits);
inline constexpr uint64_t generation_mask =
    consan_moi_bit_field_mask(generation_shift, generation_bits);
inline constexpr uint64_t instruction_offset_mask =
    consan_moi_bit_field_mask(instruction_offset_shift, instruction_offset_bits);
inline constexpr uint64_t epoch_generation_mask = epoch_mask | generation_mask;

inline constexpr uint32_t max_owner = (1u << owner_bits) - 1u;
inline constexpr uint32_t max_epoch = (1u << epoch_bits) - 1u;
inline constexpr uint32_t max_generation = (1u << generation_bits) - 1u;
inline constexpr uint32_t max_instruction_offset = (1u << instruction_offset_bits) - 1u;

} // namespace consan_moi_exact_shadow

struct ConSanMoiExactShadowEntry {
  ConSanMoiShadowAccessKind kind = ConSanMoiShadowAccessKind::Empty;
  uint32_t owner_id = 0;
  uint32_t epoch = 0;
  uint32_t generation = 0;
  uint32_t instruction_offset = 0;
};

[[nodiscard]] constexpr uint64_t pack_consan_moi_exact_shadow_entry(ConSanMoiShadowAccessKind kind,
                                                                    uint32_t owner_id,
                                                                    uint32_t epoch,
                                                                    uint32_t generation,
                                                                    uint64_t instruction_offset) {
  return ((static_cast<uint64_t>(kind) &
           consan_moi_low_bit_mask(consan_moi_exact_shadow::access_kind_bits))
          << consan_moi_exact_shadow::access_kind_shift) |
         ((static_cast<uint64_t>(owner_id) &
           consan_moi_low_bit_mask(consan_moi_exact_shadow::owner_bits))
          << consan_moi_exact_shadow::owner_shift) |
         ((static_cast<uint64_t>(epoch) &
           consan_moi_low_bit_mask(consan_moi_exact_shadow::epoch_bits))
          << consan_moi_exact_shadow::epoch_shift) |
         ((static_cast<uint64_t>(generation) &
           consan_moi_low_bit_mask(consan_moi_exact_shadow::generation_bits))
          << consan_moi_exact_shadow::generation_shift) |
         ((instruction_offset &
           consan_moi_low_bit_mask(consan_moi_exact_shadow::instruction_offset_bits))
          << consan_moi_exact_shadow::instruction_offset_shift);
}

[[nodiscard]] constexpr ConSanMoiExactShadowEntry
decode_consan_moi_exact_shadow_entry(uint64_t value) {
  return ConSanMoiExactShadowEntry{
      static_cast<ConSanMoiShadowAccessKind>((value & consan_moi_exact_shadow::access_kind_mask) >>
                                             consan_moi_exact_shadow::access_kind_shift),
      static_cast<uint32_t>((value & consan_moi_exact_shadow::owner_mask) >>
                            consan_moi_exact_shadow::owner_shift),
      static_cast<uint32_t>((value & consan_moi_exact_shadow::epoch_mask) >>
                            consan_moi_exact_shadow::epoch_shift),
      static_cast<uint32_t>((value & consan_moi_exact_shadow::generation_mask) >>
                            consan_moi_exact_shadow::generation_shift),
      static_cast<uint32_t>((value & consan_moi_exact_shadow::instruction_offset_mask) >>
                            consan_moi_exact_shadow::instruction_offset_shift),
  };
}

[[nodiscard]] constexpr bool
consan_moi_exact_shadow_entries_conflict(const ConSanMoiExactShadowEntry &current,
                                         const ConSanMoiExactShadowEntry &prior) {
  return !consan_moi_shadow_kind_is_empty(prior.kind) && current.epoch == prior.epoch &&
         current.generation == prior.generation && current.owner_id != prior.owner_id &&
         consan_moi_shadow_kind_conflicts(current.kind, prior.kind);
}

[[nodiscard]] constexpr uint32_t consan_moi_acquired_epoch_token_value(uint32_t epoch) {
  return epoch == std::numeric_limits<uint32_t>::max() ? epoch : epoch + 1u;
}

namespace consan_moi_sampled_watchpoint {

inline constexpr uint32_t granule_shift = 2;
inline constexpr uint32_t granule_bytes = 1u << granule_shift;

inline constexpr uint32_t valid_shift = 0;
inline constexpr uint32_t consumed_shift = 1;
inline constexpr uint32_t access_kind_shift = 2;
inline constexpr uint32_t access_kind_bits = 3;
inline constexpr uint32_t owner_shift = 5;
inline constexpr uint32_t owner_bits = 10;
inline constexpr uint32_t epoch_shift = 15;
inline constexpr uint32_t epoch_bits = 10;
inline constexpr uint32_t generation_shift = 25;
inline constexpr uint32_t generation_bits = 20;
inline constexpr uint32_t start_shift = 45;
inline constexpr uint32_t start_bits = 14;
inline constexpr uint32_t count_shift = 59;
inline constexpr uint32_t count_bits = 5;

inline constexpr uint64_t valid_mask = uint64_t{1} << valid_shift;
inline constexpr uint64_t consumed_mask = uint64_t{1} << consumed_shift;
inline constexpr uint64_t access_kind_mask =
    consan_moi_bit_field_mask(access_kind_shift, access_kind_bits);
inline constexpr uint64_t owner_mask = consan_moi_bit_field_mask(owner_shift, owner_bits);
inline constexpr uint64_t epoch_mask = consan_moi_bit_field_mask(epoch_shift, epoch_bits);
inline constexpr uint64_t generation_mask =
    consan_moi_bit_field_mask(generation_shift, generation_bits);
inline constexpr uint64_t start_mask = consan_moi_bit_field_mask(start_shift, start_bits);
inline constexpr uint64_t count_mask = consan_moi_bit_field_mask(count_shift, count_bits);
inline constexpr uint64_t epoch_generation_mask = epoch_mask | generation_mask;

inline constexpr uint32_t max_owner = (1u << owner_bits) - 1u;
inline constexpr uint32_t max_epoch = (1u << epoch_bits) - 1u;
inline constexpr uint32_t max_generation = (1u << generation_bits) - 1u;
inline constexpr uint32_t max_start = (1u << start_bits) - 1u;
inline constexpr uint32_t max_count = 1u << count_bits;

} // namespace consan_moi_sampled_watchpoint

struct ConSanMoiSampledWatchpointEntry {
  bool valid = false;
  bool consumed = false;
  ConSanMoiShadowAccessKind kind = ConSanMoiShadowAccessKind::Empty;
  uint32_t owner_id = 0;
  uint32_t epoch = 0;
  uint32_t generation = 0;
  uint32_t start_cell = 0;
  uint32_t cell_count = 0;
};

[[nodiscard]] constexpr uint32_t encode_consan_moi_sampled_cell_count(uint32_t cell_count) {
  return cell_count == 0 ? 0 : cell_count - 1u;
}

[[nodiscard]] constexpr uint64_t
pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind kind, uint32_t owner_id,
                                         uint32_t epoch, uint32_t generation, uint32_t start_cell,
                                         uint32_t cell_count, bool consumed = false) {
  return consan_moi_sampled_watchpoint::valid_mask |
         (consumed ? consan_moi_sampled_watchpoint::consumed_mask : uint64_t{0}) |
         ((static_cast<uint64_t>(kind) &
           consan_moi_low_bit_mask(consan_moi_sampled_watchpoint::access_kind_bits))
          << consan_moi_sampled_watchpoint::access_kind_shift) |
         ((static_cast<uint64_t>(owner_id) &
           consan_moi_low_bit_mask(consan_moi_sampled_watchpoint::owner_bits))
          << consan_moi_sampled_watchpoint::owner_shift) |
         ((static_cast<uint64_t>(epoch) &
           consan_moi_low_bit_mask(consan_moi_sampled_watchpoint::epoch_bits))
          << consan_moi_sampled_watchpoint::epoch_shift) |
         ((static_cast<uint64_t>(generation) &
           consan_moi_low_bit_mask(consan_moi_sampled_watchpoint::generation_bits))
          << consan_moi_sampled_watchpoint::generation_shift) |
         ((static_cast<uint64_t>(start_cell) &
           consan_moi_low_bit_mask(consan_moi_sampled_watchpoint::start_bits))
          << consan_moi_sampled_watchpoint::start_shift) |
         ((static_cast<uint64_t>(encode_consan_moi_sampled_cell_count(cell_count)) &
           consan_moi_low_bit_mask(consan_moi_sampled_watchpoint::count_bits))
          << consan_moi_sampled_watchpoint::count_shift);
}

[[nodiscard]] constexpr ConSanMoiSampledWatchpointEntry
decode_consan_moi_sampled_watchpoint_entry(uint64_t value) {
  return ConSanMoiSampledWatchpointEntry{
      (value & consan_moi_sampled_watchpoint::valid_mask) != 0,
      (value & consan_moi_sampled_watchpoint::consumed_mask) != 0,
      static_cast<ConSanMoiShadowAccessKind>(
          (value & consan_moi_sampled_watchpoint::access_kind_mask) >>
          consan_moi_sampled_watchpoint::access_kind_shift),
      static_cast<uint32_t>((value & consan_moi_sampled_watchpoint::owner_mask) >>
                            consan_moi_sampled_watchpoint::owner_shift),
      static_cast<uint32_t>((value & consan_moi_sampled_watchpoint::epoch_mask) >>
                            consan_moi_sampled_watchpoint::epoch_shift),
      static_cast<uint32_t>((value & consan_moi_sampled_watchpoint::generation_mask) >>
                            consan_moi_sampled_watchpoint::generation_shift),
      static_cast<uint32_t>((value & consan_moi_sampled_watchpoint::start_mask) >>
                            consan_moi_sampled_watchpoint::start_shift),
      static_cast<uint32_t>(((value & consan_moi_sampled_watchpoint::count_mask) >>
                             consan_moi_sampled_watchpoint::count_shift) +
                            1u),
  };
}

[[nodiscard]] constexpr uint32_t consan_moi_range_end_cell(uint32_t start_cell,
                                                           uint32_t cell_count) {
  return start_cell + cell_count;
}

[[nodiscard]] constexpr bool consan_moi_cell_ranges_overlap(uint32_t first_start,
                                                            uint32_t first_count,
                                                            uint32_t second_start,
                                                            uint32_t second_count) {
  return first_start < consan_moi_range_end_cell(second_start, second_count) &&
         second_start < consan_moi_range_end_cell(first_start, first_count);
}

[[nodiscard]] constexpr bool consan_moi_cell_ranges_overlap(ConSanMoiLdsCellRange first,
                                                            ConSanMoiLdsCellRange second) {
  return consan_moi_cell_ranges_overlap(first.start_cell, first.cell_count, second.start_cell,
                                        second.cell_count);
}

[[nodiscard]] constexpr ConSanMoiLdsCellRange
consan_moi_lds_cell_range_for_bytes(uint32_t byte_offset, uint32_t byte_count) {
  const uint32_t start_cell = byte_offset >> consan_moi_exact_shadow::granule_shift;
  const uint32_t end_cell =
      (byte_offset + byte_count + consan_moi_exact_shadow::granule_bytes - 1u) >>
      consan_moi_exact_shadow::granule_shift;
  return ConSanMoiLdsCellRange{start_cell, end_cell - start_cell};
}

[[nodiscard]] constexpr bool
consan_moi_sampled_watchpoints_conflict(const ConSanMoiSampledWatchpointEntry &current,
                                        const ConSanMoiSampledWatchpointEntry &prior) {
  return current.valid && prior.valid && !prior.consumed && current.epoch == prior.epoch &&
         current.generation == prior.generation && current.owner_id != prior.owner_id &&
         consan_moi_shadow_kind_conflicts(current.kind, prior.kind) &&
         consan_moi_cell_ranges_overlap(current.start_cell, current.cell_count, prior.start_cell,
                                        prior.cell_count);
}

[[nodiscard]] ConSanMoiRecordReplayAccessResult
consan_moi_record_replay_access(std::span<uint64_t> exact_shadow_entries,
                                const ConSanMoiRecordReplayAccess &access);

[[nodiscard]] ConSanMoiRecordReplayAccessResult
consan_moi_record_replay_access(std::span<uint64_t> exact_shadow_entries,
                                const ConSanMoiRecordReplayAccess &access,
                                std::span<const ConSanMoiAcquiredEpochToken> acquired_epoch_tokens);

[[nodiscard]] bool
consan_moi_acquired_epoch_orders(std::span<const ConSanMoiAcquiredEpochToken> tokens,
                                 const ConSanMoiExactShadowEntry &current,
                                 const ConSanMoiExactShadowEntry &prior);

[[nodiscard]] ConSanMoiAtomicSyncResult
consan_moi_record_replay_atomic_release(std::span<ConSanMoiAtomicReleaseRecord> release_records,
                                        uint64_t generation, uint64_t atomic_address,
                                        uint32_t producer_owner_id, uint32_t producer_epoch,
                                        uint32_t release_instruction_offset);

[[nodiscard]] ConSanMoiAtomicSyncResult consan_moi_record_replay_atomic_acquire(
    std::span<const ConSanMoiAtomicReleaseRecord> release_records,
    std::span<ConSanMoiAcquiredEpochToken> acquired_epoch_tokens, uint64_t generation,
    uint64_t atomic_address, uint32_t consumer_owner_id, uint32_t acquire_instruction_offset);

[[nodiscard]] ConSanMoiRecordReplayResult
consan_moi_record_replay_access_records(ConSanMoiReportHeader &header,
                                        std::span<const ConSanMoiAccessRecord> access_records,
                                        std::span<ConSanMoiDiagnosticRecord> diagnostic_records,
                                        std::span<uint64_t> exact_shadow_entries);

[[nodiscard]] ConSanMoiRecordReplayResult
consan_moi_record_replay_access_records(ConSanMoiReportHeader &header,
                                        std::span<const ConSanMoiAccessRecord> access_records,
                                        std::span<const ConSanMoiBarrierRecord> barrier_records,
                                        std::span<ConSanMoiDiagnosticRecord> diagnostic_records,
                                        std::span<uint64_t> exact_shadow_entries);

[[nodiscard]] ConSanMoiRecordReplayResult consan_moi_record_replay_access_records(
    ConSanMoiReportHeader &header, std::span<const ConSanMoiAccessRecord> access_records,
    std::span<const ConSanMoiBarrierRecord> barrier_records,
    std::span<const ConSanMoiRecordReplayAtomicEvent> atomic_events,
    std::span<ConSanMoiDiagnosticRecord> diagnostic_records,
    std::span<uint64_t> exact_shadow_entries);

[[nodiscard]] ConSanMoiSampledPublishResult
consan_moi_sampled_publish_access_records(const ConSanMoiReportHeader &header,
                                          std::span<const ConSanMoiAccessRecord> access_records,
                                          std::span<uint64_t> sampled_watchpoint_entries);

[[nodiscard]] ConSanMoiSampledReplayResult
consan_moi_sampled_replay_entries(ConSanMoiReportHeader &header,
                                  std::span<const uint64_t> sampled_watchpoint_entries,
                                  std::span<ConSanMoiDiagnosticRecord> diagnostic_records);

[[nodiscard]] bool consan_moi_supports_native_lds_record_replay_mnemonic(std::string_view mnemonic);

[[nodiscard]] ConSanResult try_patch_consan_moi(ConSanResult result, const ConSanOptions &options,
                                                std::span<const uint8_t> code_object_bytes,
                                                rj_code_arch_t arch);

} // namespace rocjitsu
