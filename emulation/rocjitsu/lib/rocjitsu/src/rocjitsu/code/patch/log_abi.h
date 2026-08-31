// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file log_abi.h
/// @brief Fixed host/device ABI for the DBI logging buffer.
///
/// This header is intentionally dependency-light: it pulls in only <cstddef>
/// and <cstdint> so the exact same struct definitions can be included from
/// host C++ and from HIP device code (the `rj_log_write` probe fixture). Keep
/// it that way -- do not add host-only includes here. The host ring-buffer
/// class and drain logic live in log_buffer.h.
///
/// The type names are currently unversioned, but versioned type names can be
/// introduced to extend the ABI while keeping compatibility.
///
/// Layout handshake: the host consumer strides by a fixed sizeof(RjLogRecord), so
/// both sides must agree on record geometry first. A producer compares the
/// header's abi_version/record_size before writing; the host passes the header
/// through LogBuffer::validate() before draining (see log_buffer.h).

#pragma once

#include <cstddef>
#include <cstdint>

namespace rocjitsu {

/// @brief Magic value stamped into RjLogBufferHeader::magic ("RJLG",
///        little-endian byte order 'R','J','L','G').
inline constexpr uint32_t kRjLogMagic =
    static_cast<uint32_t>('R') | (static_cast<uint32_t>('J') << 8) |
    (static_cast<uint32_t>('L') << 16) | (static_cast<uint32_t>('G') << 24);

/// @brief Current logging ABI version.
inline constexpr uint16_t kRjLogAbiVersion = 0;

/// @brief Fixed record size in bytes (power of two).
inline constexpr uint32_t kRjLogRecordSize = 64;

/// @brief Control header for a logging ring buffer.
///
/// The three monotonic counters (@c write_ptr, @c read_ptr, @c overflow_count)
/// are each placed on their own 64-byte cache line to avoid false sharing
/// between GPU producers and the host consumer. Counters are record *indices*
/// (not byte offsets); the live slot for index @c i is @c i&(slot_count-1).
struct alignas(64) RjLogBufferHeader {
  // Cache line 0: immutable control fields after initialization.
  uint32_t magic;       ///< kRjLogMagic.
  uint16_t abi_version; ///< kRjLogAbiVersion.
  uint16_t header_size; ///< sizeof(RjLogBufferHeader).
  uint32_t record_size; ///< kRjLogRecordSize.
  uint32_t slot_count;  ///< Number of records; power of two.
  uint64_t flags;       ///< Reserved; must be zero for now. Readers ignore unknown bits.
  uint8_t reserved_control[40];

  // Cache line 1: producer-owned hot counter.
  uint64_t write_ptr; ///< Next record index to claim.
  uint8_t reserved_producer[56];

  // Cache line 2: consumer-owned counter.
  uint64_t read_ptr; ///< Next record index to drain.
  uint8_t reserved_consumer[56];

  // Cache line 3: diagnostics written by producers, read by host after drain.
  uint64_t overflow_count; ///< Records dropped because the ring was full.
  uint8_t reserved_diagnostics[56];
};

static_assert(sizeof(RjLogBufferHeader) == 256);
static_assert(alignof(RjLogBufferHeader) == 64);
// Version discriminant offsets are FROZEN for all ABI versions: a host must be
// able to read magic/abi_version/header_size at these fixed positions before it
// can decide whether it understands the rest of the layout. Never move them.
static_assert(offsetof(RjLogBufferHeader, magic) == 0);
static_assert(offsetof(RjLogBufferHeader, abi_version) == 4);
static_assert(offsetof(RjLogBufferHeader, header_size) == 6);
static_assert(offsetof(RjLogBufferHeader, record_size) == 8);
static_assert(offsetof(RjLogBufferHeader, slot_count) == 12);
static_assert(offsetof(RjLogBufferHeader, flags) == 16);
static_assert(offsetof(RjLogBufferHeader, write_ptr) == 64);
static_assert(offsetof(RjLogBufferHeader, read_ptr) == 128);
static_assert(offsetof(RjLogBufferHeader, overflow_count) == 192);

/// @brief One logging record.
///
/// `payload` is kept last because it is the field most likely to grow in a
/// future ABI version; trailing it lets the payload region expand without
/// shifting the fixed metadata offsets ahead of it.
struct alignas(64) RjLogRecord {
  uint32_t valid;       ///< 0 = free, 1 = record written.
  uint16_t abi_version; ///< kRjLogAbiVersion.
  uint16_t reserved0;   ///< Reserved; must be zero for now.
  uint32_t record_size; ///< kRjLogRecordSize.
  uint32_t record_type; ///< Tool-defined record kind.
  uint64_t exec_mask;   ///< Original EXEC mask at the probe site.
  uint32_t site;        ///< Original .text byte offset of the anchor.
  uint32_t wave_id;     ///< 0 if unavailable.
  uint32_t workgroup_x; ///< 0 if unavailable.
  uint32_t workgroup_y; ///< 0 if unavailable.
  uint32_t workgroup_z; ///< 0 if unavailable.
  uint32_t active_lane_count;
  uint32_t writer_lane; ///< First active lane, or 0xffffffff if unknown.
  uint32_t reserved1;   ///< Reserved; must be zero for now.
  uint64_t payload;     ///< Tool-defined payload. Kept last (see above).
};

static_assert(sizeof(RjLogRecord) == 64);
static_assert(alignof(RjLogRecord) == 64);
static_assert(sizeof(RjLogRecord) == kRjLogRecordSize);
// Pin every field offset: a device writes this record, so a middle-field
// reorder or width change that preserved the 64-byte total would otherwise
// compile clean and silently desync the host reader from the device writer.
static_assert(offsetof(RjLogRecord, valid) == 0);
static_assert(offsetof(RjLogRecord, abi_version) == 4);
static_assert(offsetof(RjLogRecord, reserved0) == 6);
static_assert(offsetof(RjLogRecord, record_size) == 8);
static_assert(offsetof(RjLogRecord, record_type) == 12);
static_assert(offsetof(RjLogRecord, exec_mask) == 16);
static_assert(offsetof(RjLogRecord, site) == 24);
static_assert(offsetof(RjLogRecord, wave_id) == 28);
static_assert(offsetof(RjLogRecord, workgroup_x) == 32);
static_assert(offsetof(RjLogRecord, workgroup_y) == 36);
static_assert(offsetof(RjLogRecord, workgroup_z) == 40);
static_assert(offsetof(RjLogRecord, active_lane_count) == 44);
static_assert(offsetof(RjLogRecord, writer_lane) == 48);
static_assert(offsetof(RjLogRecord, reserved1) == 52);
static_assert(offsetof(RjLogRecord, payload) == 56);

} // namespace rocjitsu
