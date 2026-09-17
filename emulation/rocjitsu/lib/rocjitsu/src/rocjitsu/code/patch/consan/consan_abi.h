// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_abi.h
/// @brief Host/HIP-safe POD definitions for the ConSan report ABI.

#pragma once

#include <cstdint>

namespace rocjitsu::consan {

enum class ShadowAccessKind : uint32_t {
  Empty = 0,
  Read = 1,
  Write = 2,
  ReadWrite = 3,
  Atomic = 4,
};

enum class AtomicEventKind : uint32_t {
  Release = 1,
  Acquire = 2,
  AcquireRelease = 3,
};

enum class AtomicOperation : uint8_t {
  Rmw = 1,
  CompareExchange = 2,
};

enum class AtomicOutcome : uint16_t {
  NotApplicable = 0,
  Success = 1,
  Failure = 2,
  Unavailable = 3,
};

enum class FenceEventKind : uint8_t {
  Release = 1,
  Acquire = 2,
  AcquireRelease = 3,
};

inline constexpr uint32_t kReportMagic = 0x494f4d43u; // "CMOI" little-endian.
inline constexpr uint32_t kReportAbiVersion = 14;
struct alignas(8) ReportHeader {
  uint32_t magic = kReportMagic;
  uint32_t abi_version = kReportAbiVersion;
  uint32_t header_size = 0;
  uint32_t flags = 0;
  uint64_t generation = 0;
  uint64_t dispatch_id = 0;
  uint32_t event_counter = 0;
  uint32_t watchpoint_capacity = 0;
  uint32_t causal_window_capacity = 0;
  uint32_t causal_window_count = 0;
  uint32_t malformed_window_count = 0;
  uint32_t dropped_window_count = 0;
  uint32_t saturated_window_count = 0;
  uint32_t sync_metadata_capacity = 0;
  uint32_t sync_metadata_count = 0;
  uint32_t unsupported_sync_count = 0;
  uint32_t malformed_sync_count = 0;
  uint32_t pending_acquire_capacity = 0;
  uint32_t pending_acquire_count = 0;
  uint32_t pending_acquire_contention_count = 0;
  uint32_t pending_acquire_collision_count = 0;
  uint32_t pending_acquire_malformed_count = 0;
};

struct alignas(8) CausalWindow {
  uint64_t generation = 0;
  uint64_t dispatch_id = 0;
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  uint32_t epoch = 0;
  uint32_t first_entry = 0;
  uint32_t entry_count = 0;
  uint32_t publication_state = 0;
  // Zero for ordinary dispatches. On CDNA5 clustered kernels this carries
  // the launch-provided workgroup-within-cluster identity.
  uint32_t cluster_workgroup_id = 0;
  // Zero means unavailable. Published with the winning access under the
  // window claim, before Ready; reset with the rest of the report epoch.
  uint64_t exact_lane_mask = 0;
};

struct alignas(8) SyncMetadataPacked {
  uint64_t address = 0;
  uint32_t byte_count = 0;
  uint32_t descriptor = 0;
  uint32_t epoch_before = 0;
  uint32_t epoch_after = 0;

  [[nodiscard]] constexpr bool operator==(const SyncMetadataPacked &) const = default;
};

struct alignas(8) PendingAcquireSlot {
  uint32_t version = 0;
  uint32_t selected_slot = 0;
  uint64_t generation = 0;
  uint64_t dispatch_id = 0;
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
  uint32_t owner_id = 0;
  uint32_t source_epoch = 0;
  // One plus the absolute causal-window slot statically associated with the
  // release half of an acquire-release RMW. Zero means acquire-only.
  uint32_t reserved = 0;
  SyncMetadataPacked metadata;

  [[nodiscard]] constexpr bool operator==(const PendingAcquireSlot &) const = default;
};

static_assert(sizeof(ReportHeader) == 96);
static_assert(sizeof(CausalWindow) == 56);
static_assert(sizeof(SyncMetadataPacked) == 24);
static_assert(sizeof(PendingAcquireSlot) == 72);

} // namespace rocjitsu::consan
