// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file amd_hsa_signal.h
/// @brief Minimal AMD signal ABI mirror.
///
/// @details The source of truth is
/// `projects/rocr-runtime/runtime/hsa-runtime/inc/amd_hsa_signal.h`. This mirror
/// exposes just enough of the frozen `amd_signal_t` layout for rocjitsu to resolve
/// a doorbell signal back to its queue at dispatch time: a public `hsa_signal_t`
/// handle is the address of the embedded `amd_signal_t`, and a doorbell signal's
/// `queue_ptr` points at the `amd_queue_t` whose `hsa_queue` carries the ring base
/// and size. Keep the field order and the layout assertions in lock-step with the
/// authoritative header.

#pragma once

#include "hsa/amd_hsa_queue.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rocjitsu::amdgpu {

/// @brief AMD signal kind. A doorbell signal carries a back-pointer to its queue.
enum AmdSignalKind : int64_t {
  kAmdSignalKindInvalid = 0,
  kAmdSignalKindUser = 1,
  kAmdSignalKindDoorbell = -1,
  kAmdSignalKindLegacyDoorbell = -2,
};

/// @brief Frozen 64-byte AMD signal layout (mirror of `amd_signal_t`).
struct AmdSignal {
  int64_t kind;
  union {
    volatile int64_t value;
    volatile uint64_t *hardware_doorbell_ptr;
  };
  uint64_t event_mailbox_ptr;
  uint32_t event_id;
  uint32_t reserved1;
  uint64_t start_ts;
  uint64_t end_ts;
  union {
    /// Non-null on a doorbell signal: points at the owning queue. Typed as the
    /// base `::amd_queue_t` here (its `hsa_queue` prefix is all rocjitsu reads).
    ::amd_queue_t *queue_ptr;
    uint64_t reserved2;
  };
  uint32_t reserved3[2];
};

static_assert(std::is_trivially_copyable_v<AmdSignal>);
static_assert(sizeof(AmdSignal) == 64);
static_assert(offsetof(AmdSignal, event_mailbox_ptr) == 16);
static_assert(offsetof(AmdSignal, event_id) == 24);
static_assert(offsetof(AmdSignal, start_ts) == 32);
static_assert(offsetof(AmdSignal, end_ts) == 40);
static_assert(offsetof(AmdSignal, queue_ptr) == 48);

} // namespace rocjitsu::amdgpu
