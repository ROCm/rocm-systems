// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file sdma_executor.h
/// @brief Transport-neutral, resumable AMD SDMA packet execution.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_vm.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>

namespace rocjitsu::amdgpu {

enum class SdmaPacketDialect {
  Legacy,
  Gfx11Plus,
  Gfx1250,
};

enum class SdmaExecutionOutcome {
  Complete,
  Unavailable,
  Faulted,
  Malformed,
};

enum class SdmaCacheOperation {
  WritebackInvalidate,
  Invalidate,
};

/// Result of starting or resuming one root-ring packet.
struct SdmaExecutionResult {
  SdmaExecutionOutcome outcome = SdmaExecutionOutcome::Malformed;
  /// Root-ring packet length. Valid once the root header has been decoded.
  std::size_t packet_dwords = 0;
  /// The packet's externally visible memory/register/interrupt operation ran.
  bool operation_committed = false;
  /// All completion publication requested by the packet has finished.
  bool completion_published = false;
  /// Whether the adapter should advance past this packet before halting.
  bool retire_packet = false;
};

/// Narrow environment hooks for operations outside the GPU virtual address space.
struct SdmaExecutorCallbacks {
  /// Non-Complete must mean that no externally visible callback effect occurred.
  /// The executor may invoke the callback again after Unavailable.
  /// Evaluate a register poll. Complete means satisfied, Unavailable means wait.
  /// This lets a functional adapter intentionally accept register polls without
  /// fabricating a register value, while a PCI adapter can perform a real read.
  std::function<VmAccessOutcome(uint32_t address, uint32_t reference, uint32_t mask,
                                uint32_t function)>
      poll_register;
  std::function<VmAccessOutcome(uint32_t address, uint32_t value)> write_register;
  std::function<VmAccessOutcome(uint32_t data)> deliver_interrupt;
  /// Cache maintenance is invoked at most once for each packet operation.
  std::function<void(SdmaCacheOperation operation)> maintain_caches;
  std::function<uint64_t()> timestamp;
};

/// Executes the union of SDMA packets used by legacy CP and PCI startup queues.
///
/// One instance belongs to one queue. It retains an immutable GpuVmAccess and
/// byte/atomic/publication progress while a packet is unavailable, so resuming
/// never replays already committed work. A fresh packet may only be started
/// when pending() is false.
class SdmaExecutor {
public:
  static constexpr std::size_t kMaxIndirectDepth = 4;

  explicit SdmaExecutor(SdmaPacketDialect dialect, SdmaExecutorCallbacks callbacks = {});
  ~SdmaExecutor();

  SdmaExecutor(const SdmaExecutor &) = delete;
  SdmaExecutor &operator=(const SdmaExecutor &) = delete;
  SdmaExecutor(SdmaExecutor &&) noexcept;
  SdmaExecutor &operator=(SdmaExecutor &&) noexcept;

  /// Copy and begin the first packet in @p available_dwords.
  [[nodiscard]] SdmaExecutionResult start(std::span<const uint32_t> available_dwords,
                                          GpuVmAccess access);
  /// Continue the packet retained after an Unavailable result.
  [[nodiscard]] SdmaExecutionResult resume();

  [[nodiscard]] bool pending() const;
  /// Cancel all retained packet, VM snapshot, IB, and progress state.
  /// Already committed external effects are not rolled back.
  void reset();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace rocjitsu::amdgpu
