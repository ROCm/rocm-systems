// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file sdma_packet_processor.h
/// @brief Transport-neutral AMD SDMA packet processing.

#pragma once

#include "rocjitsu/vm/amdgpu/device_cache_coherence.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/packet_processor.h"

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

using SdmaCacheOperation = DeviceCacheOperation;
using SdmaCacheLease = DeviceCacheMaintenanceLease;

/// @brief Queue-owned semantic state for one partially executed SDMA packet.
/// @details The ring consumer owns this object alongside its immutable VM
/// snapshot. Only SdmaPacketProcessor interprets or mutates the opaque state.
class SdmaPacketContinuation {
public:
  SdmaPacketContinuation();
  ~SdmaPacketContinuation();

  SdmaPacketContinuation(const SdmaPacketContinuation &) = delete;
  SdmaPacketContinuation &operator=(const SdmaPacketContinuation &) = delete;
  SdmaPacketContinuation(SdmaPacketContinuation &&) noexcept;
  SdmaPacketContinuation &operator=(SdmaPacketContinuation &&) noexcept;

  [[nodiscard]] bool pending() const noexcept;

  class Impl;

private:
  std::unique_ptr<Impl> impl_;
  friend class SdmaPacketProcessor;
};

/// @brief Typed input for starting or continuing one SDMA root-ring packet.
struct SdmaPacketProcessRequest {
  std::span<const uint32_t> available_dwords;
  const GpuVmAccess &access;
  SdmaPacketContinuation &continuation;
};

/// @brief Common packet disposition plus SDMA-specific progress diagnostics.
struct SdmaPacketProcessResult {
  [[nodiscard]] const PacketProcessResult &packet_result() const noexcept { return packet; }

  PacketProcessResult packet;
  bool operation_committed = false;
  bool completion_published = false;
};

/// @brief Result of reading one frontend register for an SDMA poll packet.
struct SdmaPacketRegisterReadResult {
  VmAccessOutcome outcome = VmAccessOutcome::Faulted;
  uint32_t value = 0;
};

/// @brief Narrow environment hooks for operations outside the GPU virtual address space.
class SdmaPacketCallbacks {
public:
  /// Non-Complete must mean that no externally visible callback effect occurred.
  /// The processor may invoke the callback again after a blocked attempt.
  /// Read a register value without interpreting the poll comparison function.
  /// An absent callback lets a functional adapter accept register polls without
  /// fabricating register state; transport adapters provide raw reads and the
  /// processor applies the packet's mask and comparison exactly once.
  std::function<SdmaPacketRegisterReadResult(uint32_t address)> read_register;
  std::function<VmAccessOutcome(uint32_t address, uint32_t value)> write_register;
  std::function<VmAccessOutcome(uint32_t data)> deliver_interrupt;
  /// The returned lease spans every direct memory effect. It is released when
  /// execution yields and reacquired before the effect resumes.
  std::function<SdmaCacheLease(SdmaCacheOperation operation)> acquire_cache_maintenance;
  std::function<uint64_t()> timestamp;
};

/// @brief Processes the union of SDMA packets used by legacy and PCI queues.
/// @details The processor owns protocol interpretation and environment callbacks,
/// while its caller owns the semantic continuation and immutable VM snapshot.
/// Supplying a pending continuation resumes the packet without replaying effects.
class SdmaPacketProcessor {
public:
  using Request = SdmaPacketProcessRequest;
  using Result = SdmaPacketProcessResult;

  static constexpr std::size_t kMaxIndirectDepth = 4;

  explicit SdmaPacketProcessor(SdmaPacketDialect dialect, SdmaPacketCallbacks callbacks = {});
  ~SdmaPacketProcessor();

  SdmaPacketProcessor(const SdmaPacketProcessor &) = delete;
  SdmaPacketProcessor &operator=(const SdmaPacketProcessor &) = delete;
  SdmaPacketProcessor(SdmaPacketProcessor &&) noexcept;
  SdmaPacketProcessor &operator=(SdmaPacketProcessor &&) noexcept;

  /// Inspect, begin, or continue only the first packet in @p request.
  /// @details NeedInput reports the bounded prefix required to determine or
  /// execute a fresh packet. Any later packets in the span are never retained.
  [[nodiscard]] Result process(Request request);
  /// Cancel retained packet, IB, and exact-once progress state.
  /// Already committed external effects are not rolled back.
  void reset(SdmaPacketContinuation &continuation) noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

static_assert(PacketProcessor<SdmaPacketProcessor>);

} // namespace rocjitsu::amdgpu
