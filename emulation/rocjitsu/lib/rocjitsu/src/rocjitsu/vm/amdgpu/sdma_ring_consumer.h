// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file sdma_ring_consumer.h
/// @brief Transport-neutral SDMA root-ring execution.

#pragma once

#include "rocjitsu/vm/amdgpu/circular_ring_reader.h"
#include "rocjitsu/vm/amdgpu/consumer_cursor_journal.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/sdma_packet_processor.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace rocjitsu::amdgpu {

enum class SdmaRingStatus : uint8_t {
  Idle,
  Runnable,
  Blocked,
  Faulted,
  Malformed,
};

/// @brief Configuration whose lifetime is owned by one SDMA queue.
class SdmaRingConfig {
public:
  AddressSpaceHandle address_space;
  uint64_t ring_base = 0;
  uint64_t ring_bytes = 0;
  uint64_t read_pointer_address = 0;
  /// Device-side cursor supplied by an MQD or register file. When absent, the
  /// processor acquires the initial cursor from read_pointer_address.
  std::optional<uint64_t> initial_cursor = std::nullopt;

  friend bool operator==(const SdmaRingConfig &, const SdmaRingConfig &) = default;
};

/// @brief Executes one SDMA root ring independently of its doorbell transport.
/// @details Each root packet captures one immutable GpuVmAccess and retains it
/// through bounded ring fetch, packet execution, and cursor publication.
/// Blocked work is resumed without replaying completed fetch segments or packet
/// effects.
class SdmaRingConsumer {
public:
  using Config = SdmaRingConfig;
  using Status = SdmaRingStatus;

  SdmaRingConsumer(GpuVm &gpu_vm, Config config, SdmaPacketProcessor packet_processor);
  SdmaRingConsumer(GpuVm &gpu_vm, Config config, SdmaPacketDialect dialect,
                   SdmaPacketCallbacks callbacks = {});
  ~SdmaRingConsumer();

  SdmaRingConsumer(const SdmaRingConsumer &) = delete;
  SdmaRingConsumer &operator=(const SdmaRingConsumer &) = delete;
  SdmaRingConsumer(SdmaRingConsumer &&) noexcept;
  SdmaRingConsumer &operator=(SdmaRingConsumer &&) noexcept = delete;

  /// @brief Run packets through @p producer, an absolute byte cursor.
  /// @details At most @p packet_budget root-ring packets are retired in one
  /// call. Runnable reports that runnable work remains and lets the owning scheduler
  /// service another queue before resuming this one.
  [[nodiscard]] Status service(uint64_t producer,
                               std::size_t packet_budget = std::numeric_limits<std::size_t>::max());

  /// @brief Replace the queue layout when no operation is in flight.
  [[nodiscard]] bool reconfigure(Config config);
  /// @brief Discard retry and terminal state while preserving the configuration.
  void reset();

  [[nodiscard]] bool in_flight() const;
  [[nodiscard]] bool publication_pending() const { return cursor_journal_.publication_pending(); }
  [[nodiscard]] uint64_t cursor() const { return cursor_journal_.cursor(); }
  [[nodiscard]] std::optional<Status> terminal() const { return terminal_; }
  [[nodiscard]] const Config &config() const { return config_; }

private:
  [[nodiscard]] std::optional<SdmaRingStatus> validate_configuration() const;
  [[nodiscard]] SdmaRingStatus initialize_cursor();
  [[nodiscard]] SdmaRingStatus ensure_packet_bytes(uint64_t producer, std::size_t required_bytes);
  [[nodiscard]] SdmaRingStatus publish_cursor();
  [[nodiscard]] SdmaRingStatus latch(SdmaRingStatus outcome);
  void clear_packet();
  void clear_in_flight();

  GpuVm *gpu_vm_ = nullptr;
  SdmaRingConfig config_;
  CircularRingReader ring_reader_;
  ConsumerCursorJournal cursor_journal_;
  SdmaPacketProcessor packet_processor_;
  SdmaPacketContinuation continuation_;
  std::optional<GpuVmAccess> access_;
  std::vector<std::byte> packet_bytes_;
  std::size_t packet_fetch_progress_ = 0;
  std::vector<uint32_t> packet_words_;
  std::optional<SdmaPacketProcessResult> pending_retirement_;
  std::optional<SdmaRingStatus> terminal_after_publication_;
  std::optional<SdmaRingStatus> terminal_;
};

} // namespace rocjitsu::amdgpu
