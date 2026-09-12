// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file pm4_ring_consumer.h
/// @brief Ring traversal for the supported PM4 compute-queue subset.

#pragma once

#include "rocjitsu/vm/amdgpu/circular_ring_reader.h"
#include "rocjitsu/vm/amdgpu/consumer_cursor_journal.h"
#include "rocjitsu/vm/amdgpu/gpu_queue_registry.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/pm4_packet_processor.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace rocjitsu::amdgpu {

enum class Pm4RingStatus : uint8_t {
  Complete,
  Blocked,
  Faulted,
  Malformed,
  UnsupportedPacket,
  RegisterWriteRejected,
};

struct Pm4RingResult {
  Pm4RingStatus status = Pm4RingStatus::Malformed;
  uint64_t read_pointer = 0;
  uint32_t packet_header = 0;
  uint64_t register_dword = 0;
};

/// @brief Owns one PM4 ring transaction, including cursor retirement and publication.
class Pm4RingConsumer {
public:
  Pm4RingConsumer(Pm4PacketProcessor &packet_processor, GpuVm &gpu_vm,
                  AddressSpaceHandle address_space, uint64_t ring_base, uint64_t ring_bytes,
                  uint64_t consumer_pointer_address,
                  std::optional<uint64_t> initial_consumer_cursor);

  Pm4RingConsumer(const Pm4RingConsumer &) = delete;
  Pm4RingConsumer &operator=(const Pm4RingConsumer &) = delete;
  Pm4RingConsumer(Pm4RingConsumer &&) = delete;
  Pm4RingConsumer &operator=(Pm4RingConsumer &&) = delete;

  /// @brief Consume through one producer boundary using one immutable VM snapshot.
  [[nodiscard]] Pm4RingResult
  consume(uint64_t producer_cursor,
          std::size_t packet_budget = std::numeric_limits<std::size_t>::max());

  /// @brief Whether retry-critical transaction or publication state is retained.
  [[nodiscard]] bool in_flight() const noexcept;
  [[nodiscard]] bool publication_pending() const noexcept;
  [[nodiscard]] uint64_t consumer_cursor() const noexcept;

  /// @brief Prepare for graceful removal without losing committed progress.
  [[nodiscard]] QueuePrepareCloseStatus prepare_close() noexcept;

  /// @brief Replace ring geometry after the owner has established quiescence.
  void reconfigure(uint64_t ring_base, uint64_t ring_bytes) noexcept;

  /// @brief Force-cancel retained transaction state and reset the ring cursor.
  void reset() noexcept;

private:
  [[nodiscard]] Pm4RingResult consume_packets(const GpuVmAccess &access, uint64_t read_pointer,
                                              uint64_t write_pointer,
                                              std::size_t packet_budget) const;
  void release_transaction() noexcept;

  Pm4PacketProcessor *packet_processor_ = nullptr;
  GpuVm *gpu_vm_ = nullptr;
  AddressSpaceHandle address_space_;
  uint64_t ring_base_ = 0;
  uint64_t ring_bytes_ = 0;
  ConsumerCursorJournal cursor_journal_;
  std::optional<GpuVmAccess> access_;
  std::optional<uint64_t> transaction_producer_cursor_;
  std::optional<Pm4RingResult> terminal_after_publication_;
  bool publication_faulted_ = false;
};

} // namespace rocjitsu::amdgpu
