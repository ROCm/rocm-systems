// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file sdma_queue_runner.h
/// @brief Transport-neutral SDMA root-ring execution.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/sdma_executor.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace rocjitsu::amdgpu {

enum class SdmaQueueServiceOutcome : uint8_t {
  Drained,
  Unavailable,
  Faulted,
  Malformed,
};

/// Configuration whose lifetime is owned by one SDMA queue.
struct SdmaQueueRunnerConfig {
  AddressSpaceHandle address_space;
  uint64_t ring_base = 0;
  uint64_t ring_bytes = 0;
  uint64_t read_pointer_address = 0;
  /// Device-side cursor supplied by an MQD or register file. When absent, the
  /// runner acquires the initial cursor from read_pointer_address.
  std::optional<uint64_t> initial_cursor = std::nullopt;

  friend bool operator==(const SdmaQueueRunnerConfig &, const SdmaQueueRunnerConfig &) = default;
};

/// Executes one SDMA root ring independently of its doorbell transport.
///
/// A service batch captures one immutable GpuVmAccess and retains it through
/// ring fetch, packet execution, and cursor publication. Unavailable work is
/// resumed without replaying completed fetch segments or packet effects.
class SdmaQueueRunner {
public:
  using Config = SdmaQueueRunnerConfig;
  using Outcome = SdmaQueueServiceOutcome;

  SdmaQueueRunner(GpuVm &gpu_vm, Config config, SdmaExecutor executor);
  SdmaQueueRunner(GpuVm &gpu_vm, Config config, SdmaPacketDialect dialect,
                  SdmaExecutorCallbacks callbacks = {});
  ~SdmaQueueRunner();

  SdmaQueueRunner(const SdmaQueueRunner &) = delete;
  SdmaQueueRunner &operator=(const SdmaQueueRunner &) = delete;
  SdmaQueueRunner(SdmaQueueRunner &&) noexcept;
  SdmaQueueRunner &operator=(SdmaQueueRunner &&) noexcept = delete;

  /// Run packets through @p producer, an absolute byte cursor.
  [[nodiscard]] Outcome service(uint64_t producer);

  /// Replace the queue layout only when no batch, packet, or publication is in flight.
  [[nodiscard]] bool reconfigure(Config config);
  /// Discard retained retry and terminal state while preserving the configuration.
  void reset();

  [[nodiscard]] bool in_flight() const;
  [[nodiscard]] uint64_t cursor() const { return cursor_; }
  [[nodiscard]] std::optional<Outcome> terminal() const { return terminal_; }
  [[nodiscard]] const Config &config() const { return config_; }

private:
  [[nodiscard]] std::optional<SdmaQueueServiceOutcome> validate_configuration() const;
  [[nodiscard]] SdmaQueueServiceOutcome initialize_cursor();
  [[nodiscard]] SdmaQueueServiceOutcome fetch(uint64_t producer);
  [[nodiscard]] SdmaQueueServiceOutcome publish_cursor();
  [[nodiscard]] SdmaQueueServiceOutcome latch(SdmaQueueServiceOutcome outcome);
  void clear_batch();

  GpuVm *gpu_vm_ = nullptr;
  SdmaQueueRunnerConfig config_;
  SdmaExecutor executor_;
  std::optional<GpuVmAccess> access_;
  uint64_t cursor_ = 0;
  bool cursor_initialized_ = false;
  bool batch_active_ = false;
  uint64_t fetch_end_cursor_ = 0;
  std::vector<std::byte> fetch_bytes_;
  std::size_t fetch_progress_ = 0;
  std::size_t fetch_segment_progress_ = 0;
  std::vector<uint32_t> fetched_words_;
  std::size_t fetched_word_offset_ = 0;
  bool publication_pending_ = false;
  std::optional<SdmaQueueServiceOutcome> terminal_after_publication_;
  std::optional<SdmaQueueServiceOutcome> terminal_;
};

} // namespace rocjitsu::amdgpu
