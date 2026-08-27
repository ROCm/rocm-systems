// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dram_des.h
/// @brief One memory channel, and the fabric crossing in front of it.

#pragma once

#include "rocjitsu/vm/timing/timed_component.h"
#include "rocjitsu/vm/timing/tuning.h"

#include <atomic>
#include <cstdint>
#include <functional>

namespace rocjitsu::timing {

/// @brief A bandwidth-limited server with a fixed unloaded latency.
///
/// @details Used for the fabric crossing and for DRAM. Neither models banks:
/// replacing a bandwidth-limited queue with a full bank, bank-group and refresh
/// state machine moved the GPU-visible bandwidth in the reference simulator by
/// 8.5 per cent against a 6.7-fold gap, because what limits a GPU's memory
/// system is almost never the DRAM device's own timing. Queueing delay is
/// represented by the server being busy, which is what a request arriving
/// behind others actually experiences.
class ChannelDes final : public TimedComponent {
public:
  using Downstream = std::function<void(const MemoryRequest &)>;
  using Completion = std::function<void(const MemoryRequest &, std::uint64_t tick)>;

  ChannelDes(std::string name, const simdojo::ClockDomain &domain, TimingEngine &engine,
             double bytes_per_cycle, std::uint64_t latency_cycles,
             std::uint64_t row_miss_cycles = 0);

  void set_downstream(Downstream downstream) { downstream_ = std::move(downstream); }
  void set_completion(Completion completion) { completion_ = std::move(completion); }

  std::uint64_t advance(std::uint64_t now) override;

  std::uint64_t bytes() const { return bytes_.load(std::memory_order_relaxed); }
  std::uint64_t requests() const { return requests_.load(std::memory_order_relaxed); }
  /// @brief Row activations this channel performed, for the report.
  std::uint64_t activations() const { return activations_.load(std::memory_order_relaxed); }

private:
  double bytes_per_cycle_ = 1.0;
  std::uint64_t latency_cycles_ = 1;
  /// @brief Extra cycles the channel spends closing one row and opening
  ///        another. Zero disables the row model entirely.
  std::uint64_t row_miss_cycles_ = 0;
  Downstream downstream_;
  Completion completion_;
  std::atomic<std::uint64_t> bytes_{0};
  std::atomic<std::uint64_t> requests_{0};
  std::atomic<std::uint64_t> activations_{0};
};

} // namespace rocjitsu::timing
