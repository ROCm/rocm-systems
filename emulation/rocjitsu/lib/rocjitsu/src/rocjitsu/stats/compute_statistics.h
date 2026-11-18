// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "util/statistics.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace rocjitsu {

/// @brief Observations made after wave-state updates, measured in functional scheduler steps.
class ComputeStatistics {
public:
  enum class BlockedReason {
    Waitcnt,
    Barrier,
    GwsWait,
    ExportWait,
    ParamWait,
    VmRetry,
    Ending,
    MemoryTrap,
    DebugPaused,
    RuntimePaused,
    StatusHalted,
    Sleep,
    NotReady,
    Count,
  };

  void bind(const std::shared_ptr<stats::Scope> &scope) {
    *this = ComputeStatistics{};
    if (!scope)
      return;
    steps_ = scope->counter("scheduler.steps", "steps", "Functional CU scheduler iterations",
                            "scheduler_steps");
    active_ = scope->counter("scheduler.active_wave_steps", "wave_steps",
                             "Active waves observed after state updates, including paused waves",
                             "scheduler_steps");
    constexpr std::array names{"waitcnt",      "barrier",        "gws_wait",      "export_wait",
                               "param_wait",   "vm_retry",       "ending",        "memory_trap",
                               "debug_paused", "runtime_paused", "status_halted", "sleep",
                               "not_ready"};
    constexpr std::array descriptions{
        "Wave observations still waiting on wait counters after state updates",
        "Wave observations still waiting at a barrier after state updates",
        "Wave observations still parked on global wave synchronization after state updates",
        "Wave observations still waiting for export-buffer allocation after state updates",
        "Wave observations still waiting for pixel parameters in LDS after state updates",
        "Wave observations still waiting for unavailable memory backing after state updates",
        "Wave observations still draining memory at program end after state updates",
        "Wave observations still awaiting memory trap delivery after state updates",
        "Wave observations prevented from issuing by a debugger pause",
        "Wave observations prevented from issuing by runtime suspension",
        "Wave observations prevented from issuing by architectural STATUS halt",
        "Running wave observations prevented from issuing by S_SLEEP",
        "Running wave observations prevented from issuing by modeled instruction readiness"};
    for (std::size_t index = 0; index < names.size(); ++index)
      blocked_[index] = scope->counter("scheduler." + std::string(names[index]) + "_wave_steps",
                                       "wave_steps", descriptions[index], "scheduler_steps");
    observes_waves_ = active_ != nullptr;
    for (const auto &counter : blocked_)
      observes_waves_ = observes_waves_ || counter != nullptr;
  }

  bool observes_waves() const noexcept { return observes_waves_; }
  void step() const noexcept {
    if (steps_)
      steps_->increment();
  }
  void active_wave() const noexcept {
    if (active_)
      active_->increment();
  }
  void blocked_wave(BlockedReason reason) const noexcept {
    if (const auto &counter = blocked_[static_cast<std::size_t>(reason)])
      counter->increment();
  }

private:
  std::shared_ptr<stats::RegisteredCounter> steps_;
  std::shared_ptr<stats::RegisteredCounter> active_;
  std::array<std::shared_ptr<stats::RegisteredCounter>,
             static_cast<std::size_t>(BlockedReason::Count)>
      blocked_{};
  bool observes_waves_ = false;
};

/// @brief Optional memory instruction lifecycle and retry queue accounting.
class MemoryPipelineStatistics {
public:
  enum class Terminal { Completed, Cancelled, Failed };

  void bind(const std::shared_ptr<stats::Scope> &scope, std::string_view pipeline) {
    *this = MemoryPipelineStatistics{};
    if (!scope)
      return;
    const std::string prefix = "memory." + std::string(pipeline) + ".";
    accepted_ = scope->counter(prefix + "accepted", "instructions",
                               "Instructions whose ownership entered the pipeline");
    completed_ = scope->counter(prefix + "completed", "instructions",
                                "Owned instructions completing architectural writeback");
    cancelled_ =
        scope->counter(prefix + "cancelled", "instructions",
                       "Owned instructions discarded for replay, cancellation or slot reuse");
    failed_ =
        scope->counter(prefix + "failed", "instructions",
                       "Owned instructions rejected by a permanent access or writeback failure");
    pending_ = scope->gauge(
        prefix + "pending", "instructions",
        "Accepted instructions without a terminal outcome, including deferred writeback");
    retry_attempts_ =
        scope->counter(prefix + "retry_attempts", "attempts",
                       "Access attempts after an instruction entered the unavailable queue");
    unavailable_attempts_ = scope->counter(prefix + "unavailable_attempts", "attempts",
                                           "Access attempts returning transient unavailability");
    retry_depth_ = scope->gauge(prefix + "retry_depth", "instructions",
                                "Instructions retained for retry after transient unavailability");
    track_lifecycle_ = accepted_ || completed_ || cancelled_ || failed_ || pending_;
  }

  bool track_lifecycle() const noexcept { return track_lifecycle_; }
  void accept() const noexcept {
    if (accepted_)
      accepted_->increment();
    if (pending_)
      pending_->add(1);
  }
  void finish(Terminal terminal) const noexcept {
    const auto &counter = terminal == Terminal::Completed   ? completed_
                          : terminal == Terminal::Cancelled ? cancelled_
                                                            : failed_;
    if (counter)
      counter->increment();
    if (pending_)
      pending_->add(-1);
  }
  void retry() const noexcept {
    if (retry_attempts_)
      retry_attempts_->increment();
  }
  void unavailable() const noexcept {
    if (unavailable_attempts_)
      unavailable_attempts_->increment();
  }
  void retry_depth(std::size_t depth) const noexcept {
    if (retry_depth_)
      retry_depth_->set(static_cast<int64_t>(depth));
  }

private:
  std::shared_ptr<stats::RegisteredCounter> accepted_;
  std::shared_ptr<stats::RegisteredCounter> completed_;
  std::shared_ptr<stats::RegisteredCounter> cancelled_;
  std::shared_ptr<stats::RegisteredCounter> failed_;
  std::shared_ptr<stats::RegisteredCounter> retry_attempts_;
  std::shared_ptr<stats::RegisteredCounter> unavailable_attempts_;
  std::shared_ptr<stats::RegisteredGauge> pending_;
  std::shared_ptr<stats::RegisteredGauge> retry_depth_;
  bool track_lifecycle_ = false;
};

} // namespace rocjitsu
