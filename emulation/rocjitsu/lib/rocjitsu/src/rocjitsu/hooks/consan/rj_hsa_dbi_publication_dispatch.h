// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace rocjitsu::consan::hook {

/// Conservative lifetime check for a journal that covers one kernel dispatch.
/// Mutations are serialized by the queue-submission lock; report readers only
/// read the atomic result. Any possible overlap involving an instrumented
/// dispatch disables publication proofs until the registry is reset.
class PublicationDispatchIsolation {
public:
  template <typename LoadSignal>
  void note(uintptr_t queue, bool orders_prior, uint64_t signal, bool instrumented,
            LoadSignal load) {
    bool inherited_instrumentation = false;
    std::optional<int64_t> inherited_target;
    std::erase_if(pending_, [&](const Pending &prior) {
      const auto value = prior.signal ? load(prior.signal) : std::nullopt;
      if (value && *value <= prior.target)
        return true;
      if (queue && orders_prior && prior.queue == queue) {
        // Submission ordering is not completion. Keep the earlier kernel's
        // lifetime until this successor completes, even if it is uninstrumented.
        inherited_instrumentation |= prior.instrumented;
        if (signal && prior.signal == signal)
          inherited_target =
              inherited_target ? std::min(*inherited_target, prior.target) : prior.target;
        return true;
      }
      return false;
    });
    if (std::ranges::any_of(
            pending_, [&](const Pending &prior) { return instrumented || prior.instrumented; }))
      overlap_ = true;
    observed_ |= instrumented;
    Pending next{queue, signal, 0, instrumented || inherited_instrumentation};
    const auto value = signal ? load(signal) : std::nullopt;
    if (!value || *value == std::numeric_limits<int64_t>::min()) {
      next.signal = 0; // Only a later ordering packet can retire this dispatch.
    } else {
      // Requiring the entire countdown to finish also handles a shared signal.
      next.target = std::min<int64_t>(*value - 1, 0);
      if (inherited_target) {
        if (*inherited_target == std::numeric_limits<int64_t>::min()) {
          next.signal = 0;
          overlap_ = true;
        } else {
          next.target = std::min(next.target, *inherited_target - 1);
        }
      }
      for (const auto &prior : pending_) {
        if (prior.signal != signal)
          continue;
        if (prior.target == std::numeric_limits<int64_t>::min()) {
          next.signal = 0;
          overlap_ = true;
          break;
        }
        next.target = std::min(next.target, prior.target - 1);
      }
      if (next.signal)
        for (auto &prior : pending_)
          if (prior.signal == signal)
            prior.target = next.target;
    }
    pending_.push_back(next);
    isolated_.store(observed_ && !overlap_, std::memory_order_release);
  }

  template <typename LoadSignal> void forget(uint64_t signal, LoadSignal load) {
    const auto value = load(signal);
    std::erase_if(pending_, [&](const Pending &prior) {
      return prior.signal == signal && value && *value <= prior.target;
    });
    for (auto &prior : pending_)
      if (prior.signal == signal)
        prior.signal = 0; // Never read a destroyed HSA signal.
  }

  [[nodiscard]] bool isolated() const { return isolated_.load(std::memory_order_acquire); }
  void reset() {
    pending_.clear();
    observed_ = overlap_ = false;
    isolated_.store(false, std::memory_order_release);
  }

private:
  struct Pending {
    uintptr_t queue;
    uint64_t signal;
    int64_t target;
    bool instrumented;
  };
  std::vector<Pending> pending_;
  bool observed_ = false;
  bool overlap_ = false;
  std::atomic<bool> isolated_{false};
};

inline PublicationDispatchIsolation &publication_dispatch_isolation() {
  static PublicationDispatchIsolation state;
  return state;
}
} // namespace rocjitsu::consan::hook
