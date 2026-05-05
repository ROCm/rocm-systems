// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/completion_tracker.h"

#include "util/log.h"

#include <atomic>
#include <cstring>
#include <format>

namespace rocjitsu {
namespace amdgpu {

void CompletionTracker::register_dispatch(uint32_t dispatch_id, size_t queue_idx) {
  dispatch_queue_map_[dispatch_id] = queue_idx;
}

void CompletionTracker::notify_wg_complete(uint32_t dispatch_id, uint32_t wg_id,
                                           std::vector<HwQueueState> &queues) {
  auto it = dispatch_queue_map_.find(dispatch_id);
  if (it == dispatch_queue_map_.end())
    return;

  size_t qi = it->second;
  if (qi >= queues.size())
    return;

  auto &qs = queues[qi];
  for (auto &entry : qs.entries) {
    if (entry.dispatch_id == dispatch_id) {
      ++entry.completed_wgs;
      util::Logger::vm([&](auto &os) {
        os << std::format("CT: wg_complete d={} wg={} completed={}/{}", dispatch_id, wg_id,
                          entry.completed_wgs, entry.total_wgs);
      });
      break;
    }
  }

  drain_completions(queues);
}

void CompletionTracker::drain_completions(std::vector<HwQueueState> &queues) {
  for (auto &qs : queues) {
    while (!qs.entries.empty() && qs.entries.front().fully_completed()) {
      auto &entry = qs.entries.front();

      flush_caches();
      if (entry.completion_signal != 0) {
        fire_signal(entry);
      }

      dispatch_queue_map_.erase(entry.dispatch_id);

      // Adjust next_dispatch_idx since we're popping from the front.
      if (qs.next_dispatch_idx > 0)
        --qs.next_dispatch_idx;

      qs.entries.pop_front();
    }
  }
}

void CompletionTracker::flush_caches() {
  for (auto *cu : cus_)
    cu->flush_all();
}

bool CompletionTracker::all_complete(const std::vector<HwQueueState> &queues) const {
  for (const auto &qs : queues) {
    if (!qs.entries.empty())
      return false;
  }
  return true;
}

void CompletionTracker::fire_signal(const DispatchEntry &entry) {
  util::Logger::vm([&](auto &os) {
    os << std::format("CT: fire_signal d={} sig={:#x}", entry.dispatch_id, entry.completion_signal);
  });
  constexpr uint32_t SIG_VAL_OFF = 8;
  constexpr uint32_t MAILBOX_PTR_OFF = 16;
  constexpr uint32_t EVENT_ID_OFF = 24;

  if (entry.host_signal) {
    auto mailbox_ptr = *reinterpret_cast<uint64_t *>(entry.completion_signal + MAILBOX_PTR_OFF);
    auto event_id = *reinterpret_cast<uint32_t *>(entry.completion_signal + EVENT_ID_OFF);
    if (mailbox_ptr != 0) {
      std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(mailbox_ptr))
          .store(uint64_t(event_id), std::memory_order_release);
    }
    auto *val = reinterpret_cast<int64_t *>(entry.completion_signal + SIG_VAL_OFF);
    std::atomic_ref<int64_t>(*val).fetch_sub(1, std::memory_order_release);

    util::Logger::vm([&](auto &os) {
      os << std::format("CT: fire_signal d={} sig={:#x} event_id={}", entry.dispatch_id,
                        entry.completion_signal, event_id);
    });

    if (interrupt_cb_)
      interrupt_cb_(event_id);
  } else if (memory_) {
    auto old = static_cast<int64_t>(memory_->read64(entry.completion_signal + SIG_VAL_OFF));
    memory_->write64(entry.completion_signal + SIG_VAL_OFF, static_cast<uint64_t>(old - 1));
  }
}

} // namespace amdgpu
} // namespace rocjitsu
