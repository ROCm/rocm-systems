// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/completion_tracker.h"

#include "rocjitsu/vm/amdgpu/hsa_clock.h"

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/amd_hsa_queue.h"
RJ_DIAGNOSTIC_POP

#include "util/log.h"

#include <algorithm>
#include <format>
#include <optional>
#include <set>
#include <span>

namespace rocjitsu {
namespace amdgpu {

void CompletionTracker::notify_wg_complete(uint32_t dispatch_id, uint32_t wg_id,
                                           std::vector<AqlQueueRecord> &queues) {
  for (auto &qs : queues) {
    for (auto &entry : qs.entries) {
      if (entry.dispatch_id == dispatch_id) {
        ++entry.completed_wgs;
        util::Logger::vm([&](auto &os) {
          os << std::format("CT: wg_complete d={} wg={} completed={}/{}", dispatch_id, wg_id,
                            entry.completed_wgs, entry.total_wgs);
        });
        // Completion callbacks can run from CU worker threads. The CP drains
        // completions after dispatch fan-out rejoins, keeping cache flushes and
        // signal firing on the CP path.
        return;
      }
    }
  }
}

CompletionDrainResult CompletionTracker::drain_completions(std::vector<AqlQueueRecord> &queues) {
  CompletionDrainResult result;
  for (auto &qs : queues)
    qs.publication_retry_pending = false;

  for (auto &qs : queues) {
    const bool idle_generation_stale =
        qs.idle_publication.active() &&
        (!qs.entries.empty() || qs.idle_publication.activity_generation != qs.activity_generation);
    const bool idle_publication_cancelable =
        qs.idle_publication.phase == QueueIdlePublicationPhase::CaptureAccess ||
        qs.idle_publication.phase == QueueIdlePublicationPhase::ReadSignalHandle ||
        qs.idle_publication.phase == QueueIdlePublicationPhase::StoreIdleStatus;
    if (idle_generation_stale && idle_publication_cancelable) {
      // Before the status CAS commits, new work invalidates the prior empty
      // transition. After it commits, the status is externally observable and
      // its mailbox/interrupt tail must finish; ROCr clears the signal when its
      // handler consumes that notification.
      qs.idle_publication.reset();
    }
    if (qs.idle_publication.active()) {
      const VmAccessOutcome outcome = advance_queue_idle_publication(qs.idle_publication);
      if (outcome == VmAccessOutcome::Unavailable) {
        qs.publication_retry_pending = true;
        result.retry_pending = true;
        continue;
      }
      if (outcome != VmAccessOutcome::Complete) {
        result.terminal_fault = CompletionDrainFault{
            .queue_id = qs.idle_publication.queue_id,
            .process_id = qs.idle_publication.process_id,
            .outcome = outcome,
            .queue_idle = true,
        };
        qs.idle_publication.reset();
        return result;
      }
      qs.idle_publication.reset();
      result.progress_made = true;
    }

    const bool had_entries = !qs.entries.empty();
    uint32_t last_process_id = 0;
    uint32_t last_queue_id = 0;
    AddressSpaceHandle last_address_space;
    bool publication_stalled = false;
    while (!qs.entries.empty() && qs.entries.front().fully_completed()) {
      auto &entry = qs.entries.front();
      last_process_id = entry.process_id;
      last_queue_id = entry.queue_id;
      last_address_space = entry.address_space;

      // A peer may publish the shared terminal-fault latch before this CP has
      // drained its fault inbox. Leave the entry at the head for that path.
      if (entry.grid_faulted())
        break;

      util::Logger::vm([&](auto &os) {
        os << std::format("CT: drain d={} completed={}/{} sig={:#x}", entry.dispatch_id,
                          entry.completed_wgs, entry.total_wgs, entry.completion_signal);
      });

      const VmAccessOutcome outcome = deliver_completion(entry);
      if (outcome == VmAccessOutcome::Unavailable) {
        qs.publication_retry_pending = true;
        result.retry_pending = true;
        publication_stalled = true;
        break;
      }
      if (outcome != VmAccessOutcome::Complete) {
        entry.terminal_faulted = true;
        if (entry.grid_completion)
          entry.grid_completion->mark_faulted();
        result.terminal_fault = CompletionDrainFault{
            .queue_id = entry.queue_id,
            .process_id = entry.process_id,
            .dispatch_id = entry.dispatch_id,
            .outcome = outcome,
        };
        return result;
      }
      if (!entry.completion_notified)
        break;
      if (dispatch_retired_cb_)
        dispatch_retired_cb_(entry);
      if (qs.next_dispatch_idx > 0)
        --qs.next_dispatch_idx;
      qs.entries.pop_front();
      result.progress_made = true;
    }

    if (publication_stalled)
      continue;

    // The queue-inactive notification has its own journal because its dispatch
    // entry no longer exists after retirement.
    if (had_entries && qs.entries.empty() && last_process_id != 0 && !qs.fanout_replica) {
      qs.idle_publication.reset();
      qs.idle_publication.phase = QueueIdlePublicationPhase::CaptureAccess;
      qs.idle_publication.address_space = qs.address_space ? qs.address_space : last_address_space;
      qs.idle_publication.interrupt_sink = qs.interrupt_sink;
      qs.idle_publication.queue_desc_va = qs.queue_desc_va;
      qs.idle_publication.process_id = last_process_id;
      qs.idle_publication.queue_id = last_queue_id;
      qs.idle_publication.activity_generation = qs.activity_generation;
      const VmAccessOutcome outcome = advance_queue_idle_publication(qs.idle_publication);
      if (outcome == VmAccessOutcome::Unavailable) {
        qs.publication_retry_pending = true;
        result.retry_pending = true;
        continue;
      }
      if (outcome != VmAccessOutcome::Complete) {
        result.terminal_fault = CompletionDrainFault{
            .queue_id = last_queue_id,
            .process_id = last_process_id,
            .outcome = outcome,
            .queue_idle = true,
        };
        qs.idle_publication.reset();
        return result;
      }
      qs.idle_publication.reset();
      result.progress_made = true;
    }
  }
  return result;
}

VmAccessOutcome CompletionTracker::complete_non_kernel(DispatchEntry &entry) {
  return entry.is_non_kernel() ? deliver_completion(entry) : VmAccessOutcome::Malformed;
}

VmAccessOutcome CompletionTracker::deliver_completion(DispatchEntry &entry) {
  if (entry.completion_notified)
    return VmAccessOutcome::Complete;

  // Publish this XCD's share only after its own caches are flushed. The release
  // in publish_share() pairs with the acquire in grid_retired(), so the signal
  // cannot fire while another XCD's results remain in its caches.
  if (!entry.grid_share_published) {
    flush_caches(entry.process_id);
    entry.grid_share_published = true;
    if (entry.grid_completion && entry.grid_completion->publish_share(entry.total_wgs) &&
        grid_retired_cb_)
      grid_retired_cb_(entry);
  }

  // A completed local shard remains queued until all peer XCD shares retire.
  if (!entry.grid_fully_completed())
    return VmAccessOutcome::Complete;

  // Only the XCD that read the packet advances guest-visible publication.
  if (!entry.fanout_peer) {
    const VmAccessOutcome outcome = advance_signal_publication(entry);
    if (outcome != VmAccessOutcome::Complete)
      return outcome;
  }
  entry.completion_notified = true;
  return VmAccessOutcome::Complete;
}

VmAccessOutcome
CompletionTracker::advance_queue_idle_publication(QueueIdlePublicationState &state) {
  constexpr uint64_t kQueueInactiveSignalOffset = offsetof(amd_queue_t, queue_inactive_signal);
  constexpr uint32_t kSignalValueOffset = 8;
  constexpr uint32_t kMailboxPointerOffset = 16;
  constexpr uint32_t kEventIdOffset = 24;
  constexpr uint64_t kIdleStatus = 0x10;

  for (;;) {
    switch (state.phase) {
    case QueueIdlePublicationPhase::Inactive:
    case QueueIdlePublicationPhase::Complete:
      return VmAccessOutcome::Complete;

    case QueueIdlePublicationPhase::CaptureAccess:
      if (state.queue_desc_va == 0) {
        state.phase = QueueIdlePublicationPhase::DeliverGenericInterrupt;
        continue;
      }
      if (!state.address_space)
        return VmAccessOutcome::Faulted;
      if (std::optional<GpuVmAccess> access = gpu_vm_.snapshot(state.address_space))
        state.access = std::make_shared<GpuVmAccess>(std::move(*access));
      else
        return VmAccessOutcome::Faulted;
      state.phase = QueueIdlePublicationPhase::ReadSignalHandle;
      continue;

    case QueueIdlePublicationPhase::ReadSignalHandle: {
      const uint64_t address = state.queue_desc_va + kQueueInactiveSignalOffset;
      const VmAccessOutcome outcome =
          read_gpu(*state.access, address, &state.signal_address, sizeof(state.signal_address));
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      if (state.signal_address == 0) {
        state.phase = QueueIdlePublicationPhase::DeliverGenericInterrupt;
        continue;
      }
      if ((state.signal_address & 0x3f) != 0)
        return VmAccessOutcome::Malformed;
      state.phase = QueueIdlePublicationPhase::StoreIdleStatus;
      continue;
    }

    case QueueIdlePublicationPhase::StoreIdleStatus: {
      // Preserve ROCR's destructor sentinel: a completed mismatch still needs
      // the mailbox and event-age notification below.
      const AtomicCompareExchangeResult idle =
          compare_exchange_gpu(*state.access, state.signal_address + kSignalValueOffset,
                               sizeof(uint64_t), 0, kIdleStatus);
      if (idle.outcome != VmAccessOutcome::Complete)
        return idle.outcome;
      state.phase = QueueIdlePublicationPhase::ReadMailboxPointer;
      continue;
    }

    case QueueIdlePublicationPhase::ReadMailboxPointer: {
      const VmAccessOutcome outcome =
          read_gpu(*state.access, state.signal_address + kMailboxPointerOffset,
                   &state.mailbox_pointer, sizeof(state.mailbox_pointer));
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      state.phase = QueueIdlePublicationPhase::ReadEventId;
      continue;
    }

    case QueueIdlePublicationPhase::ReadEventId: {
      const VmAccessOutcome outcome = read_gpu(*state.access, state.signal_address + kEventIdOffset,
                                               &state.event_id, sizeof(state.event_id));
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      state.phase = QueueIdlePublicationPhase::StoreMailbox;
      continue;
    }

    case QueueIdlePublicationPhase::StoreMailbox:
      if (state.mailbox_pointer != 0) {
        const VmAccessOutcome outcome = atomic_store_gpu(
            *state.access, state.mailbox_pointer, sizeof(uint64_t), uint64_t(state.event_id));
        if (outcome != VmAccessOutcome::Complete)
          return outcome;
      }
      state.phase = QueueIdlePublicationPhase::DeliverEventInterrupt;
      continue;

    case QueueIdlePublicationPhase::DeliverEventInterrupt:
      if (state.event_id != 0)
        state.interrupt_sink.deliver(state.process_id, state.event_id);
      state.phase = QueueIdlePublicationPhase::DeliverGenericInterrupt;
      continue;

    case QueueIdlePublicationPhase::DeliverGenericInterrupt:
      state.interrupt_sink.deliver(state.process_id, 0);
      state.phase = QueueIdlePublicationPhase::Complete;
      return VmAccessOutcome::Complete;
    }
  }
}

void CompletionTracker::flush_caches(uint32_t vmid) {
  std::set<L2Cache *> flushed_l2s;
  for (auto *cu : cus_) {
    cu->flush_l1(vmid);
    if (auto *l2 = cu->l2(); l2 && flushed_l2s.insert(l2).second)
      l2->flush_all(vmid);
  }
  for (auto *l2 : l2_caches_) {
    if (l2 && flushed_l2s.insert(l2).second)
      l2->flush_all(vmid);
  }
}

bool CompletionTracker::all_complete(const std::vector<AqlQueueRecord> &queues) const {
  for (const auto &qs : queues) {
    if (!qs.entries.empty() || qs.idle_publication.active())
      return false;
  }
  return true;
}

VmAccessOutcome CompletionTracker::advance_signal_publication(DispatchEntry &entry) {
  constexpr uint32_t kSignalValueOffset = 8;
  constexpr uint32_t kMailboxPointerOffset = 16;
  constexpr uint32_t kEventIdOffset = 24;
  constexpr uint32_t kStartTimestampOffset = 32;
  constexpr uint32_t kEndTimestampOffset = 40;

  CompletionPublicationState &state = entry.completion_publication;
  for (;;) {
    switch (state.phase) {
    case CompletionPublicationPhase::ExecutionEnd:
      plugin_group_->onAmdgpuDispatchExecutionEnd(entry.dispatch_id);
      state.phase = entry.completion_signal == 0 ? CompletionPublicationPhase::Complete
                                                 : CompletionPublicationPhase::CaptureAccess;
      continue;

    case CompletionPublicationPhase::CaptureAccess:
      if (!entry.address_space)
        return VmAccessOutcome::Faulted;
      if (std::optional<GpuVmAccess> access = gpu_vm_.snapshot(entry.address_space))
        state.access = std::make_shared<GpuVmAccess>(std::move(*access));
      else
        return VmAccessOutcome::Faulted;
      state.start_timestamp = entry.profiling_start_timestamp;
      if (state.start_timestamp == 0)
        state.start_timestamp = hsa_system_timestamp();
      state.end_timestamp = std::max(hsa_system_timestamp(), state.start_timestamp + 1);
      state.phase = CompletionPublicationPhase::ReadMailboxPointer;
      continue;

    case CompletionPublicationPhase::ReadMailboxPointer: {
      const VmAccessOutcome outcome =
          read_gpu(*state.access, entry.completion_signal + kMailboxPointerOffset,
                   &state.mailbox_pointer, sizeof(state.mailbox_pointer));
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      state.phase = CompletionPublicationPhase::ReadEventId;
      continue;
    }

    case CompletionPublicationPhase::ReadEventId: {
      const VmAccessOutcome outcome =
          read_gpu(*state.access, entry.completion_signal + kEventIdOffset, &state.event_id,
                   sizeof(state.event_id));
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      state.phase = CompletionPublicationPhase::StoreStartTimestamp;
      continue;
    }

    case CompletionPublicationPhase::StoreStartTimestamp: {
      const VmAccessOutcome outcome =
          atomic_store_gpu(*state.access, entry.completion_signal + kStartTimestampOffset,
                           sizeof(uint64_t), state.start_timestamp);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      state.phase = CompletionPublicationPhase::StoreEndTimestamp;
      continue;
    }

    case CompletionPublicationPhase::StoreEndTimestamp: {
      const VmAccessOutcome outcome =
          atomic_store_gpu(*state.access, entry.completion_signal + kEndTimestampOffset,
                           sizeof(uint64_t), state.end_timestamp);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      state.phase = CompletionPublicationPhase::DecrementSignal;
      continue;
    }

    case CompletionPublicationPhase::DecrementSignal:
      for (;;) {
        const uint64_t desired = state.compare_expected - 1;
        const AtomicCompareExchangeResult decremented =
            compare_exchange_gpu(*state.access, entry.completion_signal + kSignalValueOffset,
                                 sizeof(uint64_t), state.compare_expected, desired);
        if (decremented.outcome != VmAccessOutcome::Complete)
          return decremented.outcome;
        if (decremented.exchanged) {
          state.signal_old_value = state.compare_expected;
          state.signal_new_value = desired;
          state.phase = CompletionPublicationPhase::StoreMailbox;
          break;
        }
        state.compare_expected = decremented.observed;
      }
      util::Logger::cp([&](auto &os) {
        os << std::format("FIRE_SIGNAL_RESULT d={} old_val={} new_val={} mailbox={:#x} event_id={} "
                          "has_interrupt_sink={}",
                          entry.dispatch_id, static_cast<int64_t>(state.signal_old_value),
                          static_cast<int64_t>(state.signal_new_value), state.mailbox_pointer,
                          state.event_id, static_cast<bool>(entry.interrupt_sink));
      });
      continue;

    case CompletionPublicationPhase::StoreMailbox:
      if (state.mailbox_pointer != 0) {
        const VmAccessOutcome outcome = atomic_store_gpu(
            *state.access, state.mailbox_pointer, sizeof(uint64_t), uint64_t(state.event_id));
        if (outcome != VmAccessOutcome::Complete)
          return outcome;
      }
      state.phase = CompletionPublicationPhase::DeliverInterrupt;
      continue;

    case CompletionPublicationPhase::DeliverInterrupt:
      entry.interrupt_sink.deliver(entry.process_id, state.event_id);
      state.phase = CompletionPublicationPhase::Complete;
      return VmAccessOutcome::Complete;

    case CompletionPublicationPhase::Complete:
      return VmAccessOutcome::Complete;
    }
  }
}

VmAccessOutcome CompletionTracker::read_gpu(const GpuVmAccess &access, uint64_t address,
                                            void *destination, size_t size) const {
  return access.read(address, std::span<std::byte>(static_cast<std::byte *>(destination), size));
}

VmAccessOutcome CompletionTracker::atomic_store_gpu(const GpuVmAccess &access, uint64_t address,
                                                    uint32_t width, uint64_t value) {
  return access.atomic_store(address, width, value);
}

AtomicCompareExchangeResult
CompletionTracker::compare_exchange_gpu(const GpuVmAccess &access, uint64_t address, uint32_t width,
                                        uint64_t expected, uint64_t desired) {
  return access.compare_exchange(address, width, expected, desired);
}

} // namespace amdgpu
} // namespace rocjitsu
