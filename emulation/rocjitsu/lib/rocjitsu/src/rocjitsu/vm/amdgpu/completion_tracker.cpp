// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/completion_tracker.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/amd_hsa_queue.h"
RJ_DIAGNOSTIC_POP

#include "util/log.h"

#include <atomic>
#include <cstring>
#include <format>

namespace rocjitsu {
namespace amdgpu {

void CompletionTracker::notify_wg_complete(uint32_t dispatch_id, uint32_t wg_id,
                                           std::vector<HwQueueState> &queues) {
  for (auto &qs : queues) {
    for (auto &entry : qs.entries) {
      if (entry.dispatch_id == dispatch_id) {
        ++entry.completed_wgs;
        util::Logger::vm([&](auto &os) {
          os << std::format("CT: wg_complete d={} wg={} completed={}/{}", dispatch_id, wg_id,
                            entry.completed_wgs, entry.total_wgs);
        });
        drain_completions(queues);
        return;
      }
    }
  }
}

void CompletionTracker::drain_completions(std::vector<HwQueueState> &queues) {
  for (auto &qs : queues) {
    bool had_entries = !qs.entries.empty();
    uint32_t last_process_id = 0;
    while (!qs.entries.empty() && qs.entries.front().fully_completed()) {
      auto &entry = qs.entries.front();
      last_process_id = entry.process_id;

      util::Logger::vm([&](auto &os) {
        os << std::format("CT: drain d={} completed={}/{} sig={:#x}", entry.dispatch_id,
                          entry.completed_wgs, entry.total_wgs, entry.completion_signal);
      });

      flush_caches(entry.process_id);
      plugin_group_->onAmdgpuDispatchExecutionEnd(entry.dispatch_id);
      if (entry.completion_signal != 0) {
        fire_signal(entry);
      }

      if (qs.next_dispatch_idx > 0)
        --qs.next_dispatch_idx;

      qs.entries.pop_front();
    }
    // HQD idle: write the queue's inactive signal and fire the interrupt.
    // On real hardware the CP writes the HQD status to amd_signal_t::value
    // and kfd_signal_event_interrupt broadcasts to all type-0 events.
    if (had_entries && qs.entries.empty() && last_process_id != 0) {
      if (qs.queue_desc_va != 0)
        fire_queue_idle_signal(qs.queue_desc_va, last_process_id);
      if (interrupt_cb_)
        interrupt_cb_(last_process_id, 0);
    }
  }
}

void CompletionTracker::fire_queue_idle_signal(uint64_t queue_desc_va, uint32_t process_id) {
  if (!memory_)
    return;

  constexpr uint64_t kQueueInactiveSigOff = offsetof(amd_queue_t, queue_inactive_signal);
  constexpr uint32_t MAILBOX_PTR_OFF = 16;
  constexpr uint32_t EVENT_ID_OFF = 24;

  uint64_t sig_handle_va = queue_desc_va + kQueueInactiveSigOff;
  auto *desc_page = memory_->resolve_host_ptr(sig_handle_va, process_id);
  if (!desc_page)
    return;

  size_t page_offset = sig_handle_va & 0xFFF;
  if (page_offset + sizeof(uint64_t) > 0x1000)
    return;

  uint64_t sig_addr = 0;
  std::memcpy(&sig_addr, desc_page + page_offset, sizeof(sig_addr));
  if (sig_addr == 0)
    return;

  if ((sig_addr & 0x3F) != 0)
    return;
  auto *sig_page = memory_->resolve_host_ptr(sig_addr, process_id);
  if (!sig_page)
    return;
  auto *sig_base = sig_page + (sig_addr & 0xFFF);

  constexpr uint32_t SIG_VAL_OFF = 8;
  constexpr uint64_t kIdleStatus = 0x10;

  // CAS: write the idle status only if the value is currently 0. This
  // prevents clobbering ROCR's destructor sentinel (0x8000000000000000)
  // which the handler must see to complete the shutdown handshake.
  auto *val_ptr = reinterpret_cast<uint64_t *>(sig_base + SIG_VAL_OFF);
  uint64_t expected = 0;
  std::atomic_ref<uint64_t>(*val_ptr).compare_exchange_strong(
      expected, kIdleStatus, std::memory_order_release, std::memory_order_relaxed);

  // Always fire the mailbox write and event-specific interrupt regardless
  // of whether we wrote the value. During shutdown ROCR's destructor has
  // already stored 0x8000… but the handler may be in the pending list and
  // needs the event age advance to trigger the merge.
  uint32_t event_id = 0;
  std::memcpy(&event_id, sig_base + EVENT_ID_OFF, sizeof(event_id));

  uint64_t mailbox_ptr = 0;
  std::memcpy(&mailbox_ptr, sig_base + MAILBOX_PTR_OFF, sizeof(mailbox_ptr));
  if (mailbox_ptr != 0) {
    auto *mb_page = memory_->resolve_host_ptr(mailbox_ptr, process_id);
    if (mb_page) {
      auto *mb_ptr = reinterpret_cast<uint64_t *>(mb_page + (mailbox_ptr & 0xFFF));
      std::atomic_ref<uint64_t>(*mb_ptr).store(uint64_t(event_id), std::memory_order_release);
    }
  }

  if (interrupt_cb_ && event_id != 0)
    interrupt_cb_(process_id, event_id);
}

void CompletionTracker::flush_caches(uint32_t vmid) {
  for (auto *cu : cus_)
    cu->flush_all(vmid);
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
  util::Logger::cp([&](auto &os) {
    os << std::format("FIRE_SIGNAL d={} sig={:#x} pid={} cus={}", entry.dispatch_id,
                      entry.completion_signal, entry.process_id, cus_.size());
  });
  constexpr uint32_t SIG_VAL_OFF = 8;
  constexpr uint32_t MAILBOX_PTR_OFF = 16;
  constexpr uint32_t EVENT_ID_OFF = 24;

  if (memory_) {
    auto *sig_page =
        memory_->resolve_host_ptr(entry.completion_signal + SIG_VAL_OFF, entry.process_id);

    util::Logger::cp([&](auto &os) {
      auto *page0 = memory_->resolve_host_ptr(entry.completion_signal, entry.process_id);
      uint64_t kind_raw = 0, val_raw = 0, mbx_raw = 0;
      uint32_t eid_raw = 0;
      if (page0) {
        std::memcpy(&kind_raw, page0 + (entry.completion_signal & 0xFFF), 8);
        std::memcpy(&val_raw, page0 + ((entry.completion_signal + SIG_VAL_OFF) & 0xFFF), 8);
        std::memcpy(&mbx_raw, page0 + ((entry.completion_signal + MAILBOX_PTR_OFF) & 0xFFF), 8);
        std::memcpy(&eid_raw, page0 + ((entry.completion_signal + EVENT_ID_OFF) & 0xFFF), 4);
      } else {
        kind_raw = memory_->read64(entry.completion_signal, entry.process_id);
        val_raw = memory_->read64(entry.completion_signal + SIG_VAL_OFF, entry.process_id);
        mbx_raw = memory_->read64(entry.completion_signal + MAILBOX_PTR_OFF, entry.process_id);
        eid_raw = memory_->read32(entry.completion_signal + EVENT_ID_OFF, entry.process_id);
      }
      os << std::format("SIGNAL_DUMP d={} sig={:#x} pid={} host_page={} kind={:#x} val={} "
                        "mailbox={:#x} event_id={}",
                        entry.dispatch_id, entry.completion_signal, entry.process_id,
                        page0 != nullptr, kind_raw, static_cast<int64_t>(val_raw), mbx_raw,
                        eid_raw);
    });

    int64_t old = 0;
    uint64_t new_val = 0;
    if (sig_page) {
      auto *sig_ptr = reinterpret_cast<uint64_t *>(
          sig_page + ((entry.completion_signal + SIG_VAL_OFF) & 0xFFF));
      old =
          static_cast<int64_t>(std::atomic_ref<uint64_t>(*sig_ptr).load(std::memory_order_relaxed));
      new_val = static_cast<uint64_t>(old - 1);
      std::atomic_ref<uint64_t>(*sig_ptr).store(new_val, std::memory_order_release);
    } else {
      old = static_cast<int64_t>(
          memory_->read64(entry.completion_signal + SIG_VAL_OFF, entry.process_id));
      new_val = static_cast<uint64_t>(old - 1);
      memory_->write64(entry.completion_signal + SIG_VAL_OFF, new_val, entry.process_id);
    }

    auto mailbox_ptr = memory_->read64(entry.completion_signal + MAILBOX_PTR_OFF, entry.process_id);
    uint32_t event_id = memory_->read32(entry.completion_signal + EVENT_ID_OFF, entry.process_id);

    util::Logger::cp([&](auto &os) {
      os << std::format("FIRE_SIGNAL_RESULT d={} old_val={} new_val={} mailbox={:#x} event_id={} "
                        "has_interrupt_cb={}",
                        entry.dispatch_id, old, static_cast<int64_t>(new_val), mailbox_ptr,
                        event_id, interrupt_cb_ != nullptr);
    });

    if (mailbox_ptr != 0) {
      auto *mb_page = memory_->resolve_host_ptr(mailbox_ptr, entry.process_id);
      if (mb_page) {
        auto *mb_ptr = reinterpret_cast<uint64_t *>(mb_page + (mailbox_ptr & 0xFFF));
        std::atomic_ref<uint64_t>(*mb_ptr).store(uint64_t(event_id), std::memory_order_release);
      } else {
        memory_->write64(mailbox_ptr, uint64_t(event_id), entry.process_id);
      }
    }

    if (interrupt_cb_)
      interrupt_cb_(entry.process_id, event_id);
  }
}

} // namespace amdgpu
} // namespace rocjitsu
