// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/command_processor.h"

#include "simdojo/sim/message.h"
#include "simdojo/sim/simulation.h"
#include "util/debug_print.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <set>
#include <string>
#include <thread>

namespace rocjitsu {
namespace amdgpu {

void CommandProcessor::init_wavefront_regs(ComputeUnitCore *cu, Wavefront *wf,
                                           const InternalDispatch &pkt, uint32_t global_wg_id,
                                           uint32_t wf_index_in_wg) {
  uint32_t sbase = wf->sgpr_alloc().base;

  // User SGPRs: kernarg pointer in s[0:1] (implicit gfx9+ ABI).
  if (pkt.kernarg_addr != 0) {
    cu->write_sgpr(sbase + 0, static_cast<uint32_t>(pkt.kernarg_addr));
    cu->write_sgpr(sbase + 1, static_cast<uint32_t>(pkt.kernarg_addr >> 32));
  }

  // System SGPR: workgroup_id_x after user SGPRs.
  cu->write_sgpr(sbase + pkt.num_user_sgprs, global_wg_id);

  // Workitem ID: v0 = workitem_id_x within the workgroup.
  // For multi-wavefront workgroups, each wavefront covers a different range:
  // wf0: 0..wf_size-1, wf1: wf_size..2*wf_size-1, etc.
  uint32_t vbase = wf->vgpr_alloc().base;
  uint32_t workitem_base = wf_index_in_wg * cu->wf_size();
  for (uint32_t lane = 0; lane < cu->wf_size(); ++lane)
    cu->write_vgpr(vbase, lane, workitem_base + lane);
}

void CommandProcessor::startup() {
  doorbell_event_.set_handler(
      [this](simdojo::Tick ts, simdojo::Message *) { handle_doorbell(ts); });
}

void CommandProcessor::register_queue(HwQueue queue) {
  bool start_poll = queue.host_accessible;
  {
    std::lock_guard<std::mutex> lock(hw_queue_mutex_);
    hw_queues_.push_back(std::move(queue));
    if (!is_primary_ && engine()) {
      engine()->register_as_primary();
      is_primary_ = true;
    }
  }
  // Only start the doorbell poll thread for KFD (host-accessible) queues.
  // Internal test queues inject doorbell events directly via schedule_event_now().
  // NOTE: the joinable() check is intentionally outside hw_queue_mutex_ — this
  // is safe because ROCR always creates queues from a single thread (the HSA
  // queue-creation path is not re-entrant), so there is no concurrent caller.
  if (start_poll && !doorbell_thread_.joinable()) {
    doorbell_thread_ = std::jthread([this](std::stop_token stop) { doorbell_poll_loop(stop); });
  }
}

void CommandProcessor::unregister_queue(uint32_t queue_id) {
  bool empty = false;
  {
    std::lock_guard<std::mutex> lock(hw_queue_mutex_);
    std::erase_if(hw_queues_, [queue_id](const HwQueue &q) { return q.queue_id == queue_id; });
    empty = hw_queues_.empty();
  }
  if (empty)
    stop_doorbell_monitor();
}

void CommandProcessor::update_queue(uint32_t queue_id, uint64_t ring_base_va, uint32_t ring_size) {
  std::lock_guard<std::mutex> lock(hw_queue_mutex_);
  for (auto &q : hw_queues_) {
    if (q.queue_id == queue_id) {
      q.ring_base_va = ring_base_va;
      q.ring_size = ring_size;
      break;
    }
  }
}

void CommandProcessor::set_doorbell_base(void *base) {
  doorbell_base_.store(base, std::memory_order_release);
}

void CommandProcessor::stop_doorbell_monitor() {
  if (doorbell_thread_.joinable()) {
    doorbell_thread_.request_stop();
    doorbell_thread_.join();
  }
}

uint64_t CommandProcessor::read_gpu_u64(uint64_t va) const {
  uint64_t val = 0;
  auto *dst = reinterpret_cast<uint8_t *>(&val);
  for (uint32_t i = 0; i < sizeof(val); ++i)
    dst[i] = memory_->read8(va + i);
  return val;
}

/// @brief Scan all HW queues for doorbell changes; return true if any changed.
/// Caller must NOT hold hw_queue_mutex_.
bool CommandProcessor::scan_doorbells() {
  bool found = false;
  std::lock_guard<std::mutex> lock(hw_queue_mutex_);
  void *base = doorbell_base_.load(std::memory_order_acquire);
  for (auto &q : hw_queues_) {
    uint64_t val;
    if (q.host_accessible) {
      if (!base)
        continue;
      val = std::atomic_ref<uint64_t>(
                *reinterpret_cast<uint64_t *>(static_cast<char *>(base) + q.doorbell_offset))
                .load(std::memory_order_acquire);
    } else {
      if (q.doorbell_va == 0)
        continue;
      val = read_gpu_u64(q.doorbell_va);
    }
    if (val != q.last_doorbell) {
      q.last_doorbell = val;
      found = true;
    }
  }
  return found;
}

void CommandProcessor::doorbell_poll_loop(std::stop_token stop) {
  using namespace std::chrono_literals;
  while (!stop.stop_requested()) {
    if (scan_doorbells())
      engine()->schedule_event_now(&doorbell_event_);
    else
      std::this_thread::sleep_for(100us);
  }
}

bool CommandProcessor::step() {
  if (dispatched_ >= dispatch_queue_.size())
    return false;

  assert(!cus_.empty() && "command processor has no compute units");

  const InternalDispatch &pkt = dispatch_queue_[dispatched_++];

  // Activate wavefront slots for each workgroup, distributing across CUs.
  // If a CU is full, try all other CUs in round-robin before giving up.
  // If ALL CUs are full, re-enqueue remaining workgroups for retry on next doorbell.
  for (uint32_t wg = 0; wg < pkt.workgroup_count; ++wg) {
    uint32_t global_wg_id = wg + pkt.workgroup_id_offset;
    bool wg_dispatched = true;
    for (uint32_t w = 0; w < pkt.wfs_per_workgroup; ++w) {
      Wavefront *wf = nullptr;
      ComputeUnitCore *chosen_cu = nullptr;

      // Try all CUs in round-robin starting from next_cu_.
      for (size_t attempt = 0; attempt < cus_.size(); ++attempt) {
        size_t cu_idx = (next_cu_ + attempt) % cus_.size();
        ComputeUnitCore *cu = cus_[cu_idx];
        wf = cu->dispatch_wf(wg, pkt.kernel_entry_pc, pkt.sgprs_per_wf, pkt.vgprs_per_wf);
        if (!wf) {
          // CU full or out of registers - retire halted wfs and retry this CU once.
          cu->retire_halted_wfs();
          wf = cu->dispatch_wf(wg, pkt.kernel_entry_pc, pkt.sgprs_per_wf, pkt.vgprs_per_wf);
        }
        if (wf) {
          chosen_cu = cu;
          // Advance round-robin past this CU for the next wavefront.
          next_cu_ = (cu_idx + 1) % cus_.size();
          break;
        }
      }

      if (wf && chosen_cu) {
        init_wavefront_regs(chosen_cu, wf, pkt, global_wg_id, w);
      } else {
        // ALL CUs are full. Re-enqueue remaining workgroups for retry.
        util::debug::print(__func__, ": all CUs full at wg=", wg, " w=", w, " - re-enqueueing ",
                           pkt.workgroup_count - wg, " remaining workgroups");
        InternalDispatch retry_pkt = pkt;
        retry_pkt.workgroup_count = pkt.workgroup_count - wg;
        retry_pkt.workgroup_id_offset = pkt.workgroup_id_offset + wg;
        // Transfer the completion signal to the retry entry so it fires only
        // after the LAST fragment of this dispatch completes. Clear the
        // original to prevent double-decrement.
        dispatch_queue_[dispatched_ - 1].completion_signal = 0;
        retry_queue_.push_back(std::move(retry_pkt));
        wg_dispatched = false;
        break;
      }
    }
    if (!wg_dispatched)
      break;
  }

  return dispatched_ < dispatch_queue_.size();
}

void CommandProcessor::check_all_idle() {
  for (auto *cu : cus_)
    if (!cu->is_idle())
      return;

  // If retry queue has pending workgroups, schedule another doorbell to retry.
  if (!retry_queue_.empty()) {
    schedule_event(&doorbell_event_, engine()->context(partition_id()).current_tick() + 1);
    return;
  }

  // Flush all CU caches to backing memory. Real hardware does this implicitly
  // (L2 writeback on kernel completion). Without this, stores using RW (write-
  // back) mtype remain in L2 and never reach host-mapped GpuMemory pages.
  for (auto *cu : cus_)
    cu->flush_all();

  // Signal completion for all dispatched kernel packets whose wavefronts have
  // finished. In real hardware, the CP microcode does this automatically.
  for (size_t i = 0; i < dispatched_; ++i) {
    auto &dp = dispatch_queue_[i];
    if (dp.completion_signal == 0)
      continue;
    constexpr uint32_t SIG_VAL_OFF = 8;
    auto *val = reinterpret_cast<int64_t *>(dp.completion_signal + SIG_VAL_OFF);
    if (dp.host_signal) {
      std::atomic_ref<int64_t>(*val).fetch_sub(1, std::memory_order_release);
    } else if (memory_) {
      auto old = static_cast<int64_t>(memory_->read64(dp.completion_signal + SIG_VAL_OFF));
      memory_->write64(dp.completion_signal + SIG_VAL_OFF, static_cast<uint64_t>(old - 1));
    }
    dp.completion_signal = 0; // Signal only once.
  }

  // Enforce in-order dispatch for ordered (KFD) queue entries: start the next
  // pending kernel only after all wavefronts from the previous dispatch have
  // completed. This mirrors real hardware where an ordered AQL queue executes
  // dispatches sequentially. Unordered (test) dispatches are not re-dispatched
  // here; they were already all submitted in handle_doorbell.
  //
  // Loop (not recursion) to handle consecutive zero-workgroup dispatches
  // (e.g. back-to-back BARRIER_AND packets) without growing the call stack.
  while (dispatched_ < dispatch_queue_.size() && dispatch_queue_[dispatched_].ordered) {
    size_t prev = dispatched_;
    step();
    if (dispatched_ <= prev)
      break;
    std::set<ComputeUnitCore *> new_cus;
    for (auto *cu : cus_)
      if (cu->has_active_wfs())
        new_cus.insert(cu);
    for (size_t i = 0; i < cus_.size(); ++i) {
      if (new_cus.count(cus_[i]) == 0)
        continue;
      if (dispatch_ports_[i]->link())
        dispatch_ports_[i]->send(std::make_unique<simdojo::Message>(simdojo::MessageHeader{}));
      else
        cus_[i]->activate();
    }
    if (!new_cus.empty())
      return; // CUs have work; they call check_all_idle when done.
    // Zero-workgroup dispatch (e.g. BARRIER_AND): no CUs activated so no
    // on_idle callback will ever fire for this entry. Signal its completion
    // now before looping to the next ordered entry.
    for (size_t i = prev; i < dispatched_; ++i) {
      auto &dp = dispatch_queue_[i];
      if (dp.completion_signal == 0)
        continue;
      constexpr uint32_t SIG_VAL_OFF = 8;
      auto *val = reinterpret_cast<int64_t *>(dp.completion_signal + SIG_VAL_OFF);
      if (dp.host_signal) {
        std::atomic_ref<int64_t>(*val).fetch_sub(1, std::memory_order_release);
        // Write the event mailbox and fire the interrupt so WAIT_EVENTS wakes.
        constexpr uint32_t MAILBOX_PTR_OFF = 16, EVENT_ID_OFF = 24;
        auto mailbox_ptr = *reinterpret_cast<uint64_t *>(dp.completion_signal + MAILBOX_PTR_OFF);
        if (mailbox_ptr != 0) {
          auto event_id = *reinterpret_cast<uint32_t *>(dp.completion_signal + EVENT_ID_OFF);
          std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(mailbox_ptr))
              .store(uint64_t(event_id), std::memory_order_release);
          if (interrupt_cb_)
            interrupt_cb_();
        }
      } else if (memory_) {
        auto old = static_cast<int64_t>(memory_->read64(dp.completion_signal + SIG_VAL_OFF));
        memory_->write64(dp.completion_signal + SIG_VAL_OFF, static_cast<uint64_t>(old - 1));
      }
      dp.completion_signal = 0;
    }
  }

  if (pending_dispatches() == 0 && is_primary_) {
    // Only release primary if no host-accessible (KFD) queues are registered.
    // KFD queues can receive new work at any time; the doorbell poll must stay
    // active to detect future dispatches.
    bool has_kfd_queues = false;
    {
      std::lock_guard<std::mutex> lock(hw_queue_mutex_);
      for (const auto &q : hw_queues_)
        if (q.host_accessible) {
          has_kfd_queues = true;
          break;
        }
    }
    if (!has_kfd_queues) {
      stop_doorbell_monitor();
      engine()->primary_release();
      is_primary_ = false;
    }
  }
}

rocr::llvm::amdhsa::kernel_descriptor_t
CommandProcessor::read_kernel_descriptor(uint64_t kernel_object, bool host_accessible) {
  using namespace rocr::llvm::amdhsa;
  kernel_descriptor_t kd{};
  if (host_accessible) {
    std::memcpy(&kd, reinterpret_cast<const void *>(kernel_object), sizeof(kd));
  } else if (memory_) {
    auto *dst = reinterpret_cast<uint8_t *>(&kd);
    for (uint32_t i = 0; i < sizeof(kd); ++i)
      dst[i] = memory_->read8(kernel_object + i);
  }
  return kd;
}

void CommandProcessor::process_aql_packet(const hsa_kernel_dispatch_packet_t &pkt,
                                          bool host_accessible) {
  using namespace rocr::llvm::amdhsa;
  kernel_descriptor_t kd = read_kernel_descriptor(pkt.kernel_object, host_accessible);
  uint32_t vgpr_gran =
      AMDHSA_BITS_GET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  uint32_t sgpr_gran =
      AMDHSA_BITS_GET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
  uint32_t vgprs = (vgpr_gran + 1) * vgpr_granularity_;
  uint32_t sgprs = (sgpr_gran + 1) * 8;
  uint32_t user_sgprs = AMDHSA_BITS_GET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT);
  uint64_t entry_pc = pkt.kernel_object + static_cast<uint64_t>(kd.kernel_code_entry_byte_offset);

  // For host-accessible (KFD) dispatches, the kernel code and kernarg are in
  // host memory. Register them in GpuMemory so the CU's instruction fetch
  // and SMEM loads can access them.
  if (host_accessible && memory_) {
    // Register the code region (kernel object + some margin for the code body).
    uint64_t code_base = pkt.kernel_object & ~0xFFFULL; // Page-align down.
    constexpr size_t CODE_MAP_SIZE = 1 << 20;           // 1MB should cover any kernel.
    memory_->map_host_pages(code_base, reinterpret_cast<void *>(code_base), CODE_MAP_SIZE);
    // Register the kernarg region.
    uint64_t karg = reinterpret_cast<uint64_t>(pkt.kernarg_address);
    if (karg != 0) {
      uint64_t karg_base = karg & ~0xFFFULL;
      memory_->map_host_pages(karg_base, reinterpret_cast<void *>(karg_base), 4096);
    }
  }

  uint32_t wg_size =
      static_cast<uint32_t>(pkt.workgroup_size_x) * pkt.workgroup_size_y * pkt.workgroup_size_z;
  uint32_t wave_size = cus_.empty() ? 64 : cus_[0]->wf_size();
  uint32_t wfs_per_wg = (wg_size + wave_size - 1) / wave_size;

  uint32_t grid_wgs_x = pkt.workgroup_size_x > 0 ? pkt.grid_size_x / pkt.workgroup_size_x : 1;
  uint32_t grid_wgs_y = pkt.workgroup_size_y > 0 ? pkt.grid_size_y / pkt.workgroup_size_y : 1;
  uint32_t grid_wgs_z = pkt.workgroup_size_z > 0 ? pkt.grid_size_z / pkt.workgroup_size_z : 1;
  uint32_t total_wgs = grid_wgs_x * grid_wgs_y * grid_wgs_z;
  InternalDispatch dp{};
  dp.kernel_entry_pc = entry_pc;
  dp.workgroup_count = total_wgs;
  dp.wfs_per_workgroup = wfs_per_wg;
  dp.sgprs_per_wf = sgprs > 0 ? sgprs : 104;
  dp.vgprs_per_wf = vgprs > 0 ? vgprs : 256;
  dp.kernarg_addr = reinterpret_cast<uint64_t>(pkt.kernarg_address);
  dp.num_user_sgprs = user_sgprs;
  dp.workgroup_id_offset = workgroup_id_offset_;
  dp.completion_signal = pkt.completion_signal.handle;
  dp.host_signal = host_accessible;
  dp.ordered = host_accessible;

  dispatch_queue_.push_back(std::move(dp));
}

void CommandProcessor::fetch_from_queue(HwQueue &queue) {
  if (!memory_)
    return;
  // For KFD queues, skip until the doorbell aperture base has been set.
  // For internal test queues, skip until a doorbell GPU VA has been assigned.
  if (queue.host_accessible ? (doorbell_base_.load(std::memory_order_acquire) == nullptr)
                            : (queue.doorbell_va == 0))
    return;

  // Read write and read indices. For KFD queues, pointers are in host memory
  // and can be read directly. For internal test queues, they're in GpuMemory.
  uint64_t write_idx, read_idx;
  if (queue.host_accessible) {
    write_idx = std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(queue.write_ptr_va))
                    .load(std::memory_order_acquire);
    read_idx = std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(queue.read_ptr_va))
                   .load(std::memory_order_acquire);
  } else {
    write_idx = read_gpu_u64(queue.write_ptr_va);
    read_idx = read_gpu_u64(queue.read_ptr_va);
  }

  if (queue.host_accessible)
    if (read_idx >= write_idx)
      return;

  constexpr uint32_t AQL_PACKET_SIZE = 64;
  uint32_t num_slots = queue.ring_size / AQL_PACKET_SIZE;

  while (read_idx < write_idx) {
    uint32_t slot = static_cast<uint32_t>(read_idx % num_slots);
    uint64_t pkt_addr = queue.ring_base_va + slot * AQL_PACKET_SIZE;

    hsa_kernel_dispatch_packet_t pkt{};
    if (queue.host_accessible) {
      // For KFD queues, read directly from host memory (bypass GpuMemory).
      std::memcpy(&pkt, reinterpret_cast<const void *>(pkt_addr), AQL_PACKET_SIZE);
    } else {
      auto *dst = reinterpret_cast<uint8_t *>(&pkt);
      for (uint32_t i = 0; i < AQL_PACKET_SIZE; ++i)
        dst[i] = memory_->read8(pkt_addr + i);
    }

    uint8_t pkt_type = pkt.header & 0xFF;
    if (pkt_type == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
      process_aql_packet(pkt, queue.host_accessible);
    } else if (pkt_type == HSA_PACKET_TYPE_BARRIER_AND || pkt_type == HSA_PACKET_TYPE_BARRIER_OR) {
      // Barriers must wait for all preceding packets to complete before
      // signaling. Enqueue a zero-workgroup dispatch entry so check_all_idle()
      // fires the completion signal only after all CUs are idle (i.e., after
      // all preceding kernel dispatches have finished).
      constexpr uint32_t SIG_OFF = 56;
      uint64_t sig = 0;
      if (queue.host_accessible)
        std::memcpy(&sig, reinterpret_cast<const void *>(pkt_addr + SIG_OFF), sizeof(sig));
      else
        sig = read_gpu_u64(pkt_addr + SIG_OFF);
      if (sig != 0) {
        InternalDispatch dp{};
        dp.workgroup_count = 0;
        dp.completion_signal = sig;
        dp.host_signal = queue.host_accessible;
        dp.ordered = queue.host_accessible;
        dispatch_queue_.push_back(std::move(dp));
      }
    } else if (pkt_type == HSA_PACKET_TYPE_VENDOR_SPECIFIC) {
      // In functional mode, vendor-specific packets (PM4 cache invalidation)
      // have no data dependencies and complete immediately.
      if (queue.host_accessible) {
        constexpr uint32_t SIG_OFF = 56, SIG_VAL_OFF = 8;
        uint64_t sig = 0;
        std::memcpy(&sig, reinterpret_cast<const void *>(pkt_addr + SIG_OFF), sizeof(sig));
        if (sig != 0) {
          auto *val = reinterpret_cast<int64_t *>(sig + SIG_VAL_OFF);
          std::atomic_ref<int64_t>(*val).fetch_sub(1, std::memory_order_release);
        }
      } else {
        signal_aql_completion(pkt_addr);
      }
    }

    ++read_idx;
  }

  // Write updated read pointer back.
  if (queue.host_accessible) {
    std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(queue.read_ptr_va))
        .store(read_idx, std::memory_order_release);
  } else {
    auto *src = reinterpret_cast<const uint8_t *>(&read_idx);
    for (uint32_t i = 0; i < sizeof(read_idx); ++i)
      memory_->write8(queue.read_ptr_va + i, src[i]);
  }

  // NOTE: Do NOT update queue.last_doorbell here. The doorbell poll thread
  // manages last_doorbell and compares it against the actual doorbell value
  // (written by ROCR via StoreRelease on the doorbell signal). The doorbell
  // value is write_index + num_packet - 1, which is one LESS than write_idx.
  // If we set last_doorbell = write_idx here, the poll thread will miss the
  // next doorbell write because the next doorbell value equals this write_idx
  // (for single-packet submissions), causing the second submission to hang.
}

void CommandProcessor::fetch_packets() {
  // Hold hw_queue_mutex_ to prevent register_queue / unregister_queue (called
  // from the ROCR app thread) from mutating hw_queues_ while we iterate it.
  std::lock_guard<std::mutex> lock(hw_queue_mutex_);
  for (auto &queue : hw_queues_)
    fetch_from_queue(queue);
}

void CommandProcessor::signal_aql_completion(uint64_t pkt_addr) {
  // The completion signal handle is at offset 56 in all 64-byte AQL packets.
  // The handle is a host pointer to an amd_signal_t:
  //   offset  8: int64_t  value            (decremented by CP)
  //   offset 16: uint64_t event_mailbox_ptr (CP writes event_id here)
  //   offset 24: uint32_t event_id
  // Per the HSA spec, the CP atomically decrements the signal value. After
  // decrement it simulates the KFD interrupt handler: if event_mailbox_ptr is
  // non-zero it writes event_id to the mailbox slot (the shared event page),
  // then fires interrupt_cb_ to wake any thread in WAIT_EVENTS.
  constexpr uint32_t SIGNAL_OFFSET = 56;
  constexpr uint32_t SIGNAL_VALUE_OFFSET = 8;
  constexpr uint32_t MAILBOX_PTR_OFFSET = 16;
  constexpr uint32_t EVENT_ID_OFFSET = 24;

  uint64_t signal_handle = read_gpu_u64(pkt_addr + SIGNAL_OFFSET);
  if (signal_handle == 0)
    return;

  // Decrement signal value (release so prior GPU stores are visible to waiters).
  auto *val = reinterpret_cast<int64_t *>(signal_handle + SIGNAL_VALUE_OFFSET);
  std::atomic_ref<int64_t>(*val).fetch_sub(1, std::memory_order_release);

  // Simulate interrupt: write event_id to the event mailbox slot so WAIT_EVENTS
  // and libhsakmt's direct signal poll both wake up promptly.
  auto mailbox_ptr = *reinterpret_cast<uint64_t *>(signal_handle + MAILBOX_PTR_OFFSET);
  if (mailbox_ptr != 0) {
    auto event_id = *reinterpret_cast<uint32_t *>(signal_handle + EVENT_ID_OFFSET);
    std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(mailbox_ptr))
        .store(uint64_t(event_id), std::memory_order_release);
    if (interrupt_cb_)
      interrupt_cb_();
  }
}

void CommandProcessor::handle_doorbell(simdojo::Tick) {
  // Fetch AQL packets from registered hardware queues.
  fetch_packets();

  // Re-enqueue any retry packets from previous doorbell cycles.
  for (auto &rpkt : retry_queue_)
    dispatch_queue_.push_back(std::move(rpkt));
  retry_queue_.clear();

  // For ordered (KFD/host-accessible) dispatches, enforce in-order execution:
  // dispatch N+1 must not start until dispatch N completes. If CUs are still
  // running, skip and let check_all_idle() start the next dispatch when idle.
  // For unordered (internal test) dispatches, dispatch all pending in parallel
  // so tests like RoundRobinScheduling can put multiple wavefronts on the CU.
  size_t prev_dispatched = dispatched_;
  {
    bool next_is_ordered =
        (dispatched_ < dispatch_queue_.size() && dispatch_queue_[dispatched_].ordered);
    if (next_is_ordered) {
      bool any_cu_active = false;
      for (auto *cu : cus_)
        if (!cu->is_idle()) {
          any_cu_active = true;
          break;
        }
      if (!any_cu_active)
        step();
    } else {
      while (step()) {
      }
    }
  }

  // Register as primary on first real dispatch so the engine stays alive.
  if (!is_primary_ && dispatched_ > prev_dispatched) {
    engine()->register_as_primary();
    is_primary_ = true;
  }

  // Collect CUs that have active wavefronts (received work from dispatch).
  std::set<ComputeUnitCore *> activated_cus;
  if (dispatched_ > prev_dispatched) {
    for (auto *cu : cus_) {
      if (cu->has_active_wfs())
        activated_cus.insert(cu);
    }
  }

  // Activate CUs that have work. Send through dispatch ports so the link's
  // exec_mode governs delivery: FUNCTIONAL = synchronous direct call,
  // CLOCKED = event-based with propagation latency.
  for (size_t i = 0; i < cus_.size(); ++i) {
    if (activated_cus.count(cus_[i]) == 0)
      continue;
    if (dispatch_ports_[i]->link())
      dispatch_ports_[i]->send(std::make_unique<simdojo::Message>(simdojo::MessageHeader{}));
    else
      cus_[i]->activate(); // Fallback if port not yet wired.
  }

  // If retry queue has pending workgroups, schedule another doorbell.
  if (!retry_queue_.empty())
    schedule_event(&doorbell_event_, engine()->context(partition_id()).current_tick() + 1);
  // If no CUs were activated (barrier-only entries or no new packets),
  // check idle immediately so barrier completion signals fire without
  // waiting for a CU on_idle callback that will never come.
  else if (activated_cus.empty())
    check_all_idle();
}

} // namespace amdgpu
} // namespace rocjitsu
