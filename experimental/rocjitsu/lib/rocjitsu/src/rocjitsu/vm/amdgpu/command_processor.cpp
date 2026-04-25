// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/command_processor.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/amd_hsa_queue.h"
RJ_DIAGNOSTIC_POP

#include "simdojo/sim/message.h"
#include "simdojo/sim/simulation.h"
#include "util/log.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <elf.h>
#include <set>
#include <string>
#include <sys/mman.h>
#include <thread>

namespace rocjitsu {
namespace amdgpu {

void CommandProcessor::init_wavefront_regs(ComputeUnitCore *cu, Wavefront *wf,
                                           const InternalDispatch &pkt, uint32_t global_wg_id,
                                           uint32_t wf_index_in_wg) {
  using namespace rocr::llvm::amdhsa;
  uint32_t sbase = wf->sgpr_alloc().base;
  uint32_t kcp = pkt.kernel_code_properties;

  // User SGPRs per AMDHSA ABI: placed sequentially based on enable bits.
  // Order: private_segment_buffer(4), dispatch_ptr(2), queue_ptr(2),
  //        kernarg_segment_ptr(2), dispatch_id(2), flat_scratch_init(2),
  //        private_segment_size(1).
  // When kernel_code_properties is 0 (internal test dispatches), fall back to
  // the legacy layout: kernarg at s[0:1].
  if (kcp != 0) {
    uint32_t idx = 0;
    if (AMDHSA_BITS_GET(kcp, KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER))
      idx += 4;
    if (AMDHSA_BITS_GET(kcp, KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR)) {
      cu->write_sgpr(sbase + idx, static_cast<uint32_t>(pkt.dispatch_ptr));
      cu->write_sgpr(sbase + idx + 1, static_cast<uint32_t>(pkt.dispatch_ptr >> 32));
      idx += 2;
    }
    if (AMDHSA_BITS_GET(kcp, KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR)) {
      cu->write_sgpr(sbase + idx, static_cast<uint32_t>(pkt.queue_ptr));
      cu->write_sgpr(sbase + idx + 1, static_cast<uint32_t>(pkt.queue_ptr >> 32));
      idx += 2;
    }
    if (AMDHSA_BITS_GET(kcp, KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR)) {
      cu->write_sgpr(sbase + idx, static_cast<uint32_t>(pkt.kernarg_addr));
      cu->write_sgpr(sbase + idx + 1, static_cast<uint32_t>(pkt.kernarg_addr >> 32));
      util::Logger::vm("CP: init_wf kernarg s[", idx, ":", idx + 1, "] = 0x", std::hex,
                       pkt.kernarg_addr, std::dec, " sbase=", sbase);
      idx += 2;
    }
    if (AMDHSA_BITS_GET(kcp, KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID)) {
      cu->write_sgpr(sbase + idx, pkt.workgroup_id_offset);
      cu->write_sgpr(sbase + idx + 1, 0);
      idx += 2;
    }
    if (AMDHSA_BITS_GET(kcp, KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT))
      idx += 2;
    if (AMDHSA_BITS_GET(kcp, KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE)) {
      cu->write_sgpr(sbase + idx, pkt.private_segment_fixed_size);
      idx += 1;
    }
  } else {
    // Legacy: kernarg at s[0:1].
    if (pkt.kernarg_addr != 0) {
      cu->write_sgpr(sbase + 0, static_cast<uint32_t>(pkt.kernarg_addr));
      cu->write_sgpr(sbase + 1, static_cast<uint32_t>(pkt.kernarg_addr >> 32));
    }
  }

  // System SGPR: workgroup_id_x after user SGPRs.
  cu->write_sgpr(sbase + pkt.num_user_sgprs, global_wg_id);

  util::Logger::vm([&](auto &os) {
    static thread_local uint64_t init_count = 0;
    if (++init_count <= 5 || (init_count % 40) == 0)
      os << std::format("CP: init_wf #{} cu={} wf={} global_wg={} s[{}]={} kernarg={:#x}",
                        init_count, cu->name(), wf->wf_id(), global_wg_id,
                        sbase + pkt.num_user_sgprs, global_wg_id, pkt.kernarg_addr);
  });

  // Workitem ID: v0 = workitem_id_x within the workgroup.
  // For multi-wavefront workgroups, each wavefront covers a different range:
  // wf0: 0..wf_size-1, wf1: wf_size..2*wf_size-1, etc.
  uint32_t vbase = wf->vgpr_alloc().base;
  uint32_t workitem_base = wf_index_in_wg * cu->wf_size();
  for (uint32_t lane = 0; lane < cu->wf_size(); ++lane)
    cu->write_vgpr(vbase, lane, workitem_base + lane);

  // Scratch (private segment) setup.
  // Each wavefront gets a unique slice of scratch memory. The per-lane
  // private size is private_segment_fixed_size; the per-wave region is
  // that multiplied by wf_size. For KFD dispatches, scratch_backing_addr
  // comes from amd_queue_t.scratch_backing_memory_location (set by the
  // runtime via KFD). For internal test queues, we use a fallback base.
  if (pkt.private_segment_fixed_size > 0) {
    uint64_t scratch_pool = pkt.scratch_backing_addr;
    if (scratch_pool == 0)
      scratch_pool = 0x1'0000'0000ULL; // Fallback for internal test queues.
    uint64_t per_wave_size = static_cast<uint64_t>(pkt.private_segment_fixed_size) * cu->wf_size();
    // Unique wave index within the dispatch for non-overlapping scratch.
    uint64_t wave_idx =
        static_cast<uint64_t>(global_wg_id) * pkt.wfs_per_workgroup + wf_index_in_wg;
    uint64_t wave_scratch = scratch_pool + wave_idx * per_wave_size;
    wf->set_scratch_base(wave_scratch);

    // Initialize FLAT_SCRATCH_LO/HI (architectural SGPRs s102/s103) so that
    // instructions that read the FLAT_SCRATCH register pair get the base.
    // On GFX9+ the hardware initialises these via the SPI; we do the same.
    // The physical register index is sbase + architectural_index.
    cu->write_sgpr(sbase + 102, static_cast<uint32_t>(wave_scratch));
    cu->write_sgpr(sbase + 103, static_cast<uint32_t>(wave_scratch >> 32));
  }
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
    // SDMA queues: standard doorbell change detection is sufficient.
    // The doorbell value changes from the sentinel (~0) to the write pointer
    // on first submission, triggering process_sdma_ring.
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
  util::Logger::vm([&](auto &os) {
    static bool first = true;
    if (first) {
      first = false;
      os << std::format("CP: {} CUs registered:", cus_.size());
      for (size_t i = 0; i < cus_.size(); ++i) {
        os << std::format("\n[rj log VM]   cu[{}] = {} wf_slots={}", i, cus_[i]->name(),
                          cus_[i]->num_wf_slots());
      }
    }
  });

  const InternalDispatch &pkt = dispatch_queue_[dispatched_++];

  // Activate wavefront slots for each workgroup, distributing across CUs.
  // Entire workgroups must land on a single CU (programming model requirement:
  // LDS is per-CU and s_barrier synchronises within a CU). Query each CU for
  // capacity before dispatching to guarantee all-or-nothing placement.
  for (uint32_t wg = 0; wg < pkt.workgroup_count; ++wg) {
    uint32_t global_wg_id = wg + pkt.workgroup_id_offset;
    bool wg_dispatched = false;

    for (size_t attempt = 0; attempt < cus_.size() && !wg_dispatched; ++attempt) {
      size_t cu_idx = (next_cu_ + attempt) % cus_.size();
      ComputeUnitCore *cu = cus_[cu_idx];
      cu->retire_halted_wfs();
      if (!cu->can_accept_workgroup(pkt.wfs_per_workgroup))
        continue;
      std::vector<Wavefront *> wg_wavefronts;
      wg_wavefronts.reserve(pkt.wfs_per_workgroup);
      for (uint32_t w = 0; w < pkt.wfs_per_workgroup; ++w) {
        Wavefront *wf =
            cu->dispatch_wf(wg, pkt.kernel_entry_pc, pkt.sgprs_per_wf, pkt.vgprs_per_wf);
        assert(wf && "dispatch_wf failed after can_accept_workgroup returned true");
        init_wavefront_regs(cu, wf, pkt, global_wg_id, w);
        wg_wavefronts.push_back(wf);
      }
      plugin_group_->onAmdgpuDispatchWorkgroup(
            global_wg_id, pkt.vgprs_per_wf, pkt.sgprs_per_wf,
            std::span<Wavefront *>(wg_wavefronts));
      next_cu_ = (cu_idx + 1) % cus_.size();
      wg_dispatched = true;
    }

    if (!wg_dispatched) {
      util::Logger::vm("CP step: all CUs full at wg=", wg, " global_wg=", global_wg_id,
                       " total=", pkt.workgroup_count, " offset=", pkt.workgroup_id_offset);
      InternalDispatch retry_pkt = pkt;
      retry_pkt.workgroup_count = pkt.workgroup_count - wg;
      retry_pkt.workgroup_id_offset = pkt.workgroup_id_offset + wg;
      dispatch_queue_[dispatched_ - 1].completion_signal = 0;
      retry_queue_.push_back(std::move(retry_pkt));
      break;
    }
  }

  return dispatched_ < dispatch_queue_.size();
}

void CommandProcessor::check_all_idle() {
  for (auto *cu : cus_)
    if (!cu->is_idle())
      return;

  // If retry queue has pending workgroups, move them to the dispatch queue and
  // process immediately. This avoids scheduling a future event which requires the
  // engine event loop to advance — in direct-activate mode (functional, quantum=0)
  // the engine may not process future events between doorbell callbacks.
  if (!retry_queue_.empty()) {
    util::Logger::vm("CP: retry ", retry_queue_.size(),
                     " entries, total_wgs=", retry_queue_[0].workgroup_count,
                     " offset=", retry_queue_[0].workgroup_id_offset);
    for (auto &rpkt : retry_queue_)
      dispatch_queue_.push_back(std::move(rpkt));
    retry_queue_.clear();
    // Dispatch and activate CUs for the retry workgroups.
    while (dispatched_ < dispatch_queue_.size()) {
      step();
      uint32_t activated = 0;
      for (auto *cu : cus_) {
        if (cu->has_active_wfs()) {
          cu->activate();
          ++activated;
        }
      }
      // Wait for CUs to finish (in direct mode, activate() calls advance() synchronously).
      bool any_active = false;
      for (auto *cu : cus_)
        if (!cu->is_idle()) {
          any_active = true;
          break;
        }
      util::Logger::vm("CP: retry inner: activated=", activated, " any_active=", any_active,
                       " retry_q=", retry_queue_.size());
      if (!any_active)
        continue; // All idle, try next dispatch.
      break;      // CUs still running (shouldn't happen in sync mode).
    }
    // Recurse to handle further retries or signal completion.
    check_all_idle();
    return;
  }

  // Flush all CU caches to backing memory. Real hardware does this implicitly
  // (L2 writeback on kernel completion). Without this, stores using RW (write-
  // back) mtype remain in L2 and never reach host-mapped GpuMemory pages.
  for (auto *cu : cus_)
    cu->flush_all();

  util::Logger::vm([&](auto &os) {
    if (dispatched_ > 0 && memory_) {
      auto &dp0 = dispatch_queue_[0];
      if (dp0.kernarg_addr != 0) {
        uint64_t ka0 = memory_->read64(dp0.kernarg_addr);
        uint64_t ka1 = memory_->read64(dp0.kernarg_addr + 8);
        os << std::format("CP: post-flush kernarg[0]={:#x} kernarg[1]={:#x}", ka0, ka1);
      }
    }
  });

  // Signal completion for all dispatched kernel packets whose wavefronts have
  // finished. In real hardware, the CP microcode does this automatically.
  for (size_t i = 0; i < dispatched_; ++i) {
    auto &dp = dispatch_queue_[i];
    if (dp.completion_signal == 0)
      continue;
    constexpr uint32_t SIG_VAL_OFF = 8;
    constexpr uint32_t MAILBOX_PTR_OFF = 16;
    constexpr uint32_t EVENT_ID_OFF = 24;
    if (dp.host_signal) {
      auto *val = reinterpret_cast<int64_t *>(dp.completion_signal + SIG_VAL_OFF);
      auto old_val = std::atomic_ref<int64_t>(*val).fetch_sub(1, std::memory_order_release);
      util::Logger::vm("CP: signal 0x", std::hex, dp.completion_signal, std::dec, " val ", old_val,
                       " -> ", old_val - 1);
      // Write event mailbox and fire interrupt so ROCR's signal wait wakes up.
      auto mailbox_ptr = *reinterpret_cast<uint64_t *>(dp.completion_signal + MAILBOX_PTR_OFF);
      util::Logger::vm("CP: signal mailbox_ptr=0x", std::hex, mailbox_ptr, std::dec,
                       " has_interrupt_cb=", (interrupt_cb_ ? 1 : 0));
      if (mailbox_ptr != 0) {
        auto event_id = *reinterpret_cast<uint32_t *>(dp.completion_signal + EVENT_ID_OFF);
        util::Logger::vm("CP: writing event_id=", event_id, " to mailbox 0x", std::hex, mailbox_ptr,
                         std::dec);
        std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(mailbox_ptr))
            .store(uint64_t(event_id), std::memory_order_release);
        if (interrupt_cb_)
          interrupt_cb_(event_id);
      }
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
            interrupt_cb_(event_id);
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

/// @brief Find the kernel symbol name from the code object containing kernel_object.
/// @details Scans backward from kernel_object to find the ELF header, then
/// parses .symtab to find the STT_FUNC symbol whose value matches the kernel
/// descriptor offset within the code object.
/// @brief Find the kernel symbol name from the code object containing kernel_object.
/// @details Uses /proc/self/maps to verify readability, then scans backward
/// from kernel_object to find the ELF header and parses the symbol table.
static std::string find_kernel_symbol(uint64_t kernel_object, bool host_accessible) {
  if (kernel_object == 0 || !host_accessible)
    return {};

  auto *ko = reinterpret_cast<const uint8_t *>(kernel_object);

  // The kernel descriptor is inside a code object ELF loaded by ROCR.
  // The ELF starts at the page-aligned base of the allocation.
  // kernel_code_entry_byte_offset is typically 0x100, so the descriptor
  // is 0x100 bytes into the ELF — the ELF header is at the page base.
  auto *elf_base = reinterpret_cast<const uint8_t *>(kernel_object & ~0xFFFULL);
  if (elf_base[0] != 0x7f || elf_base[1] != 'E' || elf_base[2] != 'L' || elf_base[3] != 'F')
    return {};

  auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(elf_base);
  if (ehdr->e_shentsize < sizeof(Elf64_Shdr) || ehdr->e_shnum == 0 || ehdr->e_shoff == 0)
    return {};

  auto *shdrs = reinterpret_cast<const Elf64_Shdr *>(elf_base + ehdr->e_shoff);
  uint64_t kd_offset = static_cast<uint64_t>(ko - elf_base);

  for (uint32_t sh_type : {SHT_SYMTAB, SHT_DYNSYM}) {
    for (uint16_t i = 0; i < ehdr->e_shnum; ++i) {
      if (shdrs[i].sh_type != sh_type || shdrs[i].sh_link >= ehdr->e_shnum)
        continue;
      auto *strtab = reinterpret_cast<const char *>(elf_base + shdrs[shdrs[i].sh_link].sh_offset);
      auto *syms = reinterpret_cast<const Elf64_Sym *>(elf_base + shdrs[i].sh_offset);
      size_t nsyms = shdrs[i].sh_size / sizeof(Elf64_Sym);
      for (size_t s = 0; s < nsyms; ++s) {
        if (syms[s].st_value != kd_offset)
          continue;
        std::string_view name(strtab + syms[s].st_name);
        if (name.ends_with(".kd"))
          name.remove_suffix(3);
        if (!name.empty())
          return std::string(name);
      }
    }
  }
  return {};
}

void CommandProcessor::process_aql_packet(const hsa_kernel_dispatch_packet_t &pkt,
                                          const HwQueue &queue, uint64_t pkt_addr) {
  bool host_accessible = queue.host_accessible;
  using namespace rocr::llvm::amdhsa;
  kernel_descriptor_t kd = read_kernel_descriptor(pkt.kernel_object, host_accessible);
  uint32_t vgpr_gran =
      AMDHSA_BITS_GET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  uint32_t sgpr_gran =
      AMDHSA_BITS_GET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
  uint32_t vgprs = (vgpr_gran + 1) * vgpr_granularity_;
  uint32_t sgprs = (sgpr_gran + 1) * 8;
  util::Logger::vm([&](auto &os) {
    os << std::format("CP: rsrc1={:#x} vgpr_gran={} vgprs={} sgpr_gran={} sgprs={} gran={}",
                      kd.compute_pgm_rsrc1, vgpr_gran, vgprs, sgpr_gran, sgprs, vgpr_granularity_);
  });
  uint32_t user_sgprs = AMDHSA_BITS_GET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT);
  uint64_t entry_pc = pkt.kernel_object + static_cast<uint64_t>(kd.kernel_code_entry_byte_offset);

  // For host-accessible (KFD) dispatches, the kernel code and kernarg are in
  // host memory. Register them in GpuMemory so the CU's instruction fetch
  // and SMEM loads can access them.
  if (host_accessible && memory_) {
    // Register the code region (kernel object + margin for the code body).
    // Use the VRAM allocation size if available; fallback to 2MB for large code objects.
    uint64_t code_base = pkt.kernel_object & ~0xFFFULL; // Page-align down.
    constexpr size_t CODE_MAP_SIZE = 2 << 20;           // 2MB to cover large code objects.
    memory_->map_host_pages(code_base, reinterpret_cast<void *>(code_base), CODE_MAP_SIZE);

    util::Logger::vm([&](auto &os) {
      auto *host_ptr = reinterpret_cast<const uint32_t *>(entry_pc);
      uint32_t gpu_w0 = memory_->fetch32(entry_pc);
      os << std::format("CP: entry_pc code: host=[{:#x},{:#x},{:#x},{:#x}] gpu=[{:#x}]",
                        host_ptr[0], host_ptr[1], host_ptr[2], host_ptr[3], gpu_w0);
      auto *ko_ptr = reinterpret_cast<const uint32_t *>(pkt.kernel_object);
      os << std::format("\n[rj log VM] CP: kernel_obj data: [{:#x},{:#x},{:#x},{:#x}] code_off={}",
                        ko_ptr[0], ko_ptr[1], ko_ptr[2], ko_ptr[3],
                        kd.kernel_code_entry_byte_offset);
      auto karg_ptr = reinterpret_cast<const uint64_t *>(pkt.kernarg_address);
      if (karg_ptr) {
        os << std::format("\n[rj log VM] CP: kernarg q[0:5]={:#x} {:#x} {:#x} {:#x} {:#x} {:#x}",
                          karg_ptr[0], karg_ptr[1], karg_ptr[2], karg_ptr[3], karg_ptr[4],
                          karg_ptr[5]);
        auto karg_dw = reinterpret_cast<const uint32_t *>(pkt.kernarg_address);
        os << std::format("\n[rj log VM] CP: kernarg dw[16:20]={} {} {} {} {}", karg_dw[16],
                          karg_dw[17], karg_dw[18], karg_dw[19], karg_dw[20]);
      }
    });
    // Register the kernarg region. Map enough pages to cover the full kernarg
    // buffer. PyTorch reduction kernels can have kernarg buffers exceeding 4KB
    // (TensorIterator packs many pointers, strides, and flags).
    uint64_t karg = reinterpret_cast<uint64_t>(pkt.kernarg_address);
    if (karg != 0) {
      uint64_t karg_base = karg & ~0xFFFULL;
      constexpr size_t KARG_MAP_SIZE = 8 * 4096; // 32KB covers large kernarg buffers.
      memory_->map_host_pages(karg_base, reinterpret_cast<void *>(karg_base), KARG_MAP_SIZE);
      // Verify: read back from GpuMemory and compare to host.
      util::Logger::vm([&](auto &os) {
        uint32_t host_val = *reinterpret_cast<const uint32_t *>(karg + 0x50);
        uint32_t gpu_val = 0;
        for (int b = 0; b < 4; ++b)
          reinterpret_cast<uint8_t *>(&gpu_val)[b] = memory_->read8(karg + 0x50 + b);
        os << std::format("CP: kernarg verify @0x50: host={:#x} gpu={:#x} {}", host_val, gpu_val,
                          host_val == gpu_val ? "MATCH" : "MISMATCH!");
        uint32_t host14 = *reinterpret_cast<const uint32_t *>(karg + 0x14);
        uint32_t gpu14 = 0;
        for (int b = 0; b < 4; ++b)
          reinterpret_cast<uint8_t *>(&gpu14)[b] = memory_->read8(karg + 0x14 + b);
        os << std::format("\n[rj log VM] CP: kernarg verify @0x14: host={:#x} gpu={:#x} {}", host14,
                          gpu14, host14 == gpu14 ? "MATCH" : "MISMATCH!");
      });
    }
  }

  uint32_t wg_size =
      static_cast<uint32_t>(pkt.workgroup_size_x) * pkt.workgroup_size_y * pkt.workgroup_size_z;
  uint32_t wave_size = cus_.empty() ? 64 : cus_[0]->wf_size();
  uint32_t wfs_per_wg = (wg_size + wave_size - 1) / wave_size;

  // Fast-path: detect ROCR multi-WF blit copy kernels and execute them as memcpy.
  // The ROCR CopyAligned blit with wfs_per_wg > 1 and kcp == 0x8 (only kernarg_ptr
  // enabled) copies data from src to dst using a shader loop. For large copies
  // (e.g., 27MB Tensile code object), this takes minutes in simulation. Detect the
  // pattern and do a direct memcpy instead.
  // Kernarg layout (from trace): dw[0:1]=src, dw[2:3]=dst, dw[8]=size_bytes.
  // Fast-path for ROCR blit copy kernels: detect multi-WF copy dispatches with
  // large grids and execute as memcpy. Only fires when both src and dst are
  // host-mapped, and the data pattern matches a CopyAligned blit.
  // Fast-path for ALL large blit copies (grid >= 4096), including single-WF
  // CopyAligned blits. These are host-to-GPU or GPU-to-GPU copies dispatched
  // by ROCR to load code objects and move tensor data.
  // Fast-path for blit copy kernels with kcp=0x8 (only kernarg_ptr enabled).
  // Covers both CopyAligned (single-WF, grid=16384) and multi-phase copy
  // (multi-WF, grid=256+). Excludes Tensile GEMM kernels which have kcp
  // with additional enable bits or different user_sgpr counts.
  // Fast-path for blit copy kernels. Matches kcp=0x8 (only kernarg_ptr) with
  // vgprs <= 24 (blits use 8-24 VGPRs, Tensile GEMM uses 64+).
  // Fast-path for multi-WF blit copies only: wfs_per_wg >= 4 AND vgprs <= 16
  // AND signal=0 AND user_sgprs >= 8. This catches the 27MB Tensile code object
  // copy kernel but not CopyAligned (single-WF) or Tensile GEMM (high vgprs).
  // Fast-path for blit copy kernels: kcp=0x8 (only kernarg_ptr), vgprs <= 24
  // (blits use 8-24), and no scratch (private_segment_fixed_size == 0).
  // This catches CopyAligned and multi-WF blits but not HIP compute kernels
  // (which use scratch) or Tensile GEMM (which uses 64+ VGPRs).
  // Fast-path for ROCR blit copy kernels. Detected by: kcp=0x8, no scratch,
  // vgprs <= 24, AND workgroup_size is 64 or 256 (ROCR blit standard sizes).
  // This excludes HIP compute kernels which use different workgroup sizes.
  // Fast-path: skip ROCR blit copy kernels by doing memcpy directly.
  // Detected by: kcp=0x8, vgprs 16-24 (copies, not fills which use 8),
  // no scratch, and both src/dst are host-mapped.
  if (false) { // DISABLED until SDMA model is ready
    auto *ka = reinterpret_cast<const uint32_t *>(pkt.kernarg_address);
    uint64_t src = static_cast<uint64_t>(ka[0]) | (static_cast<uint64_t>(ka[1]) << 32);
    uint64_t dst = static_cast<uint64_t>(ka[2]) | (static_cast<uint64_t>(ka[3]) << 32);
    uint32_t size = ka[8];
    bool src_mapped = memory_ && size > 0 && size <= 256 * 1024 * 1024 &&
                      memory_->is_host_mapped(src) && memory_->is_host_mapped(src + size - 1);
    bool dst_mapped = memory_ && size > 0 && size <= 256 * 1024 * 1024 &&
                      memory_->is_host_mapped(dst) && memory_->is_host_mapped(dst + size - 1);
    if (src != 0 && dst != 0 && src_mapped && dst_mapped) {
      std::memcpy(reinterpret_cast<void *>(dst), reinterpret_cast<const void *>(src), size);
      util::Logger::vm("CP: blit fast-path memcpy 0x", std::hex, src, " -> 0x", dst, std::dec,
                       " size=", size, " signal=0x", std::hex, pkt.completion_signal.handle);
      // Fire completion signal if present (CopyAligned blits have signals).
      if (pkt.completion_signal.handle != 0) {
        constexpr uint32_t SIG_VAL_OFF = 8, MAILBOX_PTR_OFF = 16, EVENT_ID_OFF = 24;
        auto *val = reinterpret_cast<int64_t *>(pkt.completion_signal.handle + SIG_VAL_OFF);
        std::atomic_ref<int64_t>(*val).fetch_sub(1, std::memory_order_release);
        auto mbp = *reinterpret_cast<uint64_t *>(pkt.completion_signal.handle + MAILBOX_PTR_OFF);
        if (mbp != 0) {
          auto eid = *reinterpret_cast<uint32_t *>(pkt.completion_signal.handle + EVENT_ID_OFF);
          std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(mbp))
              .store(uint64_t(eid), std::memory_order_release);
          if (interrupt_cb_)
            interrupt_cb_(eid);
        }
      }
      return;
    }
  }

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
  dp.kernel_code_properties = kd.kernel_code_properties;
  dp.private_segment_fixed_size = kd.private_segment_fixed_size;

  // For KFD dispatches, provide pointers the kernel may need via user SGPRs.
  if (host_accessible) {
    dp.dispatch_ptr = pkt_addr;
    dp.queue_ptr = queue.read_ptr_va - offsetof(amd_queue_t, read_dispatch_id);
    if (kd.private_segment_fixed_size > 0) {
      auto *amd_queue = reinterpret_cast<const amd_queue_t *>(dp.queue_ptr);
      dp.scratch_backing_addr = amd_queue->scratch_backing_memory_location;
    }
  }

  dp.workgroup_id_offset = workgroup_id_offset_;
  dp.completion_signal = pkt.completion_signal.handle;
  dp.host_signal = host_accessible;
  dp.ordered = host_accessible;

  // Dump first 32 dwords of kernarg for all host-accessible dispatches.
  util::Logger::vm([&](auto &os) {
    if (host_accessible && dp.kernarg_addr != 0) {
      auto *ka = reinterpret_cast<const uint32_t *>(dp.kernarg_addr);
      os << "CP: kernarg dump";
      for (int i = 0; i < 256; i += 4)
        os << std::format("\n[rj log VM]   dw[{:3d}:{:3d}] = {:08x} {:08x} {:08x} {:08x}", i, i + 3,
                          ka[i], ka[i + 1], ka[i + 2], ka[i + 3]);
    }
  });
  util::Logger::vm([&](auto &os) {
    std::string sym = find_kernel_symbol(pkt.kernel_object, host_accessible);
    os << std::format("CP: dispatch \"{}\" entry_pc={:#x} kernarg={:#x}"
                      " wgs={} wfs/wg={} grid=[{},{},{}] wg=[{},{},{}]"
                      " sgprs={} vgprs={} user_sgprs={} kcp={:#x} signal={:#x}",
                      sym.empty() ? "?" : sym, entry_pc,
                      reinterpret_cast<uint64_t>(pkt.kernarg_address), total_wgs, wfs_per_wg,
                      pkt.grid_size_x, pkt.grid_size_y, pkt.grid_size_z, pkt.workgroup_size_x,
                      pkt.workgroup_size_y, pkt.workgroup_size_z, dp.sgprs_per_wf, dp.vgprs_per_wf,
                      dp.num_user_sgprs, dp.kernel_code_properties, dp.completion_signal);
  });

  // Process AQL acquire fence: invalidate caches so the kernel sees the
  // latest host/agent writes (kernarg data, input buffers, etc.).
  // On real hardware the CP issues GL1_INV + GL2_INV for SYSTEM/AGENT scope.
  uint32_t acquire_scope = (pkt.header >> HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) & 0x3;
  if (acquire_scope >= HSA_FENCE_SCOPE_AGENT && !cus_.empty()) {
    cus_[0]->flush_all();
    util::Logger::vm("CP: acquire fence scope=", acquire_scope, " → L1+L2 flush+invalidate");
  }

  plugin_group_->onAmdgpuKernelDispatch(pkt.kernel_object, entry_pc);

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

  // SDMA queues use a different packet format — process them as direct
  // memory operations (memcpy, fence, trap) without CU dispatch.
  // For SDMA queues, use the doorbell value as the write pointer (in bytes).
  // ROCR writes *queue_wptr_ = *queue_doorbell_ = cached_commit_index_ (bytes).
  // If the write pointer from write_ptr_va seems too small, use the doorbell value.
  if (queue.is_sdma) {
    // Also try reading the doorbell directly.
    void *base = doorbell_base_.load(std::memory_order_acquire);
    if (base) {
      uint64_t db_val =
          std::atomic_ref<uint64_t>(
              *reinterpret_cast<uint64_t *>(static_cast<char *>(base) + queue.doorbell_offset))
              .load(std::memory_order_acquire);
      if (db_val > write_idx)
        write_idx = db_val;
    }
    if (read_idx >= write_idx)
      return;
    // Acknowledge all SDMA submissions by advancing read_ptr to match write_ptr.
    // This unblocks ROCR's SDMA queue init. Full SDMA packet parsing is in
    // process_sdma_ring() for when HSA_ENABLE_SDMA is enabled.
    if (queue.host_accessible) {
      std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(queue.read_ptr_va))
          .store(write_idx, std::memory_order_release);
    }
    return;
  }

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
      process_aql_packet(pkt, queue, pkt_addr);
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
        constexpr uint32_t MAILBOX_PTR_OFF = 16, EVENT_ID_OFF = 24;
        uint64_t sig = 0;
        std::memcpy(&sig, reinterpret_cast<const void *>(pkt_addr + SIG_OFF), sizeof(sig));
        if (sig != 0) {
          auto *val = reinterpret_cast<int64_t *>(sig + SIG_VAL_OFF);
          std::atomic_ref<int64_t>(*val).fetch_sub(1, std::memory_order_release);
          // Write event mailbox and fire interrupt for blocked-wait signals.
          auto mailbox_ptr = *reinterpret_cast<uint64_t *>(sig + MAILBOX_PTR_OFF);
          if (mailbox_ptr != 0) {
            auto event_id = *reinterpret_cast<uint32_t *>(sig + EVENT_ID_OFF);
            std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(mailbox_ptr))
                .store(uint64_t(event_id), std::memory_order_release);
            if (interrupt_cb_)
              interrupt_cb_(event_id);
          }
          util::Logger::vm("CP: vendor pkt signal 0x", std::hex, sig, std::dec, " fired");
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
      interrupt_cb_(event_id);
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

// SDMA packet processor

// SDMA opcodes (from sdma_registers.h).
namespace sdma {
constexpr uint8_t OP_NOP = 0;
constexpr uint8_t OP_COPY = 1;
constexpr uint8_t OP_WRITE = 2;
constexpr uint8_t OP_FENCE = 5;
constexpr uint8_t OP_TRAP = 6;
constexpr uint8_t OP_POLL_REGMEM = 8;
constexpr uint8_t OP_ATOMIC = 10;
constexpr uint8_t OP_CONST_FILL = 11;
constexpr uint8_t OP_TIMESTAMP = 13;
constexpr uint8_t OP_GCR = 17;
constexpr uint8_t OP_HDP_FLUSH = 0x26; // GFX9 specific

// Packet sizes in dwords.
constexpr uint32_t COPY_LINEAR_SIZE = 7;
constexpr uint32_t FENCE_SIZE = 4;
constexpr uint32_t TRAP_SIZE = 2;
constexpr uint32_t POLL_REGMEM_SIZE = 6;
constexpr uint32_t ATOMIC_SIZE = 8;
constexpr uint32_t CONST_FILL_SIZE = 5;
constexpr uint32_t TIMESTAMP_SIZE = 3;
// NOP_BASE_SIZE intentionally omitted — NOP is handled inline.
} // namespace sdma

void CommandProcessor::process_sdma_ring(HwQueue &queue, uint64_t read_idx, uint64_t write_idx) {
  auto *ring = reinterpret_cast<const uint32_t *>(queue.ring_base_va);
  uint32_t ring_mask = (queue.ring_size / sizeof(uint32_t)) - 1;

  // SDMA write/read pointers are in DWORD units on GFX9 hardware.
  // ROCR writes cached_commit_index_ (which accumulates dword-sized packet sizes)
  // to both *queue_wptr_ and *queue_doorbell_.
  uint64_t rpos = read_idx;
  uint64_t wpos = write_idx;

  auto dw = [&](uint64_t off) -> uint32_t { return ring[(rpos + off) & ring_mask]; };

  while (rpos < wpos) {
    uint32_t header = dw(0);
    uint8_t op = header & 0xFF;
    uint32_t pkt_dwords = 0;

    switch (op) {
    case sdma::OP_NOP: {
      uint32_t count = (dw(0) >> 16) & 0x3FFF;
      pkt_dwords = 1 + count;
      break;
    }
    case sdma::OP_COPY: {
      if (rpos + sdma::COPY_LINEAR_SIZE > wpos) {
        rpos = wpos;
        continue;
      }
      uint32_t count = (dw(1) & 0x3FFFFF) + 1;
      uint64_t src = static_cast<uint64_t>(dw(3)) | (static_cast<uint64_t>(dw(4)) << 32);
      uint64_t dst = static_cast<uint64_t>(dw(5)) | (static_cast<uint64_t>(dw(6)) << 32);
      if (header & (1u << 28)) {
        // Broadcast copy (2 destinations) — 9 dwords.
        uint64_t dst2 = static_cast<uint64_t>(dw(7)) | (static_cast<uint64_t>(dw(8)) << 32);
        std::memcpy(reinterpret_cast<void *>(dst), reinterpret_cast<const void *>(src), count);
        std::memcpy(reinterpret_cast<void *>(dst2), reinterpret_cast<const void *>(src), count);
        pkt_dwords = 9;
      } else {
        std::memcpy(reinterpret_cast<void *>(dst), reinterpret_cast<const void *>(src), count);
        pkt_dwords = sdma::COPY_LINEAR_SIZE;
      }
      util::Logger::vm("SDMA: copy 0x", std::hex, src, " -> 0x", dst, std::dec, " size=", count);
      break;
    }
    case sdma::OP_FENCE: {
      uint64_t addr = static_cast<uint64_t>(dw(1)) | (static_cast<uint64_t>(dw(2)) << 32);
      uint32_t data = dw(3);
      std::atomic_ref<uint32_t>(*reinterpret_cast<uint32_t *>(addr))
          .store(data, std::memory_order_release);
      pkt_dwords = sdma::FENCE_SIZE;
      break;
    }
    case sdma::OP_TRAP: {
      uint32_t event_id = dw(1) & 0x0FFFFFFF;
      if (interrupt_cb_)
        interrupt_cb_(event_id);
      pkt_dwords = sdma::TRAP_SIZE;
      break;
    }
    case sdma::OP_POLL_REGMEM: {
      // Poll memory until (value & mask) == reference.
      // In functional mode, the polled value should already be ready
      // (no pipeline latency). Spin briefly if needed.
      uint64_t addr = static_cast<uint64_t>(dw(1)) | (static_cast<uint64_t>(dw(2)) << 32);
      uint32_t ref = dw(3);
      uint32_t mask = dw(4);
      auto *ptr = reinterpret_cast<uint32_t *>(addr);
      for (int i = 0; i < 10000; ++i) {
        uint32_t val = std::atomic_ref<uint32_t>(*ptr).load(std::memory_order_acquire);
        if ((val & mask) == ref)
          break;
      }
      pkt_dwords = sdma::POLL_REGMEM_SIZE;
      break;
    }
    case sdma::OP_ATOMIC: {
      uint64_t addr = static_cast<uint64_t>(dw(1)) | (static_cast<uint64_t>(dw(2)) << 32);
      uint64_t src_data = static_cast<uint64_t>(dw(3)) | (static_cast<uint64_t>(dw(4)) << 32);
      uint32_t atomic_op = (header >> 25) & 0x7F;
      // SDMA_ATOMIC_ADD64 = 47
      if (atomic_op == 47) {
        auto *ptr = reinterpret_cast<int64_t *>(addr);
        std::atomic_ref<int64_t>(*ptr).fetch_add(static_cast<int64_t>(src_data),
                                                 std::memory_order_release);
      }
      pkt_dwords = sdma::ATOMIC_SIZE;
      break;
    }
    case sdma::OP_CONST_FILL: {
      uint64_t addr = static_cast<uint64_t>(dw(1)) | (static_cast<uint64_t>(dw(2)) << 32);
      uint32_t data = dw(3);
      uint32_t count = (dw(4) & 0x3FFFFF) + 1;
      uint32_t fillsize = (header >> 30) & 0x3;
      auto *dst = reinterpret_cast<uint8_t *>(addr);
      if (fillsize == 2) { // 32-bit fill
        for (uint32_t i = 0; i < count; i += 4)
          std::memcpy(dst + i, &data, 4);
      } else {
        std::memset(dst, static_cast<int>(data & 0xFF), count);
      }
      pkt_dwords = sdma::CONST_FILL_SIZE;
      break;
    }
    case sdma::OP_TIMESTAMP:
      pkt_dwords = sdma::TIMESTAMP_SIZE;
      break;
    case sdma::OP_GCR:
      pkt_dwords = 5; // GCR request is 5 dwords on GFX9.
      break;
    case sdma::OP_HDP_FLUSH:
      pkt_dwords = 1;
      break;
    case sdma::OP_WRITE: {
      if (rpos + 4 > wpos) {
        rpos = wpos;
        continue;
      }
      uint32_t count = (dw(3) & 0x3FFFFF) + 1;
      uint64_t addr = static_cast<uint64_t>(dw(1)) | (static_cast<uint64_t>(dw(2)) << 32);
      if (addr > 0x1000 && rpos + 4 + count <= wpos) {
        auto *dst = reinterpret_cast<uint32_t *>(addr);
        for (uint32_t i = 0; i < count; ++i)
          dst[i] = dw(4 + i);
      }
      pkt_dwords = 4 + count;
      break;
    }
    default:
      // Unknown opcode — stop processing to avoid reading garbage.
      util::Logger::vm("SDMA: unknown opcode 0x", std::hex, (unsigned)op, std::dec,
                       " at rpos=", std::dec, rpos);
      rpos = wpos; // Consume remaining to prevent infinite loop.
      continue;
    }

    rpos += pkt_dwords;
  }

  // Update the read pointer to match what we consumed (in dword units).
  if (queue.host_accessible) {
    std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(queue.read_ptr_va))
        .store(rpos, std::memory_order_release);
  }
}

} // namespace amdgpu
} // namespace rocjitsu
