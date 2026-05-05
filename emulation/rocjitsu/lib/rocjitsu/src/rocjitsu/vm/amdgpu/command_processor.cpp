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
#include <format>
#include <limits>
#include <set>
#include <string>
#include <sys/mman.h>
#include <thread>

namespace rocjitsu {
namespace amdgpu {

void CommandProcessor::init_wavefront_regs(ComputeUnitCore *cu, Wavefront *wf,
                                           const DispatchEntry &pkt, uint32_t global_wg_id,
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
    if (AMDHSA_BITS_GET(kcp, KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER)) {
      if (pkt.queue_ptr != 0) {
        auto *q = reinterpret_cast<const amd_queue_t *>(pkt.queue_ptr);
        cu->write_sgpr(sbase + idx + 0, q->scratch_resource_descriptor[0]);
        cu->write_sgpr(sbase + idx + 1, q->scratch_resource_descriptor[1]);
        cu->write_sgpr(sbase + idx + 2, q->scratch_resource_descriptor[2]);
        cu->write_sgpr(sbase + idx + 3, q->scratch_resource_descriptor[3]);
      }
      idx += 4;
    }
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
      uint64_t dispatch_id = 0;
      if (pkt.queue_ptr != 0) {
        auto *q = reinterpret_cast<const amd_queue_t *>(pkt.queue_ptr);
        dispatch_id = q->write_dispatch_id;
      }
      cu->write_sgpr(sbase + idx, static_cast<uint32_t>(dispatch_id));
      cu->write_sgpr(sbase + idx + 1, static_cast<uint32_t>(dispatch_id >> 32));
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

  // System SGPRs: workgroup_id_{x,y,z} placed sequentially after user SGPRs.
  // Only the IDs whose enable bits are set in compute_pgm_rsrc2 are written.
  // When kernel_code_properties is 0 (internal test dispatches), always write
  // workgroup_id_x as a fallback since internal kernels expect it.
  uint32_t sys_idx = pkt.num_user_sgprs;
  {
    uint32_t gx = pkt.grid_wgs_x > 0 ? pkt.grid_wgs_x : 1;
    uint32_t gy = pkt.grid_wgs_y > 0 ? pkt.grid_wgs_y : 1;
    bool kcp_zero = (pkt.kernel_code_properties == 0);
    if (pkt.enable_wg_id_x || kcp_zero) {
      uint32_t wg_x = (pkt.enable_wg_id_y || pkt.enable_wg_id_z) ? global_wg_id % gx : global_wg_id;
      cu->write_sgpr(sbase + sys_idx++, wg_x);
    }
    if (pkt.enable_wg_id_y)
      cu->write_sgpr(sbase + sys_idx++, (global_wg_id / gx) % gy);
    if (pkt.enable_wg_id_z)
      cu->write_sgpr(sbase + sys_idx++, global_wg_id / (gx * gy));
  }

  // Workitem IDs per AMDHSA ABI. The SPI decomposes the flat thread index
  // into (x, y, z) using the AQL packet's workgroup dimensions.
  // enable_vgpr_workitem_id (TIDIG_COMP_CNT from compute_pgm_rsrc2):
  //   0 = v0 only (workitem_id_x)
  //   1 = v0 + v1 (workitem_id_x, workitem_id_y)
  //   2 = v0 + v1 + v2 (workitem_id_x, workitem_id_y, workitem_id_z)
  // On CDNA3/4 (PackedTID): v0[9:0]=X, v0[19:10]=Y, v0[29:20]=Z.
  // v1/v2 are not written. Kernel extracts components via bit masks.
  uint32_t vbase = wf->vgpr_alloc().base;
  uint32_t workitem_base = wf_index_in_wg * cu->wf_size();
  uint32_t wg_x = pkt.workgroup_size_x > 0 ? pkt.workgroup_size_x : 1;
  uint32_t wg_y = pkt.workgroup_size_y > 0 ? pkt.workgroup_size_y : 1;
  uint32_t wg_xy = wg_x * wg_y;
  for (uint32_t lane = 0; lane < cu->wf_size(); ++lane) {
    uint32_t flat_id = workitem_base + lane;
    uint32_t id_x = flat_id % wg_x;
    uint32_t id_y = (flat_id / wg_x) % wg_y;
    uint32_t id_z = flat_id / wg_xy;
    if (packed_tid_ && pkt.enable_vgpr_workitem_id > 0) {
      uint32_t packed = (id_x & 0x3FFu) | ((id_y & 0x3FFu) << 10) | ((id_z & 0x3FFu) << 20);
      cu->write_vgpr(vbase, lane, packed);
    } else {
      cu->write_vgpr(vbase, lane, id_x);
      if (pkt.enable_vgpr_workitem_id >= 1)
        cu->write_vgpr(vbase + 1, lane, id_y);
      if (pkt.enable_vgpr_workitem_id >= 2)
        cu->write_vgpr(vbase + 2, lane, id_z);
    }
  }

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
  completion_ = std::make_unique<CompletionTracker>(memory_, cus_);
  if (interrupt_cb_)
    completion_->set_interrupt_callback(interrupt_cb_);
}

void CommandProcessor::register_queue(HwQueue queue) {
  util::Logger::vm([&](auto &os) {
    os << std::format("REGISTER_QUEUE id={} ring={:#x} size={} rptr={:#x} wptr={:#x} "
                      "doorbell_off={} is_sdma={}",
                      queue.queue_id, queue.ring_base_va, queue.ring_size, queue.read_ptr_va,
                      queue.write_ptr_va, queue.doorbell_offset, queue.is_sdma);
  });
  bool start_poll = queue.host_accessible;
  {
    std::lock_guard<std::mutex> lock(hw_queue_mutex_);
    hw_queues_.push_back(std::move(queue));
    new_queue_states_.push_back(HwQueueState{});
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
    for (size_t i = 0; i < hw_queues_.size(); ++i) {
      if (hw_queues_[i].queue_id == queue_id) {
        hw_queues_.erase(hw_queues_.begin() + static_cast<ptrdiff_t>(i));
        new_queue_states_.erase(new_queue_states_.begin() + static_cast<ptrdiff_t>(i));
        break;
      }
    }
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

HwQueueState *CommandProcessor::schedule_next_queue() {
  if (new_queue_states_.empty())
    return nullptr;
  size_t start = next_queue_idx_;
  for (size_t i = 0; i < new_queue_states_.size(); ++i) {
    size_t idx = (start + i) % new_queue_states_.size();
    auto &qs = new_queue_states_[idx];
    if (hw_queues_[idx].is_sdma)
      continue;
    if (qs.next_dispatch_idx < qs.entries.size()) {
      next_queue_idx_ = (idx + 1) % new_queue_states_.size();
      return &qs;
    }
  }
  return nullptr;
}

bool CommandProcessor::barrier_satisfied(const HwQueueState &qs, size_t idx) const {
  if (idx == 0 && !qs.implicit_barrier_next)
    return true;

  // Barrier bit: all prior entries must be fully completed.
  for (size_t i = 0; i < idx; ++i) {
    if (!qs.entries[i].fully_completed())
      return false;
  }
  return true;
}

uint32_t CommandProcessor::dispatch_workgroups(DispatchEntry &entry) {
  assert(!cus_.empty() && "command processor has no compute units");

  // Activate wavefront slots for each workgroup, distributing across CUs.
  // Entire workgroups must land on a single CU (programming model requirement:
  // LDS is per-CU and s_barrier synchronises within a CU). Query each CU for
  // capacity before dispatching to guarantee all-or-nothing placement.
  uint32_t dispatched = 0;
  while (entry.dispatched_wgs < entry.total_wgs) {
    uint32_t global_wg_id = entry.dispatched_wgs + entry.workgroup_id_offset;
    bool placed = false;

    for (size_t attempt = 0; attempt < cus_.size() && !placed; ++attempt) {
      size_t cu_idx = (next_cu_ + attempt) % cus_.size();
      ComputeUnitCore *cu = cus_[cu_idx];
      cu->retire_halted_wfs();
      if (!cu->can_accept_workgroup(entry.wfs_per_workgroup, entry.group_segment_fixed_size))
        continue;

      uint32_t lds_base = cu->allocate_lds(entry.group_segment_fixed_size);
      cu->begin_workgroup(entry.dispatch_id, global_wg_id, entry.wfs_per_workgroup);

      std::vector<Wavefront *> wg_wavefronts;
      wg_wavefronts.reserve(entry.wfs_per_workgroup);
      for (uint32_t w = 0; w < entry.wfs_per_workgroup; ++w) {
        Wavefront *wf = cu->dispatch_wf(global_wg_id, entry.kernel_entry_pc, entry.sgprs_per_wf,
                                        entry.vgprs_per_wf);
        assert(wf && "dispatch_wf failed after can_accept_workgroup returned true");
        wf->set_lds_base(lds_base);
        wf->set_dispatch_id(entry.dispatch_id);
        init_wavefront_regs(cu, wf, entry, global_wg_id, w);
        wg_wavefronts.push_back(wf);
      }
      plugin_group_->onAmdgpuDispatchWorkgroup(global_wg_id, entry.vgprs_per_wf, entry.sgprs_per_wf,
                                               std::span<Wavefront *>(wg_wavefronts));
      next_cu_ = (cu_idx + 1) % cus_.size();
      placed = true;
    }

    if (!placed)
      break; // Backpressure: all CUs full. Will resume when WG completes.

    ++entry.dispatched_wgs;
    ++dispatched;
  }
  return dispatched;
}

// ---------------------------------------------------------------------------
// Completion notification from CU
// ---------------------------------------------------------------------------

void CommandProcessor::notify_wg_complete(uint32_t dispatch_id, uint32_t wg_id) {
  if (completion_)
    completion_->notify_wg_complete(dispatch_id, wg_id, new_queue_states_);
  // In functional mode (quantum=0), the handle_doorbell loop handles dispatch
  // resume. In quantum>0 mode, CU work events will call on_cu_idle which can
  // trigger further dispatch via the engine event loop.
}

void CommandProcessor::on_cu_idle() {
  // In quantum=0 mode, handle_doorbell drives the dispatch-execute-complete
  // loop synchronously. on_cu_idle fires INSIDE activate(), which is INSIDE
  // handle_doorbell's dispatch loop. Dispatching more WGs here would re-enter
  // dispatch_workgroups and cause double-dispatch. So skip in quantum=0.
  //
  // In quantum>0 mode, CU execution is asynchronous (via work events). When
  // a CU finishes its quantum and goes idle, we should drain completions and
  // resume stalled dispatches. This is safe because we're not inside
  // handle_doorbell's dispatch loop.
  if (!cus_.empty() && cus_[0]->config().functional_quantum == 0)
    return;

  if (completion_)
    completion_->drain_completions(new_queue_states_);

  for (auto &qs : new_queue_states_) {
    if (qs.next_dispatch_idx < qs.entries.size()) {
      auto &entry = qs.entries[qs.next_dispatch_idx];
      if (!entry.is_non_kernel() && !entry.fully_dispatched()) {
        uint32_t sent = dispatch_workgroups(entry);
        if (sent > 0 && entry.fully_dispatched())
          ++qs.next_dispatch_idx;
      }
    }
  }
}

bool CommandProcessor::step() {
  // Process dispatches across all queues.
  process_queues();
  return pending_entries() > 0;
}

void CommandProcessor::process_queues() {
  for (size_t qi = 0; qi < new_queue_states_.size(); ++qi) {
    if (hw_queues_[qi].is_sdma)
      continue;
    auto &qs = new_queue_states_[qi];
    while (qs.next_dispatch_idx < qs.entries.size()) {
      auto &entry = qs.entries[qs.next_dispatch_idx];

      if (entry.barrier_bit && !barrier_satisfied(qs, qs.next_dispatch_idx))
        break; // Stalled on barrier bit.

      if (entry.is_non_kernel()) {
        entry.completed_wgs = entry.total_wgs; // 0 == 0, immediately complete.
        ++qs.next_dispatch_idx;
        continue;
      }

      uint32_t sent = dispatch_workgroups(entry);
      if (entry.fully_dispatched())
        ++qs.next_dispatch_idx;
      if (sent == 0)
        break; // CU backpressure.
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

/// @brief Find the kernel symbol name from the code object's AMDHSA metadata.
/// @details Uses GpuMemory::find_host_range to locate the ELF base. Parses the
/// PT_NOTE segment to find NT_AMDGPU_METADATA (msgpack-encoded). Extracts kernel
/// names by scanning for ".kd" symbol strings in the raw msgpack data and matches
/// to the dispatched kernel using the kernel descriptor fields (group_segment_fixed_size,
/// private_segment_fixed_size) read from the kernel descriptor at kernel_object.
static std::string find_kernel_symbol(uint64_t kernel_object, GpuMemory *mem) {
  if (kernel_object == 0 || !mem)
    return {};

  auto [range_base, range_size] = mem->find_host_range(kernel_object);
  if (range_base == 0)
    return {};

  auto *elf_base = reinterpret_cast<const uint8_t *>(range_base);
  if (elf_base[0] != 0x7f || elf_base[1] != 'E' || elf_base[2] != 'L' || elf_base[3] != 'F')
    return {};

  auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(elf_base);
  if (ehdr->e_phnum == 0 || ehdr->e_phoff == 0)
    return {};

  auto *phdrs = reinterpret_cast<const Elf64_Phdr *>(elf_base + ehdr->e_phoff);

  // Find PT_NOTE containing AMDHSA metadata.
  for (uint16_t i = 0; i < ehdr->e_phnum; ++i) {
    if (phdrs[i].p_type != PT_NOTE || phdrs[i].p_filesz < sizeof(Elf64_Nhdr))
      continue;
    if (phdrs[i].p_offset + phdrs[i].p_filesz > range_size)
      continue;
    auto *nhdr = reinterpret_cast<const Elf64_Nhdr *>(elf_base + phdrs[i].p_offset);
    constexpr uint32_t NT_AMDGPU_METADATA = 32;
    if (nhdr->n_type != NT_AMDGPU_METADATA)
      continue;

    uint32_t name_aligned = (nhdr->n_namesz + 3) & ~3u;
    uint64_t desc_off = phdrs[i].p_offset + sizeof(Elf64_Nhdr) + name_aligned;
    uint32_t desc_sz = nhdr->n_descsz;
    if (desc_off + desc_sz > range_size)
      continue;
    auto *note = elf_base + desc_off;

    // Scan raw msgpack for ".kd" symbol strings. Each kernel entry has a
    // ".symbol" field containing "kernel_name.kd" as a length-prefixed string.
    // We extract the kernel name and check if the corresponding kernel
    // descriptor matches our dispatch (by group_segment_fixed_size and
    // private_segment_fixed_size). For single-kernel dispatches, the first
    // match is sufficient.
    std::string best_name;
    for (size_t pos = 2; pos + 2 < desc_sz; ++pos) {
      if (note[pos] != '.' || note[pos + 1] != 'k' || note[pos + 2] != 'd')
        continue;
      size_t end = pos + 3;
      if (end < desc_sz && note[end] >= 0x20 && note[end] < 0x7f)
        continue;
      size_t start = pos;
      while (start > 0 && note[start - 1] >= 0x20 && note[start - 1] < 0x7f)
        --start;
      if (start == pos)
        continue;
      std::string_view sym(reinterpret_cast<const char *>(note + start), pos - start);
      if (best_name.empty())
        best_name = std::string(sym);
    }
    if (!best_name.empty())
      return best_name;
  }
  return {};
}

void CommandProcessor::process_aql_packet(const hsa_kernel_dispatch_packet_t &pkt,
                                          const HwQueue &queue, uint64_t pkt_addr,
                                          HwQueueState &qs) {
  bool host_accessible = queue.host_accessible;
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
    // Register the code region (kernel object + margin for the code body).
    // Use the VRAM allocation size if available; fallback to 2MB for large code objects.
    uint64_t code_base = pkt.kernel_object & ~0xFFFULL; // Page-align down.
    constexpr size_t CODE_MAP_SIZE = 2 << 20;           // 2MB to cover large code objects.
    memory_->map_host_pages(code_base, reinterpret_cast<void *>(code_base), CODE_MAP_SIZE);

    // Register the kernarg region. Map enough pages to cover the full kernarg
    // buffer. PyTorch reduction kernels can have kernarg buffers exceeding 4KB
    // (TensorIterator packs many pointers, strides, and flags).
    uint64_t karg = reinterpret_cast<uint64_t>(pkt.kernarg_address);
    if (karg != 0) {
      uint64_t karg_base = karg & ~0xFFFULL;
      constexpr size_t KARG_MAP_SIZE = 8 * 4096; // 32KB covers large kernarg buffers.
      memory_->map_host_pages(karg_base, reinterpret_cast<void *>(karg_base), KARG_MAP_SIZE);
    }
  }

  uint32_t wg_size =
      static_cast<uint32_t>(pkt.workgroup_size_x) * pkt.workgroup_size_y * pkt.workgroup_size_z;
  uint32_t wave_size = cus_.empty() ? 64 : cus_[0]->wf_size();
  uint32_t wfs_per_wg = (wg_size + wave_size - 1) / wave_size;

  uint32_t num_dims = pkt.setup & 0x3;
  uint32_t grid_wgs_x = pkt.workgroup_size_x > 0 ? pkt.grid_size_x / pkt.workgroup_size_x : 1;
  uint32_t grid_wgs_y = pkt.workgroup_size_y > 0 ? pkt.grid_size_y / pkt.workgroup_size_y : 1;
  uint32_t grid_wgs_z = pkt.workgroup_size_z > 0 ? pkt.grid_size_z / pkt.workgroup_size_z : 1;
  uint32_t total_wgs = grid_wgs_x * grid_wgs_y * grid_wgs_z;

  DispatchEntry dp{};
  dp.dispatch_id = next_dispatch_id_++;
  dp.queue_id = queue.queue_id;
  dp.kernel_entry_pc = entry_pc;
  dp.total_wgs = total_wgs;
  dp.dispatched_wgs = 0;
  dp.completed_wgs = 0;
  dp.wfs_per_workgroup = wfs_per_wg;
  dp.sgprs_per_wf = sgprs > 0 ? sgprs : 104;
  dp.vgprs_per_wf = vgprs > 0 ? vgprs : 256;
  dp.kernarg_addr = reinterpret_cast<uint64_t>(pkt.kernarg_address);
  dp.num_user_sgprs = user_sgprs;
  dp.kernel_code_properties = kd.kernel_code_properties;
  dp.private_segment_fixed_size = kd.private_segment_fixed_size;
  dp.group_segment_fixed_size = std::max(kd.group_segment_fixed_size, pkt.group_segment_size);

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
  // For WG ID decomposition, use the dispatch dimensionality (setup field).
  // A 1D dispatch flattens the entire grid into workgroup_id_x.
  dp.grid_wgs_x = (num_dims <= 1) ? total_wgs : grid_wgs_x;
  dp.grid_wgs_y = (num_dims >= 2) ? grid_wgs_y : 1;
  dp.grid_wgs_z = (num_dims >= 3) ? grid_wgs_z : 1;
  dp.enable_wg_id_x =
      AMDHSA_BITS_GET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X);
  dp.enable_wg_id_y =
      AMDHSA_BITS_GET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y);
  dp.enable_wg_id_z =
      AMDHSA_BITS_GET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z);
  dp.enable_vgpr_workitem_id = static_cast<uint8_t>(
      AMDHSA_BITS_GET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID));
  dp.workgroup_size_x = pkt.workgroup_size_x;
  dp.workgroup_size_y = pkt.workgroup_size_y;
  dp.workgroup_size_z = pkt.workgroup_size_z;
  dp.completion_signal = pkt.completion_signal.handle;
  dp.host_signal = host_accessible;
  dp.barrier_bit = (pkt.header >> HSA_PACKET_HEADER_BARRIER) & 1;

  // Process AQL acquire fence: invalidate caches so the kernel sees the
  // latest host/agent writes (kernarg data, input buffers, etc.).
  // On real hardware the CP issues GL1_INV + GL2_INV for SYSTEM/AGENT scope.
  uint32_t acquire_scope = (pkt.header >> HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) & 0x3;
  if (acquire_scope >= HSA_FENCE_SCOPE_AGENT && !cus_.empty()) {
    for (auto *cu : cus_)
      cu->flush_all();
  }

  plugin_group_->onAmdgpuKernelDispatch(pkt.kernel_object, entry_pc);
  util::Logger::vm([&](auto &os) {
    os << std::format("DISPATCH d={} pc={:#x} grid=[{},{},{}] wg=[{},{},{}] total_wgs={} "
                      "sgprs={} vgprs={} lds={} kernarg={:#x}",
                      dp.dispatch_id, entry_pc, dp.grid_wgs_x, dp.grid_wgs_y, dp.grid_wgs_z,
                      pkt.workgroup_size_x, pkt.workgroup_size_y, pkt.workgroup_size_z, total_wgs,
                      dp.sgprs_per_wf, dp.vgprs_per_wf, dp.group_segment_fixed_size,
                      dp.kernarg_addr);
  });

  {
    static uint32_t dispatch_count = 0;
    ++dispatch_count;
    util::Logger::vm([&](auto &os) {
      std::string sym = find_kernel_symbol(pkt.kernel_object, memory_);
      os << std::format("dispatch #{} d={} \"{}\" grid=[{},{},{}] wg=[{},{},{}] wgs={} "
                        "lds={} sgpr={} vgpr={} sig={:#x}",
                        dispatch_count, dp.dispatch_id, sym.empty() ? "?" : sym, pkt.grid_size_x,
                        pkt.grid_size_y, pkt.grid_size_z, pkt.workgroup_size_x,
                        pkt.workgroup_size_y, pkt.workgroup_size_z, total_wgs,
                        kd.group_segment_fixed_size, dp.sgprs_per_wf, dp.vgprs_per_wf,
                        dp.completion_signal);
    });
  }

  // Register with completion tracker for fast dispatch_id -> queue lookup.
  for (size_t qi = 0; qi < hw_queues_.size(); ++qi) {
    if (&hw_queues_[qi] == &queue) {
      if (completion_)
        completion_->register_dispatch(dp.dispatch_id, qi);
      break;
    }
  }

  ++total_dispatched_;
  qs.entries.push_back(std::move(dp));
}

void CommandProcessor::fetch_from_queue(HwQueue &queue, HwQueueState &qs) {
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

  util::Logger::vm([&](auto &os) {
    static uint64_t fetch_count = 0;
    if (write_idx != read_idx && ++fetch_count <= 50)
      os << std::format("FETCH q={} w={} r={} delta={} sdma={}", queue.queue_id, write_idx,
                        read_idx, write_idx - read_idx, queue.is_sdma);
  });

  // SDMA queues use byte-granularity pointers and have their own doorbell
  // semantics — skip the AQL doorbell clamping that assumes packet indices.
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
    process_sdma_ring(queue, read_idx, write_idx);
    return;
  }

  // AQL doorbell clamping (compute queues only).
  uint64_t process_limit = write_idx;
  if (queue.host_accessible) {
    const uint64_t doorbell = queue.last_doorbell;
    if (doorbell != std::numeric_limits<uint64_t>::max()) {
      uint64_t doorbell_limit = doorbell + 1;
      if (doorbell_limit < process_limit)
        process_limit = doorbell_limit;
    }
    if (read_idx >= process_limit)
      return;
  } else if (read_idx >= process_limit) {
    return;
  }

  constexpr uint32_t AQL_PACKET_SIZE = 64;
  uint32_t num_slots = queue.ring_size / AQL_PACKET_SIZE;

  while (read_idx < process_limit) {
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
    util::Logger::vm([&](auto &os) {
      os << std::format("PKT q={} slot={} type={} header={:#x} read_idx={}", queue.queue_id, slot,
                        pkt_type, pkt.header, read_idx);
    });

    // INVALID packet: the producer reserved this slot but hasn't written the
    // header yet. Stop fetching — the doorbell poll thread will fire another
    // event when the producer commits. Non-blocking to avoid deadlocking when
    // the host thread is blocked on a CP completion signal.
    if (pkt_type == HSA_PACKET_TYPE_INVALID) {
      process_limit = read_idx;
      break;
    }

    if (pkt_type == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
      process_aql_packet(pkt, queue, pkt_addr, qs);
    } else if (pkt_type == HSA_PACKET_TYPE_BARRIER_AND || pkt_type == HSA_PACKET_TYPE_BARRIER_OR) {
      constexpr uint32_t DEP_OFF = 8;
      constexpr uint32_t SIG_OFF = 56;
      constexpr uint32_t SIG_VAL_OFF = 8;
      bool is_and = (pkt_type == HSA_PACKET_TYPE_BARRIER_AND);

      // Non-blocking dependency check: if any dependency is unsatisfied,
      // stop fetching from this queue. The next doorbell event will retry.
      // This prevents blocking the CP from processing other queues (SDMA)
      // that may be responsible for satisfying these dependencies.
      bool deps_satisfied =
          is_and; // AND: assume true until a dep fails; OR: assume false until one passes
      bool has_deps = false;
      for (int dep = 0; dep < 5; ++dep) {
        uint64_t dep_sig = 0;
        if (queue.host_accessible)
          std::memcpy(&dep_sig, reinterpret_cast<const void *>(pkt_addr + DEP_OFF + dep * 8),
                      sizeof(dep_sig));
        else
          dep_sig = read_gpu_u64(pkt_addr + DEP_OFF + dep * 8);
        if (dep_sig == 0)
          continue;
        has_deps = true;
        auto *val = reinterpret_cast<int64_t *>(dep_sig + SIG_VAL_OFF);
        int64_t v = std::atomic_ref<int64_t>(*val).load(std::memory_order_acquire);
        if (v > 0) {
          if (is_and) {
            deps_satisfied = false;
            break;
          }
        } else {
          if (!is_and) {
            deps_satisfied = true;
            break;
          }
        }
      }
      if (!has_deps)
        deps_satisfied = true;
      if (!deps_satisfied) {
        // Dependencies not ready. Stop fetching — don't advance read pointer
        // past this packet. Schedule a re-check via doorbell event.
        process_limit = read_idx;
        engine()->schedule_event_now(&doorbell_event_);
        break;
      }

      uint64_t sig = 0;
      if (queue.host_accessible)
        std::memcpy(&sig, reinterpret_cast<const void *>(pkt_addr + SIG_OFF), sizeof(sig));
      else
        sig = read_gpu_u64(pkt_addr + SIG_OFF);

      DispatchEntry dp{};
      dp.dispatch_id = next_dispatch_id_++;
      dp.queue_id = queue.queue_id;
      dp.total_wgs = 0;
      dp.completed_wgs = 0;
      dp.dispatched_wgs = 0;
      dp.completion_signal = sig;
      dp.host_signal = queue.host_accessible;

      for (size_t qi = 0; qi < hw_queues_.size(); ++qi) {
        if (&hw_queues_[qi] == &queue) {
          if (completion_)
            completion_->register_dispatch(dp.dispatch_id, qi);
          break;
        }
      }
      qs.entries.push_back(std::move(dp));
    } else if (pkt_type == HSA_PACKET_TYPE_VENDOR_SPECIFIC) {
      constexpr uint32_t SIG_OFF = 56;
      uint64_t sig = 0;
      if (queue.host_accessible)
        std::memcpy(&sig, reinterpret_cast<const void *>(pkt_addr + SIG_OFF), sizeof(sig));
      else
        sig = read_gpu_u64(pkt_addr + SIG_OFF);

      DispatchEntry dp{};
      dp.dispatch_id = next_dispatch_id_++;
      dp.queue_id = queue.queue_id;
      dp.total_wgs = 0;
      dp.completed_wgs = 0;
      dp.dispatched_wgs = 0;
      dp.completion_signal = sig;
      dp.host_signal = queue.host_accessible;

      for (size_t qi = 0; qi < hw_queues_.size(); ++qi) {
        if (&hw_queues_[qi] == &queue) {
          if (completion_)
            completion_->register_dispatch(dp.dispatch_id, qi);
          break;
        }
      }
      qs.entries.push_back(std::move(dp));
    }

    ++read_idx;
  }

  // Write updated read pointer.
  if (queue.host_accessible) {
    std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(queue.read_ptr_va))
        .store(process_limit, std::memory_order_release);
  } else {
    auto *src = reinterpret_cast<const uint8_t *>(&process_limit);
    for (uint32_t i = 0; i < sizeof(read_idx); ++i)
      memory_->write8(queue.read_ptr_va + i, src[i]);
  }
}

void CommandProcessor::fetch_packets() {
  std::lock_guard<std::mutex> lock(hw_queue_mutex_);
  for (size_t i = 0; i < hw_queues_.size(); ++i)
    fetch_from_queue(hw_queues_[i], new_queue_states_[i]);
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
  // Hold hw_queue_mutex_ for the entire handler. register_queue/unregister_queue
  // from the host thread can resize hw_queues_/new_queue_states_ at any time;
  // iterating without the lock risks use-after-realloc. The host thread blocks
  // on queue registration until handle_doorbell returns, which is acceptable
  // because queue creation only happens during HSA init (not during dispatch).
  // Use unique_lock so we can unlock before stop_doorbell_monitor() (which
  // joins the doorbell thread that also acquires this mutex).
  std::unique_lock<std::mutex> lock(hw_queue_mutex_);

  // Fetch packets (uses last_doorbell values set by the poll thread).
  for (size_t i = 0; i < hw_queues_.size(); ++i)
    fetch_from_queue(hw_queues_[i], new_queue_states_[i]);

  // Ensure interrupt callback is set on completion tracker.
  if (completion_ && interrupt_cb_)
    completion_->set_interrupt_callback(interrupt_cb_);

  // Phase 1: Dispatch-Execute-Complete loop (functional mode).
  bool progress = true;
  while (progress) {
    progress = false;

    for (size_t qi = 0; qi < hw_queues_.size(); ++qi) {
      if (hw_queues_[qi].is_sdma)
        continue;
      auto &qs = new_queue_states_[qi];

      while (qs.next_dispatch_idx < qs.entries.size()) {
        auto &entry = qs.entries[qs.next_dispatch_idx];

        if (entry.barrier_bit && !barrier_satisfied(qs, qs.next_dispatch_idx))
          break;

        if (entry.is_non_kernel()) {
          entry.completed_wgs = entry.total_wgs;
          ++qs.next_dispatch_idx;
          if (completion_)
            completion_->drain_completions(new_queue_states_);
          progress = true;
          continue;
        }

        // Dispatch-execute-retire loop: keep dispatching WGs, activating CUs,
        // and retiring WFs until the entry is fully dispatched and completed,
        // or we hit genuine backpressure (no CU can accept any WG).
        // NOTE: drain_completions may pop entries, so we must re-check indices
        // after each drain and not hold stale references.
        uint32_t dispatch_id = entry.dispatch_id;
        for (;;) {
          if (qs.next_dispatch_idx >= qs.entries.size())
            break;
          auto &cur = qs.entries[qs.next_dispatch_idx];
          if (cur.dispatch_id != dispatch_id)
            break; // Entry was popped (completed).

          uint32_t sent = dispatch_workgroups(cur);
          if (sent > 0)
            progress = true;

          // Activate CUs synchronously (functional mode, quantum=0).
          for (auto *cu : cus_) {
            if (cu->has_active_wfs())
              cu->activate();
          }

          // WG completion notifications fire automatically from
          // Wavefront::halt() → CU::release_wf() → CP::notify_wg_complete().
          // Drain completions to pop fully-completed entries and fire signals.
          if (completion_)
            completion_->drain_completions(new_queue_states_);

          // Re-check: entry may have been popped by drain.
          if (qs.next_dispatch_idx >= qs.entries.size())
            break;
          auto &post = qs.entries[qs.next_dispatch_idx];
          if (post.dispatch_id != dispatch_id)
            break; // Entry completed and was popped.

          if (post.fully_dispatched()) {
            ++qs.next_dispatch_idx;
            break;
          }
          if (sent == 0)
            break; // CU backpressure.
        }
      }
    }
  }

  // Final drain: catch any entries that became fully_completed during the
  // last iteration but weren't drained by the re-entrant path.
  if (completion_)
    completion_->drain_completions(new_queue_states_);

  // Re-fetch: pick up any packets the host submitted while we were executing
  // (e.g., barrier packets queued after a kernel dispatch). Process them
  // immediately so host signal waits see completed barriers before returning.
  for (size_t i = 0; i < hw_queues_.size(); ++i)
    fetch_from_queue(hw_queues_[i], new_queue_states_[i]);
  // Process any new non-kernel entries (barriers with total_wgs==0).
  for (size_t qi = 0; qi < hw_queues_.size(); ++qi) {
    auto &qs = new_queue_states_[qi];
    while (qs.next_dispatch_idx < qs.entries.size()) {
      auto &entry = qs.entries[qs.next_dispatch_idx];
      if (!entry.is_non_kernel())
        break;
      entry.completed_wgs = entry.total_wgs;
      ++qs.next_dispatch_idx;
    }
  }
  if (completion_)
    completion_->drain_completions(new_queue_states_);

  // Register as primary on first dispatch.
  if (!is_primary_ && pending_entries() > 0) {
    engine()->register_as_primary();
    is_primary_ = true;
  }

  // For quantum>0 mode: activate CUs via port messages.
  for (size_t i = 0; i < cus_.size(); ++i) {
    if (cus_[i]->has_active_wfs()) {
      if (dispatch_ports_[i]->link())
        dispatch_ports_[i]->send(std::make_unique<simdojo::Message>(simdojo::MessageHeader{}));
      else
        cus_[i]->activate();
    }
  }

  // Primary release check. Determine teardown need under lock, then unlock
  // before stop_doorbell_monitor() — which joins the doorbell thread that also
  // acquires hw_queue_mutex_ (avoids deadlock).
  bool do_teardown = false;
  if (completion_ && completion_->all_complete(new_queue_states_) && !has_kfd_queues()) {
    if (is_primary_)
      do_teardown = true;
  }

  lock.unlock();

  if (do_teardown) {
    stop_doorbell_monitor();
    engine()->primary_release();
    is_primary_ = false;
  }
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

  // ROCR SDMA queue pointers are in BYTE units. Convert to dword units for
  // ring buffer indexing (each SDMA packet field is a 32-bit dword).
  uint64_t rpos = read_idx / sizeof(uint32_t);
  uint64_t wpos = write_idx / sizeof(uint32_t);

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
      uint32_t count = (dw(1) & 0x3FFFFFF) + 1;
      uint64_t src = static_cast<uint64_t>(dw(3)) | (static_cast<uint64_t>(dw(4)) << 32);
      uint64_t dst = static_cast<uint64_t>(dw(5)) | (static_cast<uint64_t>(dw(6)) << 32);
      util::Logger::vm("SDMA COPY: src=", std::hex, src, " dst=", dst, std::dec, " count=", count,
                       " (", count / 1024, " KB)");
      if (dst <= 0x4d08242f10 && dst + count > 0x4d08242f00) {
        uint64_t off = 0x4d08242f08 - dst;
        if (off + 4 <= count) {
          uint32_t val = 0;
          std::memcpy(&val, reinterpret_cast<const uint8_t *>(src) + off, 4);
          util::Logger::vm("SDMA WATCHPOINT: copying ", std::hex, val, std::dec,
                           " to NaN address 0x4d08242f08");
        }
      }
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
      bool mem_poll = (header >> 31) & 1;
      bool hdp_flush = (header >> 26) & 1;
      uint32_t func = (header >> 28) & 0x7;
      uint64_t addr = static_cast<uint64_t>(dw(1)) | (static_cast<uint64_t>(dw(2)) << 32);
      uint32_t ref = dw(3);
      uint32_t mask = dw(4);
      if (!mem_poll) {
        // Register poll / HDP flush — no-op in functional sim.
      } else if (addr > 0x1000) {
        auto *ptr = reinterpret_cast<uint32_t *>(addr);
        auto compare = [func](uint32_t val, uint32_t reference) -> bool {
          switch (func) {
          case 0:
            return true;
          case 1:
            return val < reference;
          case 2:
            return val <= reference;
          case 3:
            return val == reference;
          case 4:
            return val != reference;
          case 5:
            return val >= reference;
          case 6:
            return val > reference;
          default:
            return true;
          }
        };
        uint32_t val = std::atomic_ref<uint32_t>(*ptr).load(std::memory_order_acquire);
        if (!compare(val & mask, ref)) {
          // Condition not satisfied. Stop processing this SDMA ring and
          // schedule a retry. Non-blocking to avoid deadlocking when the
          // condition depends on a compute dispatch or another queue.
          if (queue.host_accessible) {
            std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(queue.read_ptr_va))
                .store(rpos * sizeof(uint32_t), std::memory_order_release);
          }
          engine()->schedule_event_now(&doorbell_event_);
          return;
        }
      }
      (void)hdp_flush;
      pkt_dwords = sdma::POLL_REGMEM_SIZE;
      break;
    }
    case sdma::OP_ATOMIC: {
      uint64_t addr = static_cast<uint64_t>(dw(1)) | (static_cast<uint64_t>(dw(2)) << 32);
      uint64_t src_data = static_cast<uint64_t>(dw(3)) | (static_cast<uint64_t>(dw(4)) << 32);
      uint32_t atomic_op = (header >> 25) & 0x7F;
      // SDMA_ATOMIC_ADD64 = 47
      if (atomic_op == 47 && addr > 0x1000) {
        auto *ptr = reinterpret_cast<int64_t *>(addr);
        std::atomic_ref<int64_t>(*ptr).fetch_add(static_cast<int64_t>(src_data),
                                                 std::memory_order_release);
        if (static_cast<int64_t>(src_data) < 0 && interrupt_cb_) {
          uint64_t sig_base = addr - 8;
          auto mailbox_ptr = *reinterpret_cast<uint64_t *>(sig_base + 16);
          auto event_id = *reinterpret_cast<uint32_t *>(sig_base + 24);
          if (mailbox_ptr != 0) {
            std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(mailbox_ptr))
                .store(uint64_t(event_id), std::memory_order_release);
          }
          interrupt_cb_(event_id);
        }
      }
      pkt_dwords = sdma::ATOMIC_SIZE;
      break;
    }
    case sdma::OP_CONST_FILL: {
      uint64_t addr = static_cast<uint64_t>(dw(1)) | (static_cast<uint64_t>(dw(2)) << 32);
      uint32_t data = dw(3);
      uint32_t count = (dw(4) & 0x3FFFFFF) + 1;
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
    case sdma::OP_TIMESTAMP: {
      uint64_t addr = static_cast<uint64_t>(dw(1)) | (static_cast<uint64_t>(dw(2)) << 32);
      if (addr > 0x1000) {
        auto now = std::chrono::steady_clock::now().time_since_epoch();
        uint64_t ts = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
        std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(addr))
            .store(ts, std::memory_order_release);
      }
      pkt_dwords = sdma::TIMESTAMP_SIZE;
      break;
    }
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
      uint32_t count = (dw(3) & 0x3FFFFFF) + 1;
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

  if (queue.host_accessible) {
    std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(queue.read_ptr_va))
        .store(rpos * sizeof(uint32_t), std::memory_order_release);
  }
}

} // namespace amdgpu
} // namespace rocjitsu
