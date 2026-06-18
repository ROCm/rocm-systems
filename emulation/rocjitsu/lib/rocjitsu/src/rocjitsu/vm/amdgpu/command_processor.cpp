// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/code/kernel_symbol.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/amd_hsa_queue.h"
RJ_DIAGNOSTIC_POP

#include "simdojo/sim/message.h"
#include "simdojo/sim/simulation.h"
#include "util/log.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <elf.h>
#include <format>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <thread>

namespace rocjitsu {
namespace amdgpu {

namespace {

constexpr uint8_t HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH = 3;

struct AmdExtKernelDispatchPacket {
  uint16_t header;
  uint8_t amd_format;
  uint8_t setup;
  uint16_t workgroup_size_x;
  uint16_t workgroup_size_y;
  uint16_t workgroup_size_z;
  uint16_t reserved0;
  uint32_t cluster_count_x;
  uint16_t cluster_count_y;
  uint16_t cluster_count_z;
  uint8_t cluster_size_x;
  uint8_t cluster_size_y;
  uint8_t cluster_size_z;
  uint8_t perf_hint;
  uint32_t private_segment_size;
  uint32_t group_segment_size;
  uint64_t kernel_object;
  void *kernarg_address;
  hsa_signal_t dep_signal;
  hsa_signal_t completion_signal;
};

static_assert(sizeof(AmdExtKernelDispatchPacket) == 64);

uint32_t nonzero_or_one(uint32_t v) { return v == 0 ? 1 : v; }

uint32_t read_memory_u32(GpuMemory *memory, uint64_t addr) {
  uint32_t value = 0;
  for (uint32_t i = 0; i < sizeof(value); ++i)
    value |= static_cast<uint32_t>(memory->read8(addr + i)) << (i * 8);
  return value;
}

bool sgpr_count_is_descriptor_encoded(rj_code_arch_t arch, uint32_t sgpr_gran) {
  if (sgpr_gran != 0)
    return true;

  switch (arch) {
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_RDNA4:
  case ROCJITSU_CODE_ARCH_GFX1250:
    return false;
  default:
    return true;
  }
}

} // namespace

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
  int flat_scratch_init_sgpr = -1;
  if (kcp != 0) {
    uint32_t idx = 0;
    if (AMDHSA_BITS_GET(kcp, KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER)) {
      if (pkt.queue_ptr != 0) {
        uint64_t srd_va = pkt.queue_ptr + offsetof(amd_queue_t, scratch_resource_descriptor);
        cu->write_sgpr(sbase + idx + 0, read_gpu_u32(srd_va + 0, pkt.process_id));
        cu->write_sgpr(sbase + idx + 1, read_gpu_u32(srd_va + 4, pkt.process_id));
        cu->write_sgpr(sbase + idx + 2, read_gpu_u32(srd_va + 8, pkt.process_id));
        cu->write_sgpr(sbase + idx + 3, read_gpu_u32(srd_va + 12, pkt.process_id));
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
        uint64_t wdi_va = pkt.queue_ptr + offsetof(amd_queue_t, write_dispatch_id);
        dispatch_id = read_gpu_u64(wdi_va, pkt.process_id);
      }
      cu->write_sgpr(sbase + idx, static_cast<uint32_t>(dispatch_id));
      cu->write_sgpr(sbase + idx + 1, static_cast<uint32_t>(dispatch_id >> 32));
      idx += 2;
    }
    if (AMDHSA_BITS_GET(kcp, KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT)) {
      flat_scratch_init_sgpr = static_cast<int>(idx);
      idx += 2;
    }
    if (AMDHSA_BITS_GET(kcp, KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE)) {
      cu->write_sgpr(sbase + idx, pkt.private_segment_fixed_size);
      idx += 1;
    }

    uint32_t preload_length = AMDHSA_BITS_GET(pkt.kernarg_preload, KERNARG_PRELOAD_SPEC_LENGTH);
    uint32_t preload_offset = AMDHSA_BITS_GET(pkt.kernarg_preload, KERNARG_PRELOAD_SPEC_OFFSET);
    if (preload_length != 0) {
      if (pkt.kernarg_addr == 0 || memory_ == nullptr)
        throw std::runtime_error("AMDHSA kernarg preload requires a mapped kernarg segment");
      if (idx + preload_length > pkt.num_user_sgprs)
        throw std::runtime_error("AMDHSA kernarg preload exceeds declared user SGPR count");
      uint32_t preload_end = preload_offset + preload_length;
      // Some assembly code objects leave descriptor kernarg_size at zero while
      // carrying the real size in metadata; treat zero as unknown.
      if (pkt.kernarg_size != 0 && preload_end > pkt.kernarg_size / sizeof(uint32_t))
        throw std::runtime_error("AMDHSA kernarg preload exceeds kernarg segment size");

      uint64_t preload_addr = pkt.kernarg_addr + static_cast<uint64_t>(preload_offset) * 4;
      for (uint32_t i = 0; i < preload_length; ++i)
        cu->write_sgpr(sbase + idx + i, read_memory_u32(memory_, preload_addr + i * 4));
      util::Logger::vm("CP: init_wf kernarg preload s[", idx, ":", idx + preload_length - 1,
                       "] length=", preload_length, " offset=", preload_offset, " sbase=", sbase);
      idx += preload_length;
    }
  } else {
    // Legacy: kernarg at s[0:1].
    if (pkt.kernarg_addr != 0) {
      cu->write_sgpr(sbase + 0, static_cast<uint32_t>(pkt.kernarg_addr));
      cu->write_sgpr(sbase + 1, static_cast<uint32_t>(pkt.kernarg_addr >> 32));
    }
  }

  uint32_t gx = pkt.grid_wgs_x > 0 ? pkt.grid_wgs_x : 1;
  uint32_t gy = pkt.grid_wgs_y > 0 ? pkt.grid_wgs_y : 1;
  uint32_t grid_wg_id_x = global_wg_id % gx;
  uint32_t wg_id_y = (global_wg_id / gx) % gy;
  uint32_t wg_id_z = global_wg_id / (gx * gy);
  uint32_t wg_id_x = (pkt.enable_wg_id_y || pkt.enable_wg_id_z) ? grid_wg_id_x : global_wg_id;

  // System SGPRs: workgroup_id_{x,y,z} placed sequentially after user SGPRs.
  // Only the IDs whose enable bits are set in compute_pgm_rsrc2 are written.
  // When kernel_code_properties is 0 (internal test dispatches), always write
  // workgroup_id_x as a fallback since internal kernels expect it.
  uint32_t sys_idx = pkt.num_user_sgprs;
  {
    bool kcp_zero = (pkt.kernel_code_properties == 0);
    if (pkt.enable_wg_id_x || kcp_zero)
      cu->write_sgpr(sbase + sys_idx++, wg_id_x);
    if (pkt.enable_wg_id_y)
      cu->write_sgpr(sbase + sys_idx++, wg_id_y);
    if (pkt.enable_wg_id_z)
      cu->write_sgpr(sbase + sys_idx++, wg_id_z);
  }

  if (cu->arch() == ROCJITSU_CODE_ARCH_GFX1250 || cu->arch() == ROCJITSU_CODE_ARCH_RDNA4) {
    constexpr uint32_t ttmp7 = 115;
    constexpr uint32_t ttmp9 = 117;
    // The simulator aliases TTMP scalar selectors into the wavefront SGPR
    // block, so the block must include slots through TTMP9.
    if (cu->config().sgprs_per_wf <= ttmp9) {
      throw std::runtime_error("RDNA4/gfx1250 TTMP launch payload requires at least 118 SGPR "
                               "slots per wavefront");
    }
    cu->write_sgpr(sbase + ttmp7, ((wg_id_z & 0xFFFFu) << 16) | (wg_id_y & 0xFFFFu));
    cu->write_sgpr(sbase + ttmp9, grid_wg_id_x);
  }

  // Workitem IDs per AMDHSA ABI. The SPI decomposes the flat thread index
  // into (x, y, z) using the AQL packet's workgroup dimensions.
  // enable_vgpr_workitem_id (TIDIG_COMP_CNT from compute_pgm_rsrc2):
  //   0 = v0 only (workitem_id_x)
  //   1 = v0 + v1 (workitem_id_x, workitem_id_y)
  //   2 = v0 + v1 + v2 (workitem_id_x, workitem_id_y, workitem_id_z)
  // On packed-TID targets (CDNA3/4 and gfx1250): v0[9:0]=X,
  // v0[19:10]=Y, v0[29:20]=Z. v1/v2 are not written. Kernel extracts
  // components via bit masks.
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
  // that multiplied by wf_size. The global wave index is derived from
  // (global_wg_id, wf_index_in_wg) to ensure non-overlapping scratch
  // across all CUs and workgroups in the dispatch.
  if (pkt.private_segment_fixed_size > 0) {
    uint64_t scratch_pool = pkt.scratch_backing_addr;
    if (scratch_pool == 0)
      scratch_pool = 0x1'0000'0000ULL;
    uint64_t per_wave_size = static_cast<uint64_t>(pkt.private_segment_fixed_size) * cu->wf_size();
    uint32_t wg_total_size = static_cast<uint32_t>(pkt.workgroup_size_x) *
                             std::max<uint16_t>(1, pkt.workgroup_size_y) *
                             std::max<uint16_t>(1, pkt.workgroup_size_z);
    uint32_t waves_per_wg = (wg_total_size + cu->wf_size() - 1) / cu->wf_size();
    uint64_t global_wave_idx = static_cast<uint64_t>(global_wg_id) * waves_per_wg + wf_index_in_wg;
    uint64_t wave_scratch = scratch_pool + global_wave_idx * per_wave_size;

    if (memory_ && memory_->resolve_host_ptr(wave_scratch, pkt.process_id) == nullptr &&
        scratch_allocator_) {
      uint64_t total_scratch = per_wave_size * pkt.total_wgs * waves_per_wg;
      scratch_allocator_(pkt.process_id, scratch_pool, static_cast<size_t>(total_scratch));
    }

    wf->set_scratch_base(wave_scratch);
    wf->set_scratch_lane_size(pkt.private_segment_fixed_size);
    util::Logger::cp([&](auto &os) {
      os << std::format(
          "SCRATCH wf{} pool={:#x} wave_scratch={:#x} per_wave={} priv_size={} "
          "backing_addr={:#x} mapped={}",
          wf->wf_id(), scratch_pool, wave_scratch, per_wave_size, pkt.private_segment_fixed_size,
          pkt.scratch_backing_addr,
          memory_ ? (memory_->resolve_host_ptr(wave_scratch, pkt.process_id) != nullptr) : false);
    });

    if (flat_scratch_init_sgpr >= 0) {
      cu->write_sgpr(sbase + flat_scratch_init_sgpr, static_cast<uint32_t>(wave_scratch));
      cu->write_sgpr(sbase + flat_scratch_init_sgpr + 1, static_cast<uint32_t>(wave_scratch >> 32));
    }
  }
}

void CommandProcessor::startup() {
  doorbell_event_.set_handler(
      [this](simdojo::Tick ts, simdojo::Message *) { handle_doorbell(ts); });
  completion_ = std::make_unique<CompletionTracker>(memory_, cus_);
  completion_->set_plugin_group(plugin_group_);
  if (interrupt_cb_)
    completion_->set_interrupt_callback(interrupt_cb_);
}

void CommandProcessor::register_queue(HwQueue queue) {
  util::Logger::cp([&](auto &os) {
    os << std::format("{}: REGISTER_QUEUE id={} pid={} ring={:#x} size={} rptr={:#x} wptr={:#x} "
                      "doorbell_off={} is_sdma={} db_base={}",
                      name(), queue.queue_id, queue.process_id, queue.ring_base_va, queue.ring_size,
                      queue.read_ptr_va, queue.write_ptr_va, queue.doorbell_offset, queue.is_sdma,
                      reinterpret_cast<uintptr_t>(queue.doorbell_base));
  });
  bool start_poll = queue.host_accessible;
  {
    std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
    HwQueueState qs{};
    qs.queue_desc_va = queue.queue_desc_va;
    hw_queues_.push_back(std::move(queue));
    new_queue_states_.push_back(std::move(qs));
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
    util::Logger::cp([&](auto &os) { os << std::format("{}: STARTING doorbell thread", name()); });
    doorbell_thread_ = std::jthread([this](std::stop_token stop) { doorbell_poll_loop(stop); });
  }
}

void CommandProcessor::unregister_queue(uint32_t queue_id, uint32_t process_id) {
  std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
  for (size_t i = 0; i < hw_queues_.size(); ++i) {
    if (hw_queues_[i].queue_id == queue_id && hw_queues_[i].process_id == process_id) {
      hw_queues_.erase(hw_queues_.begin() + static_cast<ptrdiff_t>(i));
      new_queue_states_.erase(new_queue_states_.begin() + static_cast<ptrdiff_t>(i));
      break;
    }
  }
  // Don't stop the doorbell monitor here — the join can deadlock when
  // the poller thread is mid-iteration (holding engine or event state
  // locks). The ~CommandProcessor destructor joins the thread safely
  // after all client activity has ceased.
}

void CommandProcessor::update_queue(uint32_t queue_id, uint32_t process_id, uint64_t ring_base_va,
                                    uint32_t ring_size) {
  std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
  for (auto &q : hw_queues_) {
    if (q.queue_id == queue_id && q.process_id == process_id) {
      q.ring_base_va = ring_base_va;
      q.ring_size = ring_size;
      break;
    }
  }
}

void CommandProcessor::set_doorbell_base(uint32_t process_id, void *base) {
  std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
  for (auto &q : hw_queues_) {
    if (q.process_id == process_id)
      q.doorbell_base = base;
  }
}

void CommandProcessor::stop_doorbell_monitor() {
  if (doorbell_thread_.joinable()) {
    doorbell_thread_.request_stop();
    doorbell_thread_.join();
  }
}

uint64_t CommandProcessor::read_gpu_u64(uint64_t va, uint32_t vmid) const {
  uint64_t val = 0;
  auto *dst = reinterpret_cast<uint8_t *>(&val);
  for (uint32_t i = 0; i < sizeof(val); ++i)
    dst[i] = memory_->read8(va + i, vmid);
  return val;
}

uint32_t CommandProcessor::read_gpu_u32(uint64_t va, uint32_t vmid) const {
  uint32_t val = 0;
  auto *dst = reinterpret_cast<uint8_t *>(&val);
  for (uint32_t i = 0; i < sizeof(val); ++i)
    dst[i] = memory_->read8(va + i, vmid);
  return val;
}

void CommandProcessor::read_gpu_block(uint64_t va, void *dst, size_t size, uint32_t vmid) const {
  auto *p = static_cast<uint8_t *>(dst);
  for (size_t i = 0; i < size; ++i)
    p[i] = memory_->read8(va + i, vmid);
}

void CommandProcessor::write_gpu_block(uint64_t va, const void *src, size_t size, uint32_t vmid) {
  auto *p = static_cast<const uint8_t *>(src);
  for (size_t i = 0; i < size; ++i)
    memory_->write8(va + i, p[i], vmid);
}

/// @brief Scan all HW queues for doorbell changes; return true if any changed.
/// Caller must NOT hold hw_queue_mutex_.
bool CommandProcessor::scan_doorbells() {
  bool found = false;
  std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
  for (auto &q : hw_queues_) {
    uint64_t val;
    if (q.host_accessible) {
      if (!q.doorbell_base)
        continue;
      val = std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(
                                          static_cast<char *>(q.doorbell_base) + q.doorbell_offset))
                .load(std::memory_order_acquire);
    } else {
      if (q.doorbell_va == 0)
        continue;
      val = read_gpu_u64(q.doorbell_va, q.process_id);
    }
    if (val != q.last_doorbell) {
      util::Logger::cp([&](auto &os) {
        os << std::format("{}: DOORBELL_CHANGE pid={} qid={} sdma={} old={:#x} new={:#x} "
                          "db_base={} db_off={}",
                          name(), q.process_id, q.queue_id, q.is_sdma, q.last_doorbell, val,
                          reinterpret_cast<uintptr_t>(q.doorbell_base), q.doorbell_offset);
      });
      q.last_doorbell = val;
      found = true;
    }
  }
  return found;
}

void CommandProcessor::doorbell_poll_loop(std::stop_token stop) {
  using namespace std::chrono_literals;
  uint64_t poll_count = 0;
  while (!stop.stop_requested()) {
    if (scan_doorbells())
      engine()->schedule_event_now(&doorbell_event_);
    else
      std::this_thread::sleep_for(100us);
    ++poll_count;

    // HQD idle monitoring: periodically fire HQD_IDLE for queues that are
    // currently empty. On real hardware the CP continuously monitors queue
    // activity and fires the idle interrupt whenever the queue is inactive.
    // Our drain_completions fires on the non-empty→empty transition, but a
    // process may create a new event AFTER that transition and miss the
    // signal. Re-broadcasting every ~10ms ensures late-created events see
    // the idle state within a bounded window.
    if (poll_count % 100 == 0) {
      std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
      for (size_t i = 0; i < hw_queues_.size(); ++i) {
        if (new_queue_states_[i].entries.empty() && hw_queues_[i].process_id != 0) {
          if (interrupt_cb_)
            interrupt_cb_(hw_queues_[i].process_id, 0);
        }
      }
    }

    if (poll_count % 5000 == 1) {
      std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
      for (auto &q : hw_queues_) {
        uint64_t val = 0;
        if (q.host_accessible && q.doorbell_base) {
          val = std::atomic_ref<uint64_t>(
                    *reinterpret_cast<uint64_t *>(static_cast<char *>(q.doorbell_base) +
                                                  q.doorbell_offset))
                    .load(std::memory_order_acquire);
        }
        util::Logger::cp([&](auto &os) {
          os << std::format("{}: DOORBELL_POLL pid={} qid={} val={:#x} last={:#x} db_base={} "
                            "db_off={} polls={}",
                            name(), q.process_id, q.queue_id, val, q.last_doorbell,
                            reinterpret_cast<uintptr_t>(q.doorbell_base), q.doorbell_offset,
                            poll_count);
        });
      }
    }
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

    // SPI selects the CU based on resource availability.
    ComputeUnitCore *cu = nullptr;
    if (!spis_.empty()) {
      for (auto *spi : spis_) {
        cu = spi->dispatch_workgroup(entry);
        if (cu)
          break;
      }
    } else {
      for (size_t attempt = 0; attempt < cus_.size(); ++attempt) {
        size_t cu_idx = (next_cu_ + attempt) % cus_.size();
        cus_[cu_idx]->retire_halted_wfs();
        if (cus_[cu_idx]->can_accept_workgroup(entry.wfs_per_workgroup,
                                               entry.group_segment_fixed_size)) {
          cu = cus_[cu_idx];
          next_cu_ = (cu_idx + 1) % cus_.size();
          break;
        }
      }
    }

    if (!cu)
      break;

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
      wf->set_process_id(entry.process_id);
      init_wavefront_regs(cu, wf, entry, global_wg_id, w);
      wg_wavefronts.push_back(wf);
    }
    plugin_group_->onAmdgpuWorkgroupDispatched(entry.dispatch_id, global_wg_id,
                                               cu->vgpr_allocation_block_size(), entry.sgprs_per_wf,
                                               std::span<Wavefront *>(wg_wavefronts));
    for (auto *wf : wg_wavefronts)
      plugin_group_->onAmdgpuWavefrontDispatched(*wf);

    ++entry.dispatched_wgs;
    ++dispatched;
  }
  return dispatched;
}

// ---------------------------------------------------------------------------
// Completion notification from CU
// ---------------------------------------------------------------------------

void CommandProcessor::notify_wg_complete(uint32_t dispatch_id, uint32_t wg_id) {
  util::Logger::cp(
      [&](auto &os) { os << std::format("WG_COMPLETE d={} wg={}", dispatch_id, wg_id); });
  std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
  if (completion_)
    completion_->notify_wg_complete(dispatch_id, wg_id, new_queue_states_);
}

void CommandProcessor::on_cu_idle() {
  if (cus_.empty())
    return;

  std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);

  if (completion_)
    completion_->drain_completions(new_queue_states_);

  for (auto &qs : new_queue_states_) {
    while (qs.next_dispatch_idx < qs.entries.size()) {
      auto &e = qs.entries[qs.next_dispatch_idx];
      if (e.barrier_bit && !barrier_satisfied(qs, qs.next_dispatch_idx))
        break;
      if (!e.is_non_kernel())
        break;
      e.completed_wgs = e.total_wgs;
      ++qs.next_dispatch_idx;
    }
  }
  if (completion_)
    completion_->drain_completions(new_queue_states_);

  std::vector<bool> was_idle(cus_.size());
  for (size_t i = 0; i < cus_.size(); ++i)
    was_idle[i] = !cus_[i]->has_active_wfs();

  for (auto &qs : new_queue_states_) {
    if (qs.next_dispatch_idx < qs.entries.size()) {
      auto &entry = qs.entries[qs.next_dispatch_idx];
      if (entry.barrier_bit && !barrier_satisfied(qs, qs.next_dispatch_idx))
        continue;
      if (!entry.is_non_kernel() && !entry.fully_dispatched()) {
        if (entry.dispatched_wgs == 0)
          plugin_group_->onAmdgpuDispatchExecutionBegin(entry.dispatch_id);
        uint32_t sent = dispatch_workgroups(entry);
        if (sent > 0 && entry.fully_dispatched())
          ++qs.next_dispatch_idx;
      }
    }
  }

  for (size_t i = 0; i < cus_.size(); ++i) {
    if (was_idle[i] && cus_[i]->has_active_wfs())
      cus_[i]->activate();
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
CommandProcessor::read_kernel_descriptor(uint64_t kernel_object, uint32_t vmid,
                                         [[maybe_unused]] bool host_accessible) {
  using namespace rocr::llvm::amdhsa;
  kernel_descriptor_t kd{};
  if (memory_)
    read_gpu_block(kernel_object, &kd, sizeof(kd), vmid);
  return kd;
}

/// Scan backward from ptr to find the ELF header (\x7fELF) at a page boundary.
/// Both ptr and limit must be readable host memory.
static const uint8_t *find_elf_base(const uint8_t *ptr, const uint8_t *limit) {
  auto *page = reinterpret_cast<const uint8_t *>(reinterpret_cast<uintptr_t>(ptr) & ~0xFFFULL);
  for (; page >= limit; page -= 0x1000) {
    if (page[0] == 0x7f && page[1] == 'E' && page[2] == 'L' && page[3] == 'F')
      return page;
  }
  return nullptr;
}

void CommandProcessor::process_aql_packet(const hsa_kernel_dispatch_packet_t &pkt,
                                          const HwQueue &queue, uint64_t pkt_addr,
                                          HwQueueState &qs) {
  bool host_accessible = queue.host_accessible;
  using namespace rocr::llvm::amdhsa;
  kernel_descriptor_t kd =
      read_kernel_descriptor(pkt.kernel_object, queue.process_id, host_accessible);
  uint32_t vgpr_gran =
      AMDHSA_BITS_GET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  uint32_t sgpr_gran =
      AMDHSA_BITS_GET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
  uint32_t vgprs = (vgpr_gran + 1) * vgpr_granularity_;
  rj_code_arch_t arch = cus_.empty() ? ROCJITSU_CODE_ARCH_CDNA1 : cus_[0]->config().arch;
  uint32_t sgprs = sgpr_count_is_descriptor_encoded(arch, sgpr_gran) ? (sgpr_gran + 1) * 8 : 0;
  uint32_t user_sgprs = AMDHSA_BITS_GET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT);
  uint64_t entry_pc = pkt.kernel_object + static_cast<uint64_t>(kd.kernel_code_entry_byte_offset);

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
  dp.process_id = queue.process_id;
  dp.kernel_entry_pc = entry_pc;
  dp.total_wgs = total_wgs;
  dp.dispatched_wgs = 0;
  dp.completed_wgs = 0;
  dp.wfs_per_workgroup = wfs_per_wg;
  dp.sgprs_per_wf = sgprs > 0 ? sgprs : 104;
  dp.vgprs_per_wf = vgprs > 0 ? vgprs : 256;
  dp.kernarg_addr = reinterpret_cast<uint64_t>(pkt.kernarg_address);
  dp.kernarg_size = kd.kernarg_size;
  dp.num_user_sgprs = user_sgprs;
  dp.kernel_code_properties = kd.kernel_code_properties;
  dp.kernarg_preload = kd.kernarg_preload;
  dp.private_segment_fixed_size = std::max(kd.private_segment_fixed_size, pkt.private_segment_size);
  dp.group_segment_fixed_size = std::max(kd.group_segment_fixed_size, pkt.group_segment_size);

  // For KFD dispatches, provide pointers the kernel may need via user SGPRs.
  // The queue_ptr and dispatch_ptr are GPU VAs that the kernel reads via SMEM.
  if (host_accessible) {
    dp.dispatch_ptr = pkt_addr;
    dp.queue_ptr = queue.read_ptr_va - offsetof(amd_queue_t, read_dispatch_id);
    if (dp.private_segment_fixed_size > 0) {
      uint64_t scratch_loc_va =
          dp.queue_ptr + offsetof(amd_queue_t, scratch_backing_memory_location);
      dp.scratch_backing_addr = read_gpu_u64(scratch_loc_va, queue.process_id);
      if (dp.scratch_backing_addr == 0 && scratch_resolver_)
        dp.scratch_backing_addr = scratch_resolver_(queue.process_id);
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
  dp.host_signal = false;
  dp.barrier_bit = (pkt.header >> HSA_PACKET_HEADER_BARRIER) & 1;

  // Process AQL acquire fence: invalidate caches so the kernel sees the
  // latest host/agent writes (kernarg data, input buffers, etc.).
  // On real hardware the CP issues GL1_INV + GL2_INV for SYSTEM/AGENT scope.
  uint32_t acquire_scope = (pkt.header >> HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) & 0x3;
  if (acquire_scope >= HSA_FENCE_SCOPE_AGENT && !cus_.empty()) {
    for (auto *cu : cus_)
      cu->flush_all(queue.process_id);
  }

  std::string kernel_sym;
  if (host_accessible && memory_) {
    auto [range_base, range_size] = memory_->find_host_range(pkt.kernel_object);
    if (range_base != 0) {
      auto *ko = reinterpret_cast<const uint8_t *>(pkt.kernel_object);
      auto *range_start = reinterpret_cast<const uint8_t *>(range_base);
      auto *elf = find_elf_base(ko, range_start);
      if (elf) {
        uint64_t accessible = range_size - static_cast<uint64_t>(elf - range_start);
        kernel_sym = find_kernel_symbol(ko, elf, accessible);
      }
    }
  }
  ++total_dispatched_;

  KernelDispatchInfo dispatch_info{};
  dispatch_info.dispatch_id = dp.dispatch_id;
  dispatch_info.kernel_object = pkt.kernel_object;
  dispatch_info.entry_pc = entry_pc;
  dispatch_info.kernel_name = kernel_sym;
  dispatch_info.grid_size_x = pkt.grid_size_x;
  dispatch_info.grid_size_y = pkt.grid_size_y;
  dispatch_info.grid_size_z = pkt.grid_size_z;
  dispatch_info.workgroup_size_x = pkt.workgroup_size_x;
  dispatch_info.workgroup_size_y = pkt.workgroup_size_y;
  dispatch_info.workgroup_size_z = pkt.workgroup_size_z;
  dispatch_info.workgroup_count = total_wgs;
  dispatch_info.wfs_per_workgroup = wfs_per_wg;
  dispatch_info.sgprs_per_wf = dp.sgprs_per_wf;
  dispatch_info.vgprs_per_wf = dp.vgprs_per_wf;
  plugin_group_->onAmdgpuDispatchPacketProcessed(dispatch_info);

  util::Logger::vm([&](auto &os) {
    os << std::format("dispatch #{} d={} \"{}\" grid=[{},{},{}] wg=[{},{},{}] wgs={} "
                      "lds={} sgpr={} vgpr={} sig={:#x}",
                      total_dispatched_, dp.dispatch_id, kernel_sym.empty() ? "?" : kernel_sym,
                      pkt.grid_size_x, pkt.grid_size_y, pkt.grid_size_z, pkt.workgroup_size_x,
                      pkt.workgroup_size_y, pkt.workgroup_size_z, total_wgs,
                      kd.group_segment_fixed_size, dp.sgprs_per_wf, dp.vgprs_per_wf,
                      dp.completion_signal);
  });
  util::Logger::cp([&](auto &os) {
    os << std::format("DISPATCH #{} d={} \"{}\" wgs={} wfs/wg={} sig={:#x} pid={} ko={:#x} pc={:#x}"
                      " kernarg={:#x} user_sgprs={}",
                      total_dispatched_, dp.dispatch_id, kernel_sym.empty() ? "?" : kernel_sym,
                      total_wgs, wfs_per_wg, dp.completion_signal, dp.process_id, pkt.kernel_object,
                      entry_pc, dp.kernarg_addr, dp.num_user_sgprs);
    if (memory_) {
      auto *ko_ptr = memory_->translate_debug(pkt.kernel_object, queue.process_id);
      auto *pc_ptr = memory_->translate_debug(entry_pc, queue.process_id);
      os << std::format(" ko_mapped={} pc_mapped={} mem={:#x}", ko_ptr != nullptr,
                        pc_ptr != nullptr, reinterpret_cast<uintptr_t>(memory_));
      if (pc_ptr) {
        uint32_t first_word;
        std::memcpy(&first_word, pc_ptr + (entry_pc & 0xFFF), 4);
        os << std::format(" first_inst={:#010x}", first_word);
      }
    }
  });

  qs.entries.push_back(std::move(dp));
}

void CommandProcessor::fetch_from_queue(HwQueue &queue, HwQueueState &qs) {
  if (!memory_)
    return;
  if (queue.host_accessible ? (queue.doorbell_base == nullptr) : (queue.doorbell_va == 0))
    return;

  // Read write and read indices. For KFD queues, pointers are in host memory
  // and can be read directly. For internal test queues, they're in GpuMemory.
  uint64_t write_idx = read_gpu_u64(queue.write_ptr_va, queue.process_id);
  uint64_t read_idx = read_gpu_u64(queue.read_ptr_va, queue.process_id);
  util::Logger::vm([&](auto &os) {
    static uint64_t fetch_count = 0;
    if (write_idx != read_idx && ++fetch_count <= 50)
      os << std::format("FETCH q={} w={} r={} delta={} sdma={}", queue.queue_id, write_idx,
                        read_idx, write_idx - read_idx, queue.is_sdma);
  });

  // SDMA queues use byte-granularity pointers and have their own doorbell
  // semantics — skip the AQL doorbell clamping that assumes packet indices.
  if (queue.is_sdma) {
    if (queue.doorbell_base) {
      uint64_t db_val = std::atomic_ref<uint64_t>(
                            *reinterpret_cast<uint64_t *>(static_cast<char *>(queue.doorbell_base) +
                                                          queue.doorbell_offset))
                            .load(std::memory_order_acquire);
      if (db_val > write_idx)
        write_idx = db_val;
    }
    util::Logger::cp([&](auto &os) {
      os << std::format("{}: SDMA_FETCH pid={} qid={} read={} write={} delta={}", name(),
                        queue.process_id, queue.queue_id, read_idx, write_idx,
                        write_idx - read_idx);
    });
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
    {
      auto *dst = reinterpret_cast<uint8_t *>(&pkt);
      for (uint32_t i = 0; i < AQL_PACKET_SIZE; ++i)
        dst[i] = memory_->read8(pkt_addr + i, queue.process_id);
    }

    uint8_t pkt_type = pkt.header & 0xFF;
    util::Logger::vm([&](auto &os) {
      os << std::format("PKT q={} slot={} type={} header={:#x} read_idx={}", queue.queue_id, slot,
                        pkt_type, pkt.header, read_idx);
    });

    if (pkt_type == HSA_PACKET_TYPE_INVALID) {
      auto *host_ptr = memory_->resolve_host_ptr(pkt_addr, queue.process_id);
      uint16_t raw_header = 0;
      if (host_ptr)
        std::memcpy(&raw_header, host_ptr, sizeof(raw_header));
      util::Logger::warn("INVALID pkt at 0x", std::hex, pkt_addr, " host=0x",
                         reinterpret_cast<uintptr_t>(host_ptr), " raw=0x", raw_header);
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
        uint64_t dep_sig = read_gpu_u64(pkt_addr + DEP_OFF + dep * 8, queue.process_id);
        if (dep_sig == 0)
          continue;
        has_deps = true;
        auto v = static_cast<int64_t>(read_gpu_u64(dep_sig + SIG_VAL_OFF, queue.process_id));
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
      sig = read_gpu_u64(pkt_addr + SIG_OFF, queue.process_id);

      DispatchEntry dp{};
      dp.dispatch_id = next_dispatch_id_++;
      dp.queue_id = queue.queue_id;
      dp.process_id = queue.process_id;
      dp.total_wgs = 0;
      dp.completed_wgs = 0;
      dp.dispatched_wgs = 0;
      dp.completion_signal = sig;
      dp.host_signal = false;

      qs.entries.push_back(std::move(dp));
    } else if (pkt_type == HSA_PACKET_TYPE_VENDOR_SPECIFIC) {
      AmdExtKernelDispatchPacket ext{};
      std::memcpy(&ext, &pkt, sizeof(ext));
      if (ext.amd_format == HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH) {
        if (ext.dep_signal.handle != 0) {
          constexpr uint32_t SIG_VAL_OFF = 8;
          auto *val = reinterpret_cast<int64_t *>(ext.dep_signal.handle + SIG_VAL_OFF);
          int64_t v = std::atomic_ref<int64_t>(*val).load(std::memory_order_acquire);
          if (v != 0) {
            process_limit = read_idx;
            engine()->schedule_event_now(&doorbell_event_);
            break;
          }
        }

        hsa_kernel_dispatch_packet_t dispatch{};
        dispatch.header = ext.header;
        dispatch.setup = ext.setup;
        dispatch.workgroup_size_x = ext.workgroup_size_x;
        dispatch.workgroup_size_y = ext.workgroup_size_y;
        dispatch.workgroup_size_z = ext.workgroup_size_z;
        dispatch.grid_size_x = nonzero_or_one(ext.cluster_count_x) *
                               nonzero_or_one(ext.cluster_size_x) *
                               nonzero_or_one(ext.workgroup_size_x);
        dispatch.grid_size_y = nonzero_or_one(ext.cluster_count_y) *
                               nonzero_or_one(ext.cluster_size_y) *
                               nonzero_or_one(ext.workgroup_size_y);
        dispatch.grid_size_z = nonzero_or_one(ext.cluster_count_z) *
                               nonzero_or_one(ext.cluster_size_z) *
                               nonzero_or_one(ext.workgroup_size_z);
        dispatch.private_segment_size = ext.private_segment_size;
        dispatch.group_segment_size = ext.group_segment_size;
        dispatch.kernel_object = ext.kernel_object;
        dispatch.kernarg_address = ext.kernarg_address;
        dispatch.completion_signal = ext.completion_signal;
        process_aql_packet(dispatch, queue, pkt_addr, qs);
      } else {
        constexpr uint32_t SIG_OFF = 56;
        uint64_t sig = 0;
        sig = read_gpu_u64(pkt_addr + SIG_OFF, queue.process_id);

        DispatchEntry dp{};
        dp.dispatch_id = next_dispatch_id_++;
        dp.queue_id = queue.queue_id;
        dp.process_id = queue.process_id;
        dp.total_wgs = 0;
        dp.completed_wgs = 0;
        dp.dispatched_wgs = 0;
        dp.completion_signal = sig;
        dp.host_signal = false;

        qs.entries.push_back(std::move(dp));
      }
    }

    ++read_idx;
  }

  {
    auto *src = reinterpret_cast<const uint8_t *>(&process_limit);
    for (uint32_t i = 0; i < sizeof(process_limit); ++i)
      memory_->write8(queue.read_ptr_va + i, src[i], queue.process_id);
  }
}

void CommandProcessor::fetch_packets() {
  std::lock_guard<std::recursive_mutex> lock(hw_queue_mutex_);
  for (size_t i = 0; i < hw_queues_.size(); ++i)
    fetch_from_queue(hw_queues_[i], new_queue_states_[i]);
}

void CommandProcessor::handle_doorbell(simdojo::Tick) {
  util::Logger::cp(
      [&](auto &os) { os << std::format("{}: DOORBELL queues={}", name(), hw_queues_.size()); });

  std::unique_lock<std::recursive_mutex> lock(hw_queue_mutex_);

  size_t entries_before = 0;
  for (auto &qs : new_queue_states_)
    entries_before += qs.entries.size();

  // Fetch packets (uses last_doorbell values set by the poll thread).
  for (size_t i = 0; i < hw_queues_.size(); ++i)
    fetch_from_queue(hw_queues_[i], new_queue_states_[i]);

  // Ensure interrupt callback is set on completion tracker.
  if (completion_ && interrupt_cb_)
    completion_->set_interrupt_callback(interrupt_cb_);

  size_t entries_after = 0;
  for (auto &qs : new_queue_states_)
    entries_after += qs.entries.size();
  util::Logger::cp([&](auto &os) {
    os << std::format("{}: FETCHED {} new entries (total={})", name(),
                      entries_after - entries_before, entries_after);
  });

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
        if (entry.dispatched_wgs == 0)
          plugin_group_->onAmdgpuDispatchExecutionBegin(dispatch_id);
        bool backpressure = false;
        for (;;) {
          if (qs.next_dispatch_idx >= qs.entries.size())
            break;
          auto &cur = qs.entries[qs.next_dispatch_idx];
          if (cur.dispatch_id != dispatch_id)
            break;

          uint32_t sent = dispatch_workgroups(cur);
          if (sent > 0)
            progress = true;

          if (completion_)
            completion_->drain_completions(new_queue_states_);

          if (qs.next_dispatch_idx >= qs.entries.size())
            break;
          auto &post = qs.entries[qs.next_dispatch_idx];
          if (post.dispatch_id != dispatch_id)
            break;

          if (post.fully_dispatched()) {
            ++qs.next_dispatch_idx;
            break;
          }
          if (sent == 0) {
            backpressure = true;
            break;
          }
        }
        if (backpressure)
          break;
      }
    }
  }

  // Final drain: catch any entries that became fully_completed during the
  // last iteration but weren't drained by the re-entrant path.
  if (completion_)
    completion_->drain_completions(new_queue_states_);

  util::Logger::cp([&](auto &os) {
    size_t remaining = 0;
    for (auto &qs : new_queue_states_)
      remaining += qs.entries.size();
    uint32_t active_cus = 0;
    for (auto *cu : cus_)
      if (cu->has_active_wfs())
        ++active_cus;
    os << std::format("{}: PHASE1_DONE remaining={} active_cus={}/{}", name(), remaining,
                      active_cus, cus_.size());
  });

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

  for (size_t i = 0; i < cus_.size(); ++i) {
    if (cus_[i]->has_active_wfs()) {
      if (dispatch_ports_[i]->link())
        dispatch_ports_[i]->send(std::make_unique<simdojo::Message>(simdojo::MessageHeader{}));
      else
        cus_[i]->activate();
    }
  }

  bool do_teardown = false;
  bool all_done = completion_ && completion_->all_complete(new_queue_states_);
  bool kfd = has_kfd_queues();
  if (all_done && !kfd) {
    if (is_primary_)
      do_teardown = true;
  }
  util::Logger::cp([&](auto &os) {
    os << std::format("{}: TEARDOWN_CHECK all_done={} kfd={} primary={} teardown={}", name(),
                      all_done, kfd, is_primary_, do_teardown);
  });

  lock.unlock();

  if (do_teardown) {
    util::Logger::cp([&](auto &os) {
      os << std::format("{}: STOPPING doorbell monitor + primary_release", name());
    });
    stop_doorbell_monitor();
    engine()->primary_release();
    is_primary_ = false;
  }
}

// ---------------------------------------------------------------------------
// SDMA packet processor
// ---------------------------------------------------------------------------

/// @brief Resolve a GPU VA to a daemon-accessible host pointer.
/// @details In daemon mode, the GPU VA belongs to the client process and cannot
/// be dereferenced directly. The VMID page table maps gpu_va -> daemon_host_ptr,
/// so we use GpuMemory::resolve_host_ptr() to find the correct address. In local
/// mode, the GPU VA IS the host VA (identity mapping), so we cast directly.
/// @returns Host pointer, or nullptr if the VA is not mapped.
static void *resolve_sdma_ptr(GpuMemory *memory, uint64_t va, uint32_t vmid) {
  if (!memory)
    return nullptr;
  auto *page_base = memory->resolve_host_ptr(va, vmid);
  if (!page_base)
    return nullptr;
  return page_base + (va & 0xFFF);
}

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

constexpr uint8_t SUBOP_COPY_LINEAR = 0;
constexpr uint8_t SUBOP_FENCE_64B = 2;
constexpr uint8_t SUBOP_POLL_MEM_64B = 5;

// Packet sizes in dwords.
constexpr uint32_t COPY_LINEAR_SIZE = 7;
constexpr uint32_t FENCE_SIZE = 4;
constexpr uint32_t TRAP_SIZE = 2;
constexpr uint32_t POLL_REGMEM_SIZE = 6;
constexpr uint32_t ATOMIC_SIZE = 8;
constexpr uint32_t CONST_FILL_SIZE = 5;
constexpr uint32_t TIMESTAMP_SIZE = 3;
constexpr uint32_t GCR_SIZE = 5;
constexpr uint32_t GCR_GFX1250_SIZE = 6;
constexpr uint32_t COPY_LINEAR_WAITSIGNAL_GFX1250_SIZE = 19;
constexpr uint32_t FENCE_64B_GFX1250_SIZE = 5;
constexpr uint32_t POLL_MEM_64B_GFX1250_SIZE = 8;
// NOP_BASE_SIZE intentionally omitted — NOP is handled inline.
} // namespace sdma

namespace {

bool sdma_compare_u64(uint32_t func, uint64_t value, uint64_t reference) {
  switch (func) {
  case 0:
    return true;
  case 1:
    return value < reference;
  case 2:
    return value <= reference;
  case 3:
    return value == reference;
  case 4:
    return value != reference;
  case 5:
    return value >= reference;
  case 6:
    return value > reference;
  default:
    return true;
  }
}

} // namespace

void CommandProcessor::process_sdma_ring(HwQueue &queue, uint64_t read_idx, uint64_t write_idx) {
  uint32_t ring_mask = (queue.ring_size / sizeof(uint32_t)) - 1;

  uint64_t rpos = read_idx / sizeof(uint32_t);
  uint64_t wpos = write_idx / sizeof(uint32_t);

  auto dw = [&](uint64_t off) -> uint32_t {
    uint64_t addr = queue.ring_base_va + (((rpos + off) & ring_mask) * sizeof(uint32_t));
    return memory_->read32(addr, queue.process_id);
  };

  // Helper: resolve a GPU VA from an SDMA packet to a host pointer.
  // In daemon mode, the VA belongs to the client process; we go through the
  // VMID page table. In local mode, the VA IS the host address.
  auto resolve = [&](uint64_t va) -> void * {
    return resolve_sdma_ptr(memory_, va, queue.process_id);
  };

  while (rpos < wpos) {
    uint32_t header = dw(0);
    uint8_t op = header & 0xFF;
    uint8_t sub_op = (header >> 8) & 0xFF;
    uint32_t pkt_dwords = 0;

    switch (op) {
    case sdma::OP_NOP: {
      uint32_t count = (dw(0) >> 16) & 0x3FFF;
      pkt_dwords = 1 + count;
      break;
    }
    case sdma::OP_COPY: {
      if (uses_gfx1250_sdma_packets() && sub_op == sdma::SUBOP_COPY_LINEAR &&
          (header & ((1u << 30) | (1u << 31)))) {
        if (rpos + sdma::COPY_LINEAR_WAITSIGNAL_GFX1250_SIZE > wpos) {
          rpos = wpos;
          continue;
        }

        constexpr uint32_t COPY_BASE = 8;
        constexpr uint32_t SIGNAL_BASE = 14;
        bool has_wait = (header & (1u << 30)) != 0;
        bool has_signal = (header & (1u << 31)) != 0;

        if (has_wait) {
          uint32_t wait_func = dw(1) & 0x7;
          uint64_t wait_addr =
              (static_cast<uint64_t>(dw(2) & ~0x7u)) | (static_cast<uint64_t>(dw(3)) << 32);
          uint64_t wait_ref = static_cast<uint64_t>(dw(4)) | (static_cast<uint64_t>(dw(5)) << 32);
          uint64_t wait_mask = static_cast<uint64_t>(dw(6)) | (static_cast<uint64_t>(dw(7)) << 32);
          if (wait_addr > 0x1000) {
            uint64_t wait_value =
                std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(wait_addr))
                    .load(std::memory_order_acquire);
            if (!sdma_compare_u64(wait_func, wait_value & wait_mask, wait_ref)) {
              if (queue.host_accessible) {
                std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(queue.read_ptr_va))
                    .store(rpos * sizeof(uint32_t), std::memory_order_release);
              }
              engine()->schedule_event_now(&doorbell_event_);
              return;
            }
          }
        }

        uint32_t count = (dw(COPY_BASE) & 0x3FFFFFFF) + 1;
        uint64_t src = static_cast<uint64_t>(dw(COPY_BASE + 2)) |
                       (static_cast<uint64_t>(dw(COPY_BASE + 3)) << 32);
        uint64_t dst = static_cast<uint64_t>(dw(COPY_BASE + 4)) |
                       (static_cast<uint64_t>(dw(COPY_BASE + 5)) << 32);

        std::memcpy(reinterpret_cast<void *>(dst), reinterpret_cast<const void *>(src), count);

        for (auto *l2 : l2_caches_)
          l2->invalidate_range(dst, count);
        for (auto *cu : cus_)
          cu->l1_vector().invalidate_all();

        if (has_signal) {
          uint32_t signal_op = dw(SIGNAL_BASE) & 0x7F;
          uint64_t signal_addr = (static_cast<uint64_t>(dw(SIGNAL_BASE + 1) & ~0x7u)) |
                                 (static_cast<uint64_t>(dw(SIGNAL_BASE + 2)) << 32);
          uint64_t signal_data = static_cast<uint64_t>(dw(SIGNAL_BASE + 3)) |
                                 (static_cast<uint64_t>(dw(SIGNAL_BASE + 4)) << 32);

          if (signal_addr > 0x1000 && signal_op == 0x70) {
            std::atomic_ref<int64_t>(*reinterpret_cast<int64_t *>(signal_addr))
                .fetch_sub(static_cast<int64_t>(signal_data), std::memory_order_release);
          }
        }

        pkt_dwords = sdma::COPY_LINEAR_WAITSIGNAL_GFX1250_SIZE;
        break;
      }

      if (rpos + sdma::COPY_LINEAR_SIZE > wpos) {
        rpos = wpos;
        continue;
      }
      uint32_t count = (dw(1) & 0x3FFFFFF) + 1;
      uint64_t src_va = static_cast<uint64_t>(dw(3)) | (static_cast<uint64_t>(dw(4)) << 32);
      uint64_t dst_va = static_cast<uint64_t>(dw(5)) | (static_cast<uint64_t>(dw(6)) << 32);
      util::Logger::vm("SDMA COPY: src=", std::hex, src_va, " dst=", dst_va, std::dec,
                       " count=", count, " (", count / 1024, " KB)");
      auto *src_ptr = resolve(src_va);
      auto *dst_ptr = resolve(dst_va);
      if (src_ptr && dst_ptr) {
        if (header & (1u << 28)) {
          uint64_t dst2_va = static_cast<uint64_t>(dw(7)) | (static_cast<uint64_t>(dw(8)) << 32);
          auto *dst2_ptr = resolve(dst2_va);
          std::memcpy(dst_ptr, src_ptr, count);
          for (auto *l2 : l2_caches_)
            l2->invalidate_range(dst_va, count);
          if (dst2_ptr) {
            std::memcpy(dst2_ptr, src_ptr, count);
            for (auto *l2 : l2_caches_)
              l2->invalidate_range(dst2_va, count);
          }
          pkt_dwords = 9;
        } else {
          std::memcpy(dst_ptr, src_ptr, count);
          for (auto *l2 : l2_caches_)
            l2->invalidate_range(dst_va, count);
          pkt_dwords = sdma::COPY_LINEAR_SIZE;
        }
      } else {
        pkt_dwords = (header & (1u << 28)) ? 9 : sdma::COPY_LINEAR_SIZE;
      }
      break;
    }
    case sdma::OP_FENCE: {
      if (uses_gfx1250_sdma_packets() && sub_op == sdma::SUBOP_FENCE_64B) {
        if (rpos + sdma::FENCE_64B_GFX1250_SIZE > wpos) {
          rpos = wpos;
          continue;
        }

        uint64_t addr_va =
            static_cast<uint64_t>(dw(1) & ~0x7u) | (static_cast<uint64_t>(dw(2)) << 32);
        uint64_t data = static_cast<uint64_t>(dw(3)) | (static_cast<uint64_t>(dw(4)) << 32);
        auto *ptr = static_cast<uint64_t *>(resolve(addr_va));
        if (ptr) {
          std::atomic_ref<uint64_t>(*ptr).store(data, std::memory_order_release);
          for (auto *l2 : l2_caches_)
            l2->invalidate_range(addr_va, sizeof(uint64_t));
        }
        pkt_dwords = sdma::FENCE_64B_GFX1250_SIZE;
        break;
      }

      uint64_t addr_va = static_cast<uint64_t>(dw(1)) | (static_cast<uint64_t>(dw(2)) << 32);
      uint32_t data = dw(3);
      auto *ptr = static_cast<uint32_t *>(resolve(addr_va));
      if (ptr) {
        std::atomic_ref<uint32_t>(*ptr).store(data, std::memory_order_release);
        for (auto *l2 : l2_caches_)
          l2->invalidate_range(addr_va, sizeof(uint32_t));
      }
      pkt_dwords = sdma::FENCE_SIZE;
      break;
    }
    case sdma::OP_TRAP: {
      uint32_t event_id = dw(1) & 0x0FFFFFFF;
      if (interrupt_cb_)
        interrupt_cb_(queue.process_id, event_id);
      pkt_dwords = sdma::TRAP_SIZE;
      break;
    }
    case sdma::OP_POLL_REGMEM: {
      if (uses_gfx1250_sdma_packets() && sub_op == sdma::SUBOP_POLL_MEM_64B) {
        if (rpos + sdma::POLL_MEM_64B_GFX1250_SIZE > wpos) {
          rpos = wpos;
          continue;
        }

        uint32_t func = (header >> 28) & 0x7;
        uint64_t addr = static_cast<uint64_t>(dw(1) & ~0x7u) | (static_cast<uint64_t>(dw(2)) << 32);
        uint64_t ref = static_cast<uint64_t>(dw(3)) | (static_cast<uint64_t>(dw(4)) << 32);
        uint64_t mask = static_cast<uint64_t>(dw(5)) | (static_cast<uint64_t>(dw(6)) << 32);
        if (addr > 0x1000) {
          auto *ptr = reinterpret_cast<uint64_t *>(addr);
          uint64_t val = std::atomic_ref<uint64_t>(*ptr).load(std::memory_order_acquire);
          if (!sdma_compare_u64(func, val & mask, ref)) {
            if (queue.host_accessible) {
              std::atomic_ref<uint64_t>(*reinterpret_cast<uint64_t *>(queue.read_ptr_va))
                  .store(rpos * sizeof(uint32_t), std::memory_order_release);
            }
            engine()->schedule_event_now(&doorbell_event_);
            return;
          }
        }
        pkt_dwords = sdma::POLL_MEM_64B_GFX1250_SIZE;
        break;
      }

      bool mem_poll = (header >> 31) & 1;
      bool hdp_flush = (header >> 26) & 1;
      uint32_t func = (header >> 28) & 0x7;
      uint64_t addr_va = static_cast<uint64_t>(dw(1)) | (static_cast<uint64_t>(dw(2)) << 32);
      uint32_t ref = dw(3);
      uint32_t mask = dw(4);
      if (!mem_poll) {
        // Register poll / HDP flush — no-op in functional sim.
      } else if (addr_va > 0x1000) {
        auto *ptr = static_cast<uint32_t *>(resolve(addr_va));
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
        uint32_t val = ptr ? std::atomic_ref<uint32_t>(*ptr).load(std::memory_order_acquire) : 0;
        if (!compare(val & mask, ref)) {
          {
            uint64_t rptr_val = rpos * sizeof(uint32_t);
            auto *rp_src = reinterpret_cast<const uint8_t *>(&rptr_val);
            for (uint32_t b = 0; b < sizeof(rptr_val); ++b)
              memory_->write8(queue.read_ptr_va + b, rp_src[b], queue.process_id);
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
      uint64_t addr_va = static_cast<uint64_t>(dw(1)) | (static_cast<uint64_t>(dw(2)) << 32);
      uint64_t src_data = static_cast<uint64_t>(dw(3)) | (static_cast<uint64_t>(dw(4)) << 32);
      uint32_t atomic_op = (header >> 25) & 0x7F;
      // SDMA_ATOMIC_ADD64 = 47
      if (atomic_op == 47 && addr_va > 0x1000) {
        auto *ptr = static_cast<int64_t *>(resolve(addr_va));
        if (ptr) {
          std::atomic_ref<int64_t>(*ptr).fetch_add(static_cast<int64_t>(src_data),
                                                   std::memory_order_release);
          for (auto *l2 : l2_caches_)
            l2->invalidate_range(addr_va, sizeof(int64_t));
          if (static_cast<int64_t>(src_data) < 0 && interrupt_cb_) {
            // Signal layout: addr is at offset 8 (value field) from sig base.
            uint64_t sig_base = addr_va - 8;
            auto *mb = static_cast<uint64_t *>(resolve(sig_base + 16));
            auto *eid = static_cast<uint32_t *>(resolve(sig_base + 24));
            uint64_t mailbox_ptr = mb ? *mb : 0;
            uint32_t event_id = eid ? *eid : 0;
            if (mailbox_ptr != 0) {
              auto *mb_dst = static_cast<uint64_t *>(resolve(mailbox_ptr));
              if (mb_dst) {
                std::atomic_ref<uint64_t>(*mb_dst).store(uint64_t(event_id),
                                                         std::memory_order_release);
                for (auto *l2 : l2_caches_)
                  l2->invalidate_range(mailbox_ptr, sizeof(uint64_t));
              }
            }
            interrupt_cb_(queue.process_id, event_id);
          }
        }
      }
      pkt_dwords = sdma::ATOMIC_SIZE;
      break;
    }
    case sdma::OP_CONST_FILL: {
      uint64_t addr_va = static_cast<uint64_t>(dw(1)) | (static_cast<uint64_t>(dw(2)) << 32);
      uint32_t data = dw(3);
      uint32_t count = (dw(4) & 0x3FFFFFF) + 1;
      uint32_t fillsize = (header >> 30) & 0x3;
      auto *dst = static_cast<uint8_t *>(resolve(addr_va));
      if (dst) {
        if (fillsize == 2) {
          for (uint32_t i = 0; i < count; i += 4)
            std::memcpy(dst + i, &data, 4);
        } else {
          std::memset(dst, static_cast<int>(data & 0xFF), count);
        }
        for (auto *l2 : l2_caches_)
          l2->invalidate_range(addr_va, count);
      }
      pkt_dwords = sdma::CONST_FILL_SIZE;
      break;
    }
    case sdma::OP_TIMESTAMP: {
      uint64_t addr_va = static_cast<uint64_t>(dw(1)) | (static_cast<uint64_t>(dw(2)) << 32);
      if (addr_va > 0x1000) {
        auto now = std::chrono::steady_clock::now().time_since_epoch();
        uint64_t ts = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
        auto *ptr = static_cast<uint64_t *>(resolve(addr_va));
        if (ptr)
          std::atomic_ref<uint64_t>(*ptr).store(ts, std::memory_order_release);
      }
      pkt_dwords = sdma::TIMESTAMP_SIZE;
      break;
    }
    case sdma::OP_GCR: {
      uint64_t base_va =
          (static_cast<uint64_t>(dw(1)) | (static_cast<uint64_t>(dw(2)) << 32)) & ~0xFULL;
      uint64_t size_field =
          (static_cast<uint64_t>(dw(3)) | (static_cast<uint64_t>(dw(4)) << 32)) & ~0xFULL;
      uint32_t range =
          size_field > 0 ? static_cast<uint32_t>(std::min(size_field, uint64_t(UINT32_MAX))) : 0;
      for (auto *l2 : l2_caches_) {
        if (range > 0)
          l2->invalidate_range(base_va, range);
        else
          l2->invalidate_all();
      }
      pkt_dwords = uses_gfx1250_sdma_packets() ? sdma::GCR_GFX1250_SIZE : sdma::GCR_SIZE;
      break;
    }
    case sdma::OP_HDP_FLUSH:
      pkt_dwords = 1;
      break;
    case sdma::OP_WRITE: {
      if (rpos + 4 > wpos) {
        rpos = wpos;
        continue;
      }
      uint32_t count = (dw(3) & 0x3FFFFFF) + 1;
      uint64_t addr_va = static_cast<uint64_t>(dw(1)) | (static_cast<uint64_t>(dw(2)) << 32);
      if (addr_va > 0x1000 && rpos + 4 + count <= wpos) {
        auto *dst = static_cast<uint32_t *>(resolve(addr_va));
        if (dst) {
          for (uint32_t i = 0; i < count; ++i)
            dst[i] = dw(4 + i);
          for (auto *l2 : l2_caches_)
            l2->invalidate_range(addr_va, count * sizeof(uint32_t));
        }
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

  {
    uint64_t rptr_val = rpos * sizeof(uint32_t);
    auto *src = reinterpret_cast<const uint8_t *>(&rptr_val);
    for (uint32_t b = 0; b < sizeof(rptr_val); ++b)
      memory_->write8(queue.read_ptr_va + b, src[b], queue.process_id);
  }
}

} // namespace amdgpu
} // namespace rocjitsu
