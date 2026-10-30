// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/isa/arch/amdgpu/shared/ds_transpose.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/isa/arch/amdgpu/shared/tensor_dma.h"
#include "rocjitsu/isa/isa_traits.h"
#include "rocjitsu/vm/amdgpu/buffer_format.h"
#include "rocjitsu/vm/amdgpu/cluster_lds_multicast.h"
#include "rocjitsu/vm/amdgpu/coalesced_lane_runs.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/l1_scalar_cache.h"
#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/lds_barrier_cell.h"
#include "rocjitsu/vm/amdgpu/lds_stack.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/plugins/memory_access_observation.h"
#include "util/log.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

MemoryPipeline::~MemoryPipeline() {
  while (!issued_.empty()) {
    delete issued_.front().inst;
    issued_.pop();
  }
  while (!returned_.empty()) {
    delete returned_.front().inst;
    returned_.pop();
  }
}

MemoryPipeline::WaitCounterTokens MemoryPipeline::issue_counters(const Instruction &inst) const {
  WaitCounterTokens counters;
  const DynamicInstState *state = inst.data();
  const auto *issue = inst.amdgpu_memory_issue_info();
  if (state && (state->tag() == GLOBAL_MEM || state->tag() == LOCAL_MEM)) {
    const auto &routed = inst.data_as<VectorMemState>()->routed_issue_info;
    if (routed)
      issue = &*routed;
  }
  if (issue) {
    for (const auto obligation : issue->counter_obligations()) {
      for (uint8_t token = 0; token < obligation.counter_increment(); ++token)
        counters.types[counters.size++] = obligation.wait_counter_type();
    }
    return counters;
  }

  WaitCounterType counter = counter_type_;
  if (state != nullptr) {
    switch (state->tag()) {
    case SCALAR_MEM:
      counter = inst.data_as<ScalarMemState>()->wait_counter_type;
      break;
    case GLOBAL_MEM:
    case LOCAL_MEM:
      counter = inst.data_as<VectorMemState>()->wait_counter_type;
      break;
    default:
      break;
    }
  }
  counters.types[counters.size++] = counter;
  return counters;
}

void MemoryPipeline::acquire_wait_counters(Wavefront &wf, const WaitCounterTokens &counters) {
  for (uint8_t i = 0; i < counters.size; ++i)
    wf.wait_counters().increment(counters.types[i]);
}

void MemoryPipeline::release_wait_counters(Wavefront &wf, const WaitCounterTokens &counters) {
  for (uint8_t i = 0; i < counters.size; ++i)
    wf.release_wait_counter(counters.types[i]);
}

void MemoryPipeline::complete_entry(PipelineEntry entry) {
  MemoryAccessDeferredCompletion deferred_completion = [this, inst = entry.inst, wf = entry.wf,
                                                        counters = entry.counters,
                                                        wave_generation = entry.wave_generation]() {
    finish_completed_access(inst, *wf, counters, wave_generation);
  };
  const MemoryAccessCompletion completion =
      complete_access(*entry.inst, *entry.wf, std::move(deferred_completion));
  if (completion == MemoryAccessCompletion::Complete)
    finish_completed_access(entry.inst, *entry.wf, entry.counters, entry.wave_generation);
}

VmAccessOutcome MemoryPipeline::issue_impl(Instruction *inst, Wavefront &wf,
                                           bool retain_unavailable) {
  const WaitCounterTokens counters = issue_counters(*inst);
  acquire_wait_counters(wf, counters);
  const VmAccessOutcome access = initiate_access(*inst, wf);
  if (access == VmAccessOutcome::Unavailable && retain_unavailable) {
    issued_.push({.inst = inst,
                  .wf = &wf,
                  .counters = counters,
                  .wave_generation = wf.dispatch_generation(),
                  .issue_pc = wf.pc});
    wf.set_state(WfState::VM_RETRY);
    wf.cu().request_functional_yield();
    return VmAccessOutcome::Complete;
  }
  if (access != VmAccessOutcome::Complete) {
    if (access == VmAccessOutcome::Faulted) {
      // An immediate fault can be retained by the same exception owner as a
      // retry fault. Discard its diagnostic producer before losing the instruction.
      [[maybe_unused]] const bool cancelled =
          cancel_wait_check_issue({.inst = inst,
                                   .wf = &wf,
                                   .counters = counters,
                                   .wave_generation = wf.dispatch_generation(),
                                   .issue_pc = wf.pc});
      assert(cancelled && "Faulted memory issue has no matching wait-check producer");
    }
    release_wait_counters(wf, counters);
    delete inst;
    return access;
  }
  complete_entry({.inst = inst,
                  .wf = &wf,
                  .counters = counters,
                  .wave_generation = wf.dispatch_generation(),
                  .issue_pc = wf.pc});
  return VmAccessOutcome::Complete;
}

VmAccessOutcome MemoryPipeline::issue(Instruction *inst, Wavefront &wf) {
  return issue_impl(inst, wf, false);
}

VmAccessOutcome MemoryPipeline::issue_deferred(Instruction *inst, Wavefront &wf) {
  return issue_impl(inst, wf, true);
}

void MemoryPipeline::defer_unavailable(Instruction *inst, Wavefront &wf) {
  const WaitCounterTokens counters = issue_counters(*inst);
  acquire_wait_counters(wf, counters);
  issued_.push({.inst = inst,
                .wf = &wf,
                .counters = counters,
                .wave_generation = wf.dispatch_generation(),
                .issue_pc = wf.pc});
  wf.set_state(WfState::VM_RETRY);
  wf.cu().request_functional_yield();
}

bool MemoryPipeline::rewind_unavailable_load(Wavefront &wf) {
  // VM_RETRY prevents further instruction issue, so the one retained load is
  // the next architectural instruction. Never replay multiple operations whose
  // ordering or side effects cannot be recovered from this queue alone.
  if (wf.state() != WfState::VM_RETRY)
    return false;
  Instruction *load = nullptr;
  PipelineEntry replay{};
  for (auto entries = issued_; !entries.empty(); entries.pop()) {
    const auto &entry = entries.front();
    if (entry.wf != &wf || entry.wave_generation != wf.dispatch_generation())
      continue;
    if (load)
      return false;
    const auto *state = entry.inst->data();
    if (state->tag() == SCALAR_MEM) {
      const auto &scalar = *entry.inst->data_as<ScalarMemState>();
      if (!scalar.is_load || scalar.atomic_op != AtomicOp::NONE)
        return false;
    } else if (state->tag() == GLOBAL_MEM) {
      const auto &vector = *entry.inst->data_as<VectorMemState>();
      if (!vector.is_load || vector.atomic_op != AtomicOp::NONE || vector.lds_dst)
        return false;
    } else {
      return false;
    }
    load = entry.inst;
    replay = entry;
  }
  if (!load)
    return false;
  if (!cancel_wait_check_issue(replay))
    return false;
  const size_t pending = issued_.size();
  for (size_t i = 0; i < pending; ++i) {
    auto entry = issued_.front();
    issued_.pop();
    if (entry.inst != load) {
      issued_.push(entry);
      continue;
    }
    wf.pc = entry.issue_pc;
    release_wait_counters(wf, entry.counters);
    delete entry.inst;
    wf.set_state(WfState::RUNNING);
  }
  return true;
}

bool MemoryPipeline::cancel_wait_check_issue(const PipelineEntry &entry) {
  auto &wf = *entry.wf;
  if (auto *scoreboard = wf.memory_wait_scoreboard();
      scoreboard && entry.inst->is_memory_wait_producer()) {
    bool xcnt = false;
    // An unavailable load with replay-source checking has an outstanding X
    // position. MODE gates VMEM groups; scalar loads always use their X group.
    if (scoreboard->outstanding(WaitCounterKind::X)) {
      const auto events =
          waitcheck_detail::WaitcheckTarget::classify_events(*entry.inst, wf.cu().arch());
      if (events.failed())
        return false;
      const auto xevent = std::ranges::find(events.value(), WaitCounterKind::X,
                                            &waitcheck_detail::ClassifiedEvent::counter);
      xcnt =
          xevent != events.value().end() &&
          (xevent->kind == waitcheck_detail::WaitEventKind::Smem || (wf.mode_raw() & (1u << 25)));
    }
    if (!scoreboard->cancel_last_issue(entry.issue_pc,
                                       std::span(entry.counters.types).first(entry.counters.size),
                                       wf.cu().arch(), xcnt))
      return false;
  }
  return true;
}

void MemoryPipeline::cancel(Wavefront &wf) {
  std::queue<PipelineEntry> retained;
  while (!issued_.empty()) {
    PipelineEntry entry = issued_.front();
    issued_.pop();
    if (entry.wf == &wf && entry.wave_generation == wf.dispatch_generation()) {
      delete entry.inst;
      continue;
    }
    retained.push(entry);
  }
  issued_.swap(retained);
}

void MemoryPipeline::tick() { retry_entries(true, 0, 0); }

void MemoryPipeline::retry_for_queue(uint32_t process_id, uint32_t queue_id) {
  retry_entries(false, process_id, queue_id);
}

void MemoryPipeline::retry_entries(bool all_queues, uint32_t process_id, uint32_t queue_id) {
  const std::size_t pending = issued_.size();
  for (std::size_t index = 0; index < pending && !issued_.empty(); ++index) {
    PipelineEntry entry = issued_.front();
    issued_.pop();
    if (!all_queues && (entry.wf->process_id() != process_id || entry.wf->queue_id() != queue_id)) {
      issued_.push(entry);
      continue;
    }
    if (entry.wf->dispatch_generation() != entry.wave_generation) {
      delete entry.inst;
      continue;
    }
    const VmAccessOutcome access = initiate_access(*entry.inst, *entry.wf);
    if (access == VmAccessOutcome::Unavailable) {
      issued_.push(entry);
      if (all_queues)
        entry.wf->cu().request_functional_yield();
      continue;
    }
    if (access == VmAccessOutcome::Complete) {
      complete_entry(entry);
      continue;
    }

    // The retained instruction no longer owns a retry or a diagnostic producer.
    // An exception owner may keep this wave for replay; settle both before the
    // callback can suspend it, enter a trap handler, or cancel its dispatch.
    [[maybe_unused]] const bool cancelled = cancel_wait_check_issue(entry);
    assert(cancelled && "Deferred memory fault has no matching wait-check producer");
    release_wait_counters(*entry.wf, entry.counters);
    if (entry.wf->state() == WfState::VM_RETRY)
      entry.wf->set_state(WfState::RUNNING);
    delete entry.inst;
    if (fault_handler_)
      fault_handler_(*entry.wf, access, entry.issue_pc);
    if (all_queues)
      return;
    // Snapshot service must settle the other paused waves even when a retained
    // wave faults. The handler may cancel its dispatch and remove later entries;
    // the bounded pass rechecks the live queue before accessing the next one.
  }
}

namespace {

uint32_t extend_scalar_load(const uint8_t *bytes, uint32_t elem_size, bool sign_extend) {
  uint32_t value = 0;
  std::memcpy(&value, bytes, elem_size);
  if (!sign_extend)
    return value;
  if (elem_size == 1)
    return static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(value & 0xffu)));
  if (elem_size == 2)
    return static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(value & 0xffffu)));
  return value;
}

std::vector<ClusterLdsTarget> resolve_lds_write_targets(VectorMemState &d, Wavefront &wf,
                                                        ComputeUnitCore &cu) {
  std::vector<ClusterLdsTarget> targets;
  if (d.cluster_multicast && d.cluster_mcast_mask != 0) {
    if (auto *cp = cu.command_processor())
      targets = cp->cluster_lds_targets(wf.dispatch_id(), wf.wg_id(), d.cluster_mcast_mask);
  }

  const bool writes_self =
      !d.cluster_multicast || d.cluster_mcast_mask == 0 ||
      (d.cluster_mcast_mask & cluster_multicast_rank_mask(wf.cluster_rank())) != 0;
  if (targets.empty() && writes_self)
    targets.push_back({&cu, wf.wg_id(), d.lds_base, wf.cluster_rank()});
  return targets;
}

void write_lds_dst_load_direct(const VectorMemState &d, Lds &lds, uint32_t per_lane_bytes,
                               uint64_t write_mask) {
  for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
    if ((write_mask & (1ULL << lane)) == 0)
      continue;
    uint32_t data_offset = lane * per_lane_bytes;
    if (data_offset + per_lane_bytes > d.response_data.size()) {
      throw std::runtime_error(std::format(
          "LDS-destination load payload too small: lane={} offset={} bytes={} payload={}", lane,
          data_offset, per_lane_bytes, d.response_data.size()));
    }
    const uint32_t output_bytes = d.lds_dword_extend ? d.num_elems * 4 : per_lane_bytes;
    uint32_t lds_addr =
        d.lds_per_lane_addr ? d.per_lane_lds_addr[lane] : d.lds_base + lane * output_bytes;
    if (lds_addr == kInvalidLdsAddress)
      continue;
    if (d.lds_dword_extend && d.elem_size < 4) {
      assert(d.num_elems == 1);
      uint32_t value = 0;
      std::memcpy(&value, &d.response_data[data_offset], d.elem_size);
      if (d.sign_extend) {
        value = d.elem_size == 1
                    ? static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(value)))
                    : static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(value)));
      }
      lds.write32(lds_addr, value);
    } else {
      lds.write(lds_addr, &d.response_data[data_offset], per_lane_bytes);
    }
  }
}

MemoryAccessCompletion complete_lds_dst_load(VectorMemState &d, Wavefront &wf, ComputeUnitCore &cu,
                                             MemoryAccessDeferredCompletion complete) {
  uint32_t per_lane_bytes = d.num_elems * d.elem_size;
  std::vector<ClusterLdsTarget> targets;
  size_t target_count = 1;
  const bool cluster_downgrades_to_ordinary = d.cluster_multicast && wf.cluster_size() <= 1;
  if (d.cluster_multicast && !cluster_downgrades_to_ordinary) {
    targets = resolve_lds_write_targets(d, wf, cu);
    target_count = targets.size();
  }

  // Per-lane LDS-dst buffer load trace.
  util::Logger::vm([&](auto &os) {
    static thread_local uint64_t lds_dst_trace = 0;
    if (++lds_dst_trace > 80)
      return;
    os << std::format("{} wg[{}] wf[{}] BUF->LDS: lds_base={:#x} plb={} mcast={:#x} targets={}",
                      cu.full_path(), d.wg_id, d.wf_id, d.lds_base, per_lane_bytes,
                      d.cluster_mcast_mask, target_count);
    for (uint32_t ln = 0; ln < d.wf_size; ++ln) {
      if (!(d.lane_mask & (1ULL << ln)))
        continue;
      uint32_t v = 0;
      if (per_lane_bytes >= 4)
        std::memcpy(&v, &d.response_data[ln * per_lane_bytes], 4);
      uint32_t lds_addr =
          d.lds_per_lane_addr ? d.per_lane_lds_addr[ln] : d.lds_base + ln * per_lane_bytes;
      if (lds_addr == kInvalidLdsAddress) {
        os << std::format(" L{}:@{:#x}->lds[dropped]", ln, d.per_lane_addr[ln]);
        continue;
      }
      os << std::format(" L{}:@{:#x}->lds[{:#x}]={:#x}", ln, d.per_lane_addr[ln], lds_addr, v);
    }
  });

  if (!d.cluster_multicast || cluster_downgrades_to_ordinary) {
    const uint64_t write_mask = lds_destination_write_mask(d, cu.arch());
    write_lds_dst_load_direct(d, wf.lds(), per_lane_bytes, write_mask);
    return MemoryAccessCompletion::Complete;
  }

  auto txn = make_cluster_lds_multicast_transaction(d, wf, std::move(targets));
  auto result = cu.cluster_lds_multicast_engine().submit(std::move(txn), std::move(complete));

  return result == ClusterLdsMulticastResult::Deferred ? MemoryAccessCompletion::Deferred
                                                       : MemoryAccessCompletion::Complete;
}

MemoryAccessCompletion vector_complete(VectorMemState &d, Wavefront &wf, ComputeUnitCore &cu,
                                       MemoryAccessDeferredCompletion complete) {
  if (!d.is_load)
    return MemoryAccessCompletion::Complete;

  if (d.buffer_components) {
    complete_buffer_format_load(wf, cu, d);
    return MemoryAccessCompletion::Complete;
  }

  // Buffer load with LDS bit: scatter loaded data into LDS instead of VGPRs.
  // Each lane writes num_elems * elem_size bytes to LDS at lds_base + lane_offset.
  if (d.lds_dst)
    return complete_lds_dst_load(d, wf, cu, std::move(complete));

  // Atomics: response layout is [lane * elem_size], regular loads are
  // [lane * (num_elems * elem_size) + elem * elem_size].
  bool is_atomic = (d.atomic_op != AtomicOp::NONE);
  uint32_t stride = is_atomic ? d.elem_size : d.num_elems * d.elem_size;
  uint32_t vgpr_count = d.destination_vgpr_count();
  if (!d.is_block_transfer && !cu.owns_vgpr_range(wf, d.dst_reg_base, vgpr_count))
    return MemoryAccessCompletion::Complete;
  const auto writes_dword = [&](uint32_t word) {
    return d.block_dword_enabled(word) &&
           (!d.is_block_transfer || cu.owns_vgpr_range(wf, d.dst_reg_base + word, 1));
  };

  // Zero destination VGPRs for OOB lanes. Per AMD ISA spec, out-of-bounds
  // buffer loads return 0. exec_mask is the effective issue mask; ordinary OOB
  // accesses retain it while architecturally ignored resource types clear it.
  // lane_mask has OOB lanes removed. The difference gives issue-active OOB lanes.
  uint64_t exec = d.exec_mask;
  uint64_t oob_mask = exec & ~d.lane_mask; // exec-active but OOB
  if (oob_mask) {
    util::Logger::vm([&](auto &os) {
      static uint64_t oob_count = 0;
      if (++oob_count > 10)
        return;
      os << cu.full_path() << " wg[" << d.wg_id << "] wf[" << d.wf_id
         << "] OOB zeroing: exec=" << std::hex << d.exec_mask << " lane=" << d.lane_mask
         << " oob=" << oob_mask << std::dec << " dst=" << d.dst_reg_base << " vgprs=" << vgpr_count;
    });
    for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
      if (!(oob_mask & (1ULL << lane)))
        continue;
      for (uint32_t i = 0; i < vgpr_count; ++i) {
        if (!writes_dword(i))
          continue;
        uint32_t val = 0;
        if (!cu.sram_ecc() && d.elem_size <= 2 && (d.d16_hi || d.d16_lo)) {
          const uint32_t old = cu.read_vgpr_storage(d.dst_reg_base + i, lane);
          val = d.d16_hi ? (old & 0x0000FFFFu) : (old & 0xFFFF0000u);
        }
        cu.write_vgpr(d.dst_reg_base + i, lane, val);
      }
    }
  }
  // Performance-only fast path for ordinary dword loads. The general completion
  // path below remains authoritative for atomics, conversions, and partial-register
  // writes. Ownership is validated above for ordinary loads and per DWORD for
  // blocks, so this path can use one raw pointer per written register. Memory completion is VM
  // bookkeeping and deliberately has no instruction-side observation to preserve.
  if (!is_atomic && d.elem_size == sizeof(uint32_t) && !d.sign_extend && !d.d16_hi && !d.d16_lo) {
    for (uint32_t i = 0; i < vgpr_count; ++i) {
      if (!writes_dword(i))
        continue;
      auto *destination = reinterpret_cast<uint32_t *>(cu.raw_vgpr_data(d.dst_reg_base + i));
      uint64_t lanes = d.lane_mask;
      while (lanes) {
        const uint32_t lane = std::countr_zero(lanes);
        lanes &= lanes - 1;
        std::memcpy(&destination[lane], &d.response_data[lane * stride + i * sizeof(uint32_t)],
                    sizeof(uint32_t));
      }
    }
    return MemoryAccessCompletion::Complete;
  }
  for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
    if (!(d.lane_mask & (1ULL << lane)))
      continue;
    for (uint32_t i = 0; i < vgpr_count; ++i) {
      if (!writes_dword(i))
        continue;
      uint32_t val = 0;
      uint32_t data_offset = lane * stride + i * 4;
      uint32_t copy_size =
          is_atomic ? std::min(d.elem_size - i * 4, 4u) : std::min(d.elem_size, 4u);
      // Constant sizes let the compiler inline common response-word copies.
      if (copy_size == 4)
        std::memcpy(&val, &d.response_data[data_offset], 4);
      else if (copy_size == 2)
        std::memcpy(&val, &d.response_data[data_offset], 2);
      else
        std::memcpy(&val, &d.response_data[data_offset], copy_size);
      if (d.sign_extend && i == 0 && d.elem_size < 4) {
        if (d.elem_size == 1)
          val = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(val)));
        else if (d.elem_size == 2)
          val = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(val)));
      }
      if (copy_size <= 2 && (d.d16_hi || d.d16_lo)) {
        if (cu.sram_ecc()) {
          if (d.d16_hi)
            val = val << 16;
          else
            val = val & 0xFFFF;
        } else {
          uint32_t old = cu.read_vgpr_storage(d.dst_reg_base + i, lane);
          if (d.d16_hi)
            val = (old & 0xFFFF) | (val << 16);
          else
            val = (old & 0xFFFF0000) | (val & 0xFFFF);
        }
      }
      cu.write_vgpr(d.dst_reg_base + i, lane, val);
    }
  }
  return MemoryAccessCompletion::Complete;
}

/// @brief Read data operand @p index of a scalar atomic as one T-wide value.
/// @details T is the atomic width. Operand 0 is the source; compare-swap's
/// compare value is operand 1.
template <typename T> T scalar_atomic_operand(const ScalarMemState &d, uint32_t index) {
  T value;
  std::memcpy(&value, d.store_data + index * (sizeof(T) / sizeof(uint32_t)), sizeof(value));
  return value;
}

/// @brief Compute the memory value a scalar atomic stores over @p old.
/// @details Operations other than compare-swap ignore the compare value.
template <typename T> T scalar_atomic_result(const ScalarMemState &d, T old) {
  return apply_int_atomic(d.atomic_op, old, scalar_atomic_operand<T>(d, 0),
                          scalar_atomic_operand<T>(d, 1));
}

template <typename T> VmAccessOutcome execute_translated_scalar_atomic_rmw(ScalarMemState &d) {
  assert(d.translated.access.has_value());
  if (!d.translated.atomic_loaded) {
    const AtomicLoadResult loaded = d.translated.access->atomic_load(d.addr, sizeof(T));
    if (loaded.outcome != VmAccessOutcome::Complete)
      return loaded.outcome;
    d.translated.atomic_loaded_value = loaded.value;
    d.translated.atomic_loaded = true;
  }

  // Retain the load across unavailable CAS attempts, as for vector atomics.
  while (true) {
    const T old = static_cast<T>(d.translated.atomic_loaded_value);
    // A failed compare-swap returns the loaded value without writing, as for
    // vector atomics.
    if (d.atomic_op == AtomicOp::CMPSWAP && old != scalar_atomic_operand<T>(d, 1)) {
      std::memcpy(d.response_data, &old, sizeof(old));
      d.translated.atomic_loaded = false;
      return VmAccessOutcome::Complete;
    }
    const T value = scalar_atomic_result(d, old);
    const AtomicCompareExchangeResult exchanged =
        d.translated.access->compare_exchange(d.addr, sizeof(T), old, value);
    if (exchanged.outcome != VmAccessOutcome::Complete)
      return exchanged.outcome;
    if (exchanged.exchanged) {
      std::memcpy(d.response_data, &old, sizeof(old));
      d.translated.atomic_loaded = false;
      return VmAccessOutcome::Complete;
    }
    d.translated.atomic_loaded_value = exchanged.observed;
  }
}

template <typename T>
VmAccessOutcome execute_cached_scalar_atomic_rmw(ScalarMemState &d, L2Cache &l2, uint32_t vmid) {
  return l2.atomic_rmw(
      d.addr, sizeof(T),
      [&](uint8_t *line, uint32_t offset) {
        T old;
        std::memcpy(&old, line + offset, sizeof(old));
        const T value = scalar_atomic_result(d, old);
        std::memcpy(line + offset, &value, sizeof(value));
        std::memcpy(d.response_data, &old, sizeof(old));
      },
      vmid);
}

} // namespace

VmAccessOutcome ScalarMemPipeline::initiate_access(Instruction &inst, Wavefront &wf) {
  auto &d = *inst.data_as<ScalarMemState>();
  // A scalar atomic accesses one 4- or 8-byte element and returns it as dwords.
  if (d.atomic_op != AtomicOp::NONE &&
      ((d.elem_size != sizeof(uint32_t) && d.elem_size != sizeof(uint64_t)) ||
       d.num_dwords * sizeof(uint32_t) != d.elem_size))
    return VmAccessOutcome::Malformed;
  // Masked loads still write the zero response, without taking a VM snapshot or
  // touching backing memory, including byte and halfword requests.
  // Fully masked stores likewise perform no VM or backing-memory access.
  if (d.load_dword_mask == 0)
    return VmAccessOutcome::Complete;
  if (wf.address_space()) {
    const bool may_refresh_unready =
        d.translated.initialized && d.translated.access && !d.translated.access->info().ready &&
        d.translated.completed_bytes == 0 && d.translated.request_index == 0;
    if (!d.translated.initialized || may_refresh_unready) {
      d.translated.access = wf.snapshot_vm_access();
      if (!d.translated.access)
        return VmAccessOutcome::Faulted;
      d.translated.initialized = true;
    }
    if (!d.translated.access->info().legacy_cache_compatible) {
      if (d.atomic_op != AtomicOp::NONE)
        return d.elem_size == sizeof(uint32_t) ? execute_translated_scalar_atomic_rmw<uint32_t>(d)
                                               : execute_translated_scalar_atomic_rmw<uint64_t>(d);
      if (d.is_load) {
        if (d.elem_size >= 4 && d.load_dword_mask != 0xffff) {
          while (d.translated.request_index < d.num_dwords) {
            const auto index = d.translated.request_index;
            if (d.load_dword_mask & (1u << index)) {
              const auto outcome = d.translated.access->read(
                  d.addr + index * sizeof(uint32_t),
                  {reinterpret_cast<std::byte *>(&d.response_data[index]), sizeof(uint32_t)},
                  d.translated.completed_bytes);
              if (outcome != VmAccessOutcome::Complete)
                return outcome;
            }
            ++d.translated.request_index;
            d.translated.completed_bytes = 0;
          }
        } else if (d.elem_size < 4) {
          uint8_t *bytes = reinterpret_cast<uint8_t *>(d.response_data);
          const VmAccessOutcome outcome = d.translated.access->read(
              d.addr, std::span<std::byte>(reinterpret_cast<std::byte *>(bytes), d.elem_size),
              d.translated.completed_bytes);
          if (outcome != VmAccessOutcome::Complete)
            return outcome;
          d.response_data[0] = extend_scalar_load(bytes, d.elem_size, d.sign_extend);
        } else {
          const VmAccessOutcome outcome = d.translated.access->read(
              d.addr,
              std::span<std::byte>(reinterpret_cast<std::byte *>(d.response_data),
                                   d.num_dwords * sizeof(uint32_t)),
              d.translated.completed_bytes);
          if (outcome != VmAccessOutcome::Complete)
            return outcome;
        }
      } else {
        if (d.load_dword_mask != 0xffff) {
          while (d.translated.request_index < d.num_dwords) {
            const auto index = d.translated.request_index;
            if (d.load_dword_mask & (1u << index)) {
              const auto outcome = d.translated.access->write(
                  d.addr + index * sizeof(uint32_t),
                  {reinterpret_cast<const std::byte *>(&d.store_data[index]), sizeof(uint32_t)},
                  d.translated.completed_bytes);
              if (outcome != VmAccessOutcome::Complete)
                return outcome;
            }
            ++d.translated.request_index;
            d.translated.completed_bytes = 0;
          }
          return VmAccessOutcome::Complete;
        }
        const VmAccessOutcome outcome = d.translated.access->write(
            d.addr,
            std::span<const std::byte>(reinterpret_cast<const std::byte *>(d.store_data),
                                       d.num_dwords * sizeof(uint32_t)),
            d.translated.completed_bytes);
        if (outcome != VmAccessOutcome::Complete)
          return outcome;
      }
      return VmAccessOutcome::Complete;
    }
  }

  if (d.atomic_op != AtomicOp::NONE) {
    // GpuVmAccess rejects and reports scalar atomics not aligned to their access
    // width. VMID 0 reaches physical memory without it, so reject here to match; a
    // misaligned X2 atomic was observed to fault without updating memory on gfx90a.
    if (wf.process_id() == 0 && d.addr % d.elem_size != 0)
      return VmAccessOutcome::Malformed;
    L2Cache &l2 = *wf.raw_cu().l2();
    return d.elem_size == sizeof(uint32_t)
               ? execute_cached_scalar_atomic_rmw<uint32_t>(d, l2, wf.process_id())
               : execute_cached_scalar_atomic_rmw<uint64_t>(d, l2, wf.process_id());
  }
  if (d.is_load) {
    if (d.load_dword_mask == 0) {
      // Descriptor bounds suppress memory access while preserving zero writeback.
    } else if (d.elem_size < 4) {
      uint8_t bytes[4] = {};
      const VmAccessOutcome outcome =
          l1_->load_bytes(d.addr, d.elem_size, bytes, wf.process_id(), d.mtype);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      d.response_data[0] = extend_scalar_load(bytes, d.elem_size, d.sign_extend);
    } else if (d.load_dword_mask != 0xffff) {
      for (uint32_t i = 0; i < d.num_dwords; ++i)
        if (d.load_dword_mask & (1u << i)) {
          const auto outcome = l1_->load(d.addr + i * 4, 1, &d.response_data[i], wf.process_id(),
                                         /*allow_private_batch=*/false, d.mtype);
          if (outcome != VmAccessOutcome::Complete)
            return outcome;
        }
    } else {
      auto &cu = wf.raw_cu();
      const bool allow_private_batch =
          GpuVmAccessBatchGuard::active() && !cu.debug_active() && cu.plugin_group().empty();
      const VmAccessOutcome outcome = l1_->load(d.addr, d.num_dwords, d.response_data,
                                                wf.process_id(), allow_private_batch, d.mtype);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
    }
  } else {
    if (d.load_dword_mask != 0xffff) {
      while (d.translated.request_index < d.num_dwords) {
        const auto index = d.translated.request_index;
        if (d.load_dword_mask & (1u << index)) {
          const auto outcome =
              l1_->store(d.addr + index * 4, 1, &d.store_data[index], wf.process_id(), d.mtype);
          if (outcome != VmAccessOutcome::Complete)
            return outcome;
        }
        ++d.translated.request_index;
      }
      return VmAccessOutcome::Complete;
    }
    const VmAccessOutcome outcome =
        l1_->store(d.addr, d.num_dwords, d.store_data, wf.process_id(), d.mtype);
    if (outcome != VmAccessOutcome::Complete)
      return outcome;
  }
  return VmAccessOutcome::Complete;
}

MemoryAccessCompletion
ScalarMemPipeline::complete_access(Instruction &inst, Wavefront &wf,
                                   MemoryAccessDeferredCompletion /*complete*/) {
  auto &d = *inst.data_as<ScalarMemState>();
  if (!d.is_load)
    return MemoryAccessCompletion::Complete;
  assert(d.dst_register.width == d.num_dwords);
  RegisterAccess registers(wf);
  for (uint32_t i = 0; i < d.num_dwords; ++i)
    registers.write_scalar_unobserved(d.dst_register, i, d.response_data[i]);
  // Trace: log SMEM load values for debugging.
  util::Logger::vm([&](auto &os) {
    if (wf.wg_id() == 0) {
      static thread_local uint32_t slw_count = 0;
      if (++slw_count <= 100) {
        os << std::format("SMEM complete: addr={:#x} dst={} ndw={} data=[{:#x}", d.addr,
                          d.dst_register.index, d.num_dwords, d.response_data[0]);
        for (uint32_t i = 1; i < d.num_dwords && i < 4; ++i)
          os << std::format(",{:#x}", d.response_data[i]);
        os << std::format("] wg={}", wf.wg_id());
      }
    }
  });
  return MemoryAccessCompletion::Complete;
}

namespace {

// One mapping selects both scalar-FP dispatch and its evaluator operation.
// Packed 16-bit additions have their own evaluator and are excluded.
std::optional<fp_mode::ScalarAtomicOp> scalar_fp_atomic_operation(AtomicOp op) {
  switch (op) {
  case AtomicOp::FADD:
    return fp_mode::ScalarAtomicOp::FADD;
  case AtomicOp::FMIN:
    return fp_mode::ScalarAtomicOp::FMIN;
  case AtomicOp::FMAX:
    return fp_mode::ScalarAtomicOp::FMAX;
  case AtomicOp::FCMPSWAP:
    return fp_mode::ScalarAtomicOp::FCMPSWAP;
  default:
    return std::nullopt;
  }
}

// Translate the deferred request into the shared ISA floating-point contract.
template <typename Bits>
Bits apply_fp_atomic(const VectorMemState &state, fp_mode::ScalarAtomicOp operation, Bits old_bits,
                     Bits source_bits, Bits compare_bits, uint32_t denorm_mode,
                     bool local = false) {
  return fp_mode::atomic_scalar(operation, old_bits, source_bits, compare_bits, denorm_mode,
                                state.atomic_legacy_minmax, local || state.atomic_source_nan_first);
}

bool is_packed_add(AtomicOp op) {
  return op == AtomicOp::PK_ADD_F16 || op == AtomicOp::PK_ADD_BF16;
}

uint32_t atomic_source_stride(const VectorMemState &d, const std::vector<uint8_t> &store_data) {
  const bool uses_two_sources = atomic_uses_two_sources(d.atomic_op);
  const uint32_t fallback = uses_two_sources ? d.elem_size * 2 : d.elem_size;
  if (d.wf_size == 0 || store_data.empty())
    return fallback;

  const size_t per_lane = store_data.size() / d.wf_size;
  return per_lane == 0 ? fallback : static_cast<uint32_t>(per_lane);
}

bool all_elements_use_lane_mask(std::span<const uint64_t> element_lane_masks, uint64_t lane_mask,
                                uint32_t num_elems) {
  if (element_lane_masks.empty())
    return true;
  assert(element_lane_masks.size() == num_elems);
  (void)num_elems;
  return std::ranges::all_of(element_lane_masks,
                             [lane_mask](uint64_t mask) { return mask == lane_mask; });
}

uint64_t fully_valid_lane_mask(std::span<const uint64_t> element_lane_masks, uint64_t lane_mask) {
  uint64_t full_lane_mask = lane_mask;
  for (uint64_t element_mask : element_lane_masks)
    full_lane_mask &= element_mask;
  return full_lane_mask;
}

void append_translated_requests(VectorMemState &d, uint64_t lane_mask, uint32_t addr_stride,
                                uint32_t addr_base_offset, MemoryTransactionPolicy transactions) {
  const uint32_t stride = d.num_elems * d.elem_size;
  auto append = [&](uint64_t address, uint32_t data_offset, uint32_t size) {
    // Preserve transport batching for the compatibility/default model. A
    // backend may observe a larger span together; forcing a DMA per 128 bytes
    // would add transport round trips without strengthening its contract.
    if (transactions.bytes() == MemoryTransactionPolicy{}.bytes()) {
      d.translated.requests.push_back(
          {.address = address, .data_offset = data_offset, .size = size});
      return;
    }
    while (size) {
      const uint32_t chunk = transactions.chunk_size(address, size);
      d.translated.requests.push_back(
          {.address = address, .data_offset = data_offset, .size = chunk});
      address += chunk;
      data_offset += chunk;
      size -= chunk;
    }
  };

  auto append_run = [&](const CoalescedLaneRun &run) {
    if (run.lane_ordered) {
      append(run.address, run.first_lane * stride, run.size);
      return true;
    }
    const uint32_t offset = d.translated.reordered_data.size();
    d.translated.reordered_data.resize(offset + run.size);
    d.translated.requests.push_back({.address = run.address,
                                     .data_offset = offset,
                                     .size = run.size,
                                     .reordered_lane_mask = run.lane_mask});
    if (!d.is_load)
      gather_lane_run(run.address, run.lane_mask, run.size, d.per_lane_addr.data(),
                      d.store_data.data(), stride, d.translated.reordered_data.data() + offset);
    return true;
  };

  if (addr_stride != 0) {
    assert((d.scratch_swizzle_unit == 4 || d.scratch_swizzle_unit == 16) &&
           addr_base_offset < d.scratch_swizzle_unit);
    for (uint32_t elem = 0; elem < d.num_elems; ++elem) {
      uint64_t mask =
          d.element_lane_masks.empty() ? lane_mask : (d.element_lane_masks[elem] & lane_mask);
      while (mask) {
        const uint32_t lane = std::countr_zero(mask);
        mask &= mask - 1;
        const uint64_t base = d.per_lane_addr[lane];
        const uint32_t first_byte_in_unit =
            static_cast<uint32_t>((base - addr_base_offset) % d.scratch_swizzle_unit);
        uint32_t copied = elem * d.elem_size;
        const uint32_t elem_end = copied + d.elem_size;
        while (copied < elem_end) {
          const uint32_t logical_byte = first_byte_in_unit + copied;
          const uint32_t byte_in_unit = logical_byte % d.scratch_swizzle_unit;
          const uint32_t chunk = std::min(elem_end - copied, d.scratch_swizzle_unit - byte_in_unit);
          const uint64_t address = base - first_byte_in_unit +
                                   logical_byte / d.scratch_swizzle_unit * addr_stride +
                                   byte_in_unit;
          append(address, lane * stride + copied, chunk);
          copied += chunk;
        }
      }
    }
    return;
  }

  if (!all_elements_use_lane_mask(d.element_lane_masks.view(), lane_mask, d.num_elems)) {
    const uint64_t full_lane_mask = fully_valid_lane_mask(d.element_lane_masks.view(), lane_mask);
    for_each_coalesced_lane_run(d.per_lane_addr.data(), full_lane_mask, d.wf_size, stride,
                                transactions, append_run);
    for (uint32_t elem = 0; elem < d.num_elems; ++elem) {
      uint64_t mask = d.element_lane_masks[elem] & lane_mask & ~full_lane_mask;
      while (mask) {
        const uint32_t lane = std::countr_zero(mask);
        mask &= mask - 1;
        append(d.per_lane_addr[lane] + static_cast<uint64_t>(elem) * d.elem_size,
               lane * stride + elem * d.elem_size, d.elem_size);
      }
    }
    return;
  }

  for_each_coalesced_lane_run(d.per_lane_addr.data(), lane_mask, d.wf_size, stride, transactions,
                              append_run);
}

VmAccessOutcome execute_translated_transfer(VectorMemState &d) {
  assert(d.translated.access.has_value());
  while (d.translated.request_index < d.translated.requests.size()) {
    const TranslatedMemoryRequest &request = d.translated.requests[d.translated.request_index];
    VmAccessOutcome outcome = VmAccessOutcome::Malformed;
    uint8_t *data = request.reordered_lane_mask ? d.translated.reordered_data.data()
                    : d.is_load                 ? d.response_data.data()
                                                : d.store_data.data();
    if (d.is_load) {
      outcome = d.translated.access->read(
          request.address,
          std::span<std::byte>(reinterpret_cast<std::byte *>(data + request.data_offset),
                               request.size),
          d.translated.completed_bytes);
    } else {
      outcome = d.translated.access->write(
          request.address,
          std::span<const std::byte>(
              reinterpret_cast<const std::byte *>(data + request.data_offset), request.size),
          d.translated.completed_bytes);
    }
    if (outcome != VmAccessOutcome::Complete)
      return outcome;
    if (d.is_load && request.reordered_lane_mask) {
      const uint32_t stride = d.num_elems * d.elem_size;
      scatter_lane_run(request.address, request.reordered_lane_mask, request.size,
                       d.per_lane_addr.data(), data + request.data_offset, stride,
                       d.response_data.data());
    }
    ++d.translated.request_index;
    d.translated.completed_bytes = 0;
  }
  return VmAccessOutcome::Complete;
}

VmAccessOutcome execute_translated_atomic_rmw(VectorMemState &d, uint64_t lane_mask) {
  if (d.atomic_op == AtomicOp::APPEND || d.atomic_op == AtomicOp::CONSUME ||
      d.atomic_op == AtomicOp::BARRIER_ARRIVE) {
    return VmAccessOutcome::Malformed;
  }

  const uint32_t width = d.elem_size;
  d.response_data.resize(d.wf_size * width);
  const bool uses_two_sources = atomic_uses_two_sources(d.atomic_op);
  const uint32_t source_stride = atomic_source_stride(d, d.store_data);
  const std::optional<fp_mode::ScalarAtomicOp> fp_operation =
      scalar_fp_atomic_operation(d.atomic_op);

  assert(d.translated.access.has_value());
  for (uint32_t lane = d.translated.atomic_lane; lane < d.wf_size; ++lane) {
    if (!(lane_mask & (uint64_t{1} << lane)))
      continue;

    if (!d.translated.atomic_loaded) {
      const AtomicLoadResult loaded =
          d.translated.access->atomic_load(d.per_lane_addr[lane], width);
      if (loaded.outcome != VmAccessOutcome::Complete) {
        d.translated.atomic_lane = lane;
        return loaded.outcome;
      }
      d.translated.atomic_loaded_value = loaded.value;
      d.translated.atomic_loaded = true;
    }

    while (true) {
      const uint64_t old_value = d.translated.atomic_loaded_value;
      uint64_t new_value = old_value;
      uint64_t compare_value = 0;
      if (width == sizeof(uint32_t)) {
        const uint32_t old32 = static_cast<uint32_t>(old_value);
        uint32_t source = 0;
        uint32_t compare = 0;
        std::memcpy(&source, &d.store_data[lane * source_stride], sizeof(source));
        if (uses_two_sources)
          std::memcpy(&compare, &d.store_data[lane * source_stride + sizeof(source)],
                      sizeof(compare));
        compare_value = compare;
        new_value =
            is_packed_add(d.atomic_op)
                ? fp_mode::atomic_add_packed_16(old32, source, d.atomic_op == AtomicOp::PK_ADD_BF16,
                                                3, d.atomic_source_nan_first)
            : fp_operation
                ? apply_fp_atomic(d, *fp_operation, old32, source, compare, d.atomic_denorm_mode)
                : apply_int_atomic(d.atomic_op, old32, source, compare);
      } else if (width == sizeof(uint64_t)) {
        uint64_t source = 0;
        uint64_t compare = 0;
        std::memcpy(&source, &d.store_data[lane * source_stride], sizeof(source));
        if (uses_two_sources)
          std::memcpy(&compare, &d.store_data[lane * source_stride + sizeof(source)],
                      sizeof(compare));
        compare_value = compare;
        new_value = fp_operation ? apply_fp_atomic(d, *fp_operation, old_value, source, compare,
                                                   d.atomic_denorm_mode)
                                 : apply_int_atomic(d.atomic_op, old_value, source, compare);
      } else {
        return VmAccessOutcome::Malformed;
      }

      if (d.atomic_op == AtomicOp::CMPSWAP && old_value != compare_value) {
        std::memcpy(&d.response_data[lane * width], &old_value, width);
        break;
      }

      const AtomicCompareExchangeResult exchanged =
          d.translated.access->compare_exchange(d.per_lane_addr[lane], width, old_value, new_value);
      if (exchanged.outcome != VmAccessOutcome::Complete) {
        d.translated.atomic_lane = lane;
        return exchanged.outcome;
      }
      if (exchanged.exchanged) {
        std::memcpy(&d.response_data[lane * width], &old_value, width);
        break;
      }
      d.translated.atomic_loaded_value = exchanged.observed;
    }
    d.translated.atomic_loaded = false;
    d.translated.atomic_lane = lane + 1;
  }
  return VmAccessOutcome::Complete;
}

/// @brief Perform a per-lane atomic RMW through L2.
///
/// Reads the old value through L2's backing-memory atomic path, applies the
/// operation, and writes the new value back. The device coherence epoch makes
/// cached L1/L2 lines stale at the atomic boundary. Old values are stored in
/// response_data for GLC return.
VmAccessOutcome execute_atomic_rmw(VectorMemState &d, L2Cache *l2, uint32_t vmid,
                                   uint64_t lane_mask) {
  const uint32_t esz = d.elem_size;
  d.response_data.resize(d.wf_size * esz);
  const bool uses_two_sources = atomic_uses_two_sources(d.atomic_op);
  const uint32_t src_stride = atomic_source_stride(d, d.store_data);
  const std::optional<fp_mode::ScalarAtomicOp> fp_operation =
      scalar_fp_atomic_operation(d.atomic_op);

  if (lane_mask == 0)
    return VmAccessOutcome::Complete;
  // A retry keeps completed lanes but must not reuse a retired VM generation.
  const auto vm_access = l2->snapshot_atomic_access(vmid);
  DeviceCacheCoherence::AtomicBoundary boundary = l2->coherence_domain()->acquire_atomic_boundary();
  if (boundary.outcome() != VmAccessOutcome::Complete)
    return boundary.outcome();

  // Reuse the vector's cache-coherence boundary; each backing update remains atomic.
  uint32_t next_lane = d.translated.atomic_lane;
  const VmAccessOutcome outcome = l2->atomic_rmw_lanes(
      boundary, std::span<const uint64_t>(d.per_lane_addr.data(), d.wf_size), lane_mask, esz,
      [&](uint32_t lane, uint8_t *line_data, uint32_t offset) {
        if (esz == 4) {
          uint32_t old_val;
          std::memcpy(&old_val, line_data + offset, 4);

          uint32_t new_val;
          if (is_packed_add(d.atomic_op)) {
            uint32_t src_val;
            std::memcpy(&src_val, &d.store_data[lane * src_stride], 4);
            new_val = fp_mode::atomic_add_packed_16(old_val, src_val,
                                                    d.atomic_op == AtomicOp::PK_ADD_BF16,
                                                    /*denorm_mode=*/3, d.atomic_source_nan_first);
          } else if (fp_operation) {
            uint32_t source_bits = 0, compare_bits = 0;
            std::memcpy(&source_bits, &d.store_data[lane * src_stride], 4);
            if (uses_two_sources)
              std::memcpy(&compare_bits, &d.store_data[lane * src_stride + 4], 4);
            new_val = apply_fp_atomic(d, *fp_operation, old_val, source_bits, compare_bits,
                                      d.atomic_denorm_mode);
          } else {
            uint32_t src_val = 0, cmp_val = 0;
            std::memcpy(&src_val, &d.store_data[lane * src_stride], 4);
            if (uses_two_sources)
              std::memcpy(&cmp_val, &d.store_data[lane * src_stride + 4], 4);
            new_val = apply_int_atomic(d.atomic_op, old_val, src_val, cmp_val);
          }

          std::memcpy(line_data + offset, &new_val, 4);
          std::memcpy(&d.response_data[lane * 4], &old_val, 4);
        } else if (esz == 8) {
          uint64_t old_val;
          std::memcpy(&old_val, line_data + offset, 8);

          uint64_t new_val;
          if (fp_operation) {
            uint64_t source_bits = 0, compare_bits = 0;
            std::memcpy(&source_bits, &d.store_data[lane * src_stride], 8);
            if (uses_two_sources)
              std::memcpy(&compare_bits, &d.store_data[lane * src_stride + 8], 8);
            new_val = apply_fp_atomic(d, *fp_operation, old_val, source_bits, compare_bits,
                                      d.atomic_denorm_mode);
          } else {
            uint64_t src_val = 0, cmp_val = 0;
            std::memcpy(&src_val, &d.store_data[lane * src_stride], 8);
            if (uses_two_sources)
              std::memcpy(&cmp_val, &d.store_data[lane * src_stride + 8], 8);
            new_val = apply_int_atomic(d.atomic_op, old_val, src_val, cmp_val);
          }

          std::memcpy(line_data + offset, &new_val, 8);
          std::memcpy(&d.response_data[lane * 8], &old_val, 8);
        }
      },
      vmid, vm_access, next_lane);
  d.translated.atomic_lane = next_lane;
  return outcome;
}

/// @brief Perform a per-lane atomic RMW on LDS memory.
void execute_lds_atomic_rmw(VectorMemState &d, Lds *lds,
                            const std::array<uint64_t, 64> &per_lane_addr,
                            const std::vector<uint8_t> &store_data,
                            std::vector<uint8_t> &response_data, uint64_t lane_mask) {
  const uint32_t esz = d.elem_size;
  response_data.resize(d.wf_size * esz);
  const bool uses_two_sources = atomic_uses_two_sources(d.atomic_op);
  const uint32_t src_stride = atomic_source_stride(d, store_data);
  const std::optional<fp_mode::ScalarAtomicOp> fp_operation =
      scalar_fp_atomic_operation(d.atomic_op);

  if (d.atomic_op == AtomicOp::APPEND || d.atomic_op == AtomicOp::CONSUME) {
    uint32_t addr = 0;
    bool any_lane = false;
    for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
      if (lane_mask & (1ULL << lane)) {
        addr = static_cast<uint32_t>(per_lane_addr[lane]);
        any_lane = true;
        break;
      }
    }
    if (!any_lane)
      return;

    const uint32_t old_val = lds->read32(addr);
    const uint32_t active_count = static_cast<uint32_t>(std::popcount(lane_mask));
    for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
      if (!(lane_mask & (1ULL << lane)))
        continue;
      // Both operations broadcast the pre-operation counter to active lanes.
      std::memcpy(&response_data[lane * 4], &old_val, 4);
    }
    const uint32_t new_val =
        d.atomic_op == AtomicOp::APPEND ? old_val + active_count : old_val - active_count;
    lds->write32(addr, new_val);
    return;
  }

  for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
    if (!(lane_mask & (1ULL << lane)))
      continue;

    auto addr = static_cast<uint32_t>(per_lane_addr[lane]);

    if (esz == 4) {
      uint32_t old_val = lds->read32(addr);
      uint32_t new_val;
      if (is_packed_add(d.atomic_op)) {
        uint32_t src_val;
        std::memcpy(&src_val, &store_data[lane * src_stride], 4);
        new_val =
            fp_mode::atomic_add_packed_16(old_val, src_val, d.atomic_op == AtomicOp::PK_ADD_BF16,
                                          d.packed_denorm_mode, /*source_first=*/true);
      } else if (fp_operation) {
        uint32_t source_bits = 0, compare_bits = 0;
        std::memcpy(&source_bits, &store_data[lane * src_stride], 4);
        if (uses_two_sources)
          std::memcpy(&compare_bits, &store_data[lane * src_stride + 4], 4);
        new_val = apply_fp_atomic(d, *fp_operation, old_val, source_bits, compare_bits,
                                  d.atomic_lds_denorm_mode,
                                  /*local=*/true);
      } else {
        uint32_t src_val = 0, cmp_val = 0;
        std::memcpy(&src_val, &store_data[lane * src_stride], 4);
        if (uses_two_sources)
          std::memcpy(&cmp_val, &store_data[lane * src_stride + 4], 4);
        new_val = apply_int_atomic(d.atomic_op, old_val, src_val, cmp_val);
      }
      lds->write32(addr, new_val);
      std::memcpy(&response_data[lane * 4], &old_val, 4);
    } else if (esz == 8) {
      uint64_t old_val = lds->read64(addr);
      uint64_t new_val;
      if (d.atomic_op == AtomicOp::BARRIER_ARRIVE) {
        uint64_t decrement = 0;
        const bool has_decrement = store_data.size() >= lane * src_stride + 8;
        if (has_decrement)
          std::memcpy(&decrement, &store_data[lane * src_stride], 8);
        new_val = lds_barrier_cell_update_arrive(old_val, has_decrement ? decrement : 1);
      } else if (fp_operation) {
        uint64_t source_bits = 0, compare_bits = 0;
        std::memcpy(&source_bits, &store_data[lane * src_stride], 8);
        if (uses_two_sources)
          std::memcpy(&compare_bits, &store_data[lane * src_stride + 8], 8);
        new_val = apply_fp_atomic(d, *fp_operation, old_val, source_bits, compare_bits,
                                  d.atomic_lds_denorm_mode,
                                  /*local=*/true);
      } else {
        uint64_t src_val = 0, cmp_val = 0;
        std::memcpy(&src_val, &store_data[lane * src_stride], 8);
        if (uses_two_sources)
          std::memcpy(&cmp_val, &store_data[lane * src_stride + 8], 8);
        new_val = apply_int_atomic(d.atomic_op, old_val, src_val, cmp_val);
      }
      lds->write64(addr, new_val);
      std::memcpy(&response_data[lane * 8], &old_val, 8);
    }
  }
}

// Dispatch ordinary LDS loads, stores and atomics for uniform and mixed routes.
// Callers prepare buffers and complete retryable global work before entering.
// FLAT LDS addresses include the workgroup base. Reject the captured RDNA4
// allocation-OOB lanes and out-of-backing FLAT lanes before LDS helpers narrow
// to 32 bits, preserving zero reads and discarded writes. Ordinary DS lanes
// retain the existing LDS helper behavior.
void execute_lds_access(VectorMemState &d, Lds &lds, const std::array<uint64_t, 64> &addresses,
                        uint64_t lane_mask, const std::vector<uint8_t> &store_data,
                        std::vector<uint8_t> &response_data) {
  uint64_t out_of_bounds = lane_mask & d.flat_local_oob_lane_mask;
  for (uint64_t lanes = lane_mask & d.flat_local_lane_mask; lanes; lanes &= lanes - 1) {
    const uint32_t lane = std::countr_zero(lanes);
    if (addresses[lane] >= lds.size_bytes())
      out_of_bounds |= uint64_t{1} << lane;
  }
  lane_mask &= ~out_of_bounds;
  if (d.atomic_op == AtomicOp::NONE && d.flat_local_allocation_end && d.elem_size != 0) {
    // gfx1201 FLAT bounds retain complete DWORDs, including for byte-unaligned
    // accesses. Stores crossing the end also discard the incomplete DWORD;
    // the DS unaligned byte-store rule does not describe these FLAT requests.
    const uint64_t limit = std::min<uint64_t>(*d.flat_local_allocation_end, lds.size_bytes());
    const uint32_t bytes_per_lane = d.elem_size * d.num_elems;
    const uint64_t local_lanes = lane_mask & d.flat_local_lane_mask;
    for (uint64_t lanes = local_lanes; lanes; lanes &= lanes - 1) {
      const uint32_t lane = std::countr_zero(lanes);
      const uint64_t address = addresses[lane];
      const uint32_t bytes = static_cast<uint32_t>(
          whole_element_bytes_in_range(address, bytes_per_lane, d.elem_size, limit));
      const uint32_t data_offset = lane * bytes_per_lane;
      if (d.is_load) {
        std::memset(response_data.data() + data_offset, 0, bytes_per_lane);
        std::as_const(lds).read(static_cast<uint32_t>(address), response_data.data() + data_offset,
                                bytes);
      } else {
        lds.write(static_cast<uint32_t>(address), store_data.data() + data_offset, bytes);
      }
    }
    lane_mask &= ~local_lanes;
  }
  if (d.atomic_op != AtomicOp::NONE)
    execute_lds_atomic_rmw(d, &lds, addresses, store_data, response_data, lane_mask);
  else if (d.is_load)
    lds.vector_load(addresses.data(), lane_mask, d.elem_size, d.num_elems, response_data.data());
  else
    lds.vector_store(addresses.data(), lane_mask, d.elem_size, d.num_elems, store_data.data());
  if (d.is_load || d.atomic_op != AtomicOp::NONE) {
    const uint32_t bytes_per_lane = d.elem_size * d.num_elems;
    for (uint64_t lanes = out_of_bounds; lanes; lanes &= lanes - 1)
      std::memset(response_data.data() + std::countr_zero(lanes) * bytes_per_lane, 0,
                  bytes_per_lane);
  }
}

bool rejects_flat_dds(const Instruction &inst, const Wavefront &wf, uint64_t request_lanes) {
  if (!(inst.data_as<VectorMemState>()->flat_dds_lane_mask & request_lanes))
    return false;
  util::Logger::warn("unsupported FLAT DDS execution: ", inst.mnemonic(),
                     " pc=", std::format("{:#x}", wf.pc), ": CDNA5 DDS backing is not implemented");
  return true;
}

} // namespace

VmAccessOutcome GlobalMemPipeline::initiate_access(Instruction &inst, Wavefront &wf) {
  auto &d = *inst.data_as<VectorMemState>();
  d.wg_id = wf.wg_id();
  d.wf_id = wf.wf_id();
  // DDS requires a separate backing implementation. Reject the complete
  // instruction before global or LDS side effects rather than narrowing a
  // DDS address into another workgroup's LDS allocation.
  const uint64_t request_lanes = transpose_request_lane_mask(d, wf.wf_size());
  if (rejects_flat_dds(inst, wf, request_lanes))
    return VmAccessOutcome::Faulted;
  const uint64_t shared_lanes = d.flat_local_lane_mask & request_lanes;
  const VmAccessOutcome outcome = initiate_global_access(d, wf, request_lanes & ~shared_lanes);
  if (outcome != VmAccessOutcome::Complete || shared_lanes == 0)
    return outcome;

  // Commit LDS only after all retryable global work completes. In particular,
  // a global retry must not repeat an LDS atomic or discard either result set.
  execute_lds_access(d, wf.lds(), d.per_lane_addr, shared_lanes, d.store_data, d.response_data);
  return VmAccessOutcome::Complete;
}

VmAccessOutcome GlobalMemPipeline::initiate_global_access(VectorMemState &d, Wavefront &wf,
                                                          uint64_t request_lanes) {
  const uint64_t swizzled_lanes = d.scratch_swizzle ? d.scratch_lane_mask & request_lanes : 0;
  if (d.requires_scratch_backing && swizzled_lanes != 0 && wf.scratch_base() == 0)
    return VmAccessOutcome::Faulted;

  if (wf.address_space()) {
    const bool has_progress = d.translated.request_index != 0 ||
                              d.translated.completed_bytes != 0 || d.translated.atomic_lane != 0 ||
                              d.translated.atomic_loaded;
    const bool may_refresh_unready = d.translated.initialized && d.translated.access &&
                                     !d.translated.access->info().ready && !has_progress;
    if (!d.translated.initialized || may_refresh_unready) {
      d.translated.access = wf.snapshot_vm_access();
      if (!d.translated.access)
        return VmAccessOutcome::Faulted;
      d.translated.initialized = true;
    }
  }
  const bool translated_address_space =
      d.translated.access && !d.translated.access->info().legacy_cache_compatible;

  if (d.atomic_op != AtomicOp::NONE) {
    if (translated_address_space)
      return execute_translated_atomic_rmw(d, request_lanes);
    return execute_atomic_rmw(d, l2_, wf.process_id(), request_lanes);
  }

  // Swizzle is a per-lane layout property. FLAT can mix private-aperture and
  // global lanes, while a buffer SRD can apply swizzling to every active lane.
  // Issue swizzled and plain groups separately so one group's element stride
  // never leaks into the other. Uniform waves take the single-request path.
  const uint64_t plain_lanes = request_lanes & ~swizzled_lanes;
  const uint32_t stride = d.scratch_addr_stride;
  const uint32_t base_offset = d.scratch_addr_base_offset;

  if (translated_address_space) {
    if (d.translated.requests.empty() && d.translated.request_index == 0) {
      if (d.is_load)
        d.response_data.assign(d.wf_size * d.num_elems * d.elem_size, 0);
      if (swizzled_lanes)
        append_translated_requests(d, swizzled_lanes, stride, base_offset,
                                   l1_->transaction_policy());
      if (plain_lanes)
        append_translated_requests(d, plain_lanes, 0, 0, l1_->transaction_policy());
    }
    return execute_translated_transfer(d);
  }

  if (d.is_load) {
    d.response_data.assign(d.wf_size * d.num_elems * d.elem_size, 0);
    if (swizzled_lanes) {
      const VmAccessOutcome outcome = l1_->load(
          d.per_lane_addr.data(), swizzled_lanes, d.elem_size, d.num_elems, d.response_data.data(),
          d.mtype, d.non_temporal, d.request_force_l1_bypass, d.wf_size, wf.process_id(), stride,
          base_offset, d.element_lane_masks.view(), d.scratch_swizzle_unit);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
    }
    if (plain_lanes) {
      const VmAccessOutcome outcome = l1_->load(
          d.per_lane_addr.data(), plain_lanes, d.elem_size, d.num_elems, d.response_data.data(),
          d.mtype, d.non_temporal, d.request_force_l1_bypass, d.wf_size, wf.process_id(), 0,
          /*addr_base_offset=*/0, d.element_lane_masks.view());
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
    }
  } else {
    if (swizzled_lanes) {
      const VmAccessOutcome outcome =
          l1_->store(d.per_lane_addr.data(), swizzled_lanes, d.elem_size, d.num_elems,
                     d.store_data.data(), d.mtype, d.non_temporal, d.wf_size, wf.process_id(),
                     stride, base_offset, d.element_lane_masks.view(), d.scratch_swizzle_unit,
                     GpuVmAccessBatchGuard::active() && !wf.raw_cu().debug_active() &&
                         wf.raw_cu().plugin_group().empty());
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
    }
    if (plain_lanes) {
      const VmAccessOutcome outcome =
          l1_->store(d.per_lane_addr.data(), plain_lanes, d.elem_size, d.num_elems,
                     d.store_data.data(), d.mtype, d.non_temporal, d.wf_size, wf.process_id(), 0,
                     /*addr_base_offset=*/0, d.element_lane_masks.view());
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
    }
  }
  return VmAccessOutcome::Complete;
}

MemoryAccessCompletion GlobalMemPipeline::complete_access(Instruction &inst, Wavefront &wf,
                                                          MemoryAccessDeferredCompletion complete) {
  auto &d = *inst.data_as<VectorMemState>();
  if (d.transpose != 0)
    transpose_response(d);
  return vector_complete(d, wf, wf.raw_cu(), std::move(complete));
}

VmAccessOutcome LocalMemPipeline::initiate_access(Instruction &inst, Wavefront &wf) {
  auto &d = *inst.data_as<VectorMemState>();
  auto &lds = wf.lds();
  d.wf_size = wf.wf_size();
  const uint64_t request_lanes = transpose_request_lane_mask(d, wf.wf_size());
  // Uniform shared-aperture FLAT must reject unsupported DDS too.
  if (rejects_flat_dds(inst, wf, request_lanes))
    return VmAccessOutcome::Faulted;
  d.wg_id = wf.wg_id();
  d.wf_id = wf.wf_id();
  if (d.lds_stack_inputs) {
    execute_lds_stack(wf, d);
    return VmAccessOutcome::Complete;
  }
  const bool plain_load = d.is_load && d.atomic_op == AtomicOp::NONE;
  if (plain_load) {
    d.response_data.resize(d.wf_size * d.num_elems * d.elem_size);
    if (d.ds2_active)
      d.ds2_response_data.resize(d.wf_size * d.num_elems * d.elem_size);
  }
  execute_lds_access(d, lds, d.per_lane_addr, request_lanes, d.store_data, d.response_data);
  if (d.ds2_active)
    execute_lds_access(d, lds, d.ds2_per_lane_addr, d.lane_mask, d.ds2_store_data,
                       d.ds2_response_data);
  if (d.atomic_op != AtomicOp::NONE)
    return VmAccessOutcome::Complete;

  if (d.is_load) {
    // Per-lane LDS load trace: log addresses and loaded values for first 4 lanes.
    util::Logger::vm([&](auto &os) {
      static thread_local uint64_t ds_ld_trace = 0;
      if (++ds_ld_trace > 80)
        return;
      os << std::format("{} wg[{}] wf[{}] DS load: esz={} nelm={} ds2={}", wf.cu().full_path(),
                        d.wg_id, d.wf_id, d.elem_size, d.num_elems, d.ds2_active);
      uint32_t stride = d.num_elems * d.elem_size;
      for (uint32_t ln = 0; ln < d.wf_size; ++ln) {
        if (!(d.lane_mask & (1ULL << ln)))
          continue;
        uint32_t v = 0;
        uint32_t read_size = std::min(stride, 4u);
        if (d.response_data.size() >= ln * stride + read_size)
          std::memcpy(&v, &d.response_data[ln * stride], read_size);
        os << std::format(" L{}:lds[{:#x}]={:#x}", ln, static_cast<uint32_t>(d.per_lane_addr[ln]),
                          v);
        if (d.ds2_active) {
          uint32_t v2 = 0;
          if (d.ds2_response_data.size() >= ln * stride + read_size)
            std::memcpy(&v2, &d.ds2_response_data[ln * stride], read_size);
          os << std::format(",lds2[{:#x}]={:#x}", static_cast<uint32_t>(d.ds2_per_lane_addr[ln]),
                            v2);
        }
      }
    });
  } else {
    // Per-lane LDS store trace: log addresses and values for all lanes.
    util::Logger::vm([&](auto &os) {
      static thread_local uint64_t ds_st_trace = 0;
      bool in_region = false;
      for (uint32_t ln = 0; ln < d.wf_size && !in_region; ++ln)
        if ((d.lane_mask & (1ULL << ln)) && d.per_lane_addr[ln] >= 0x2000 &&
            d.per_lane_addr[ln] < 0x3200)
          in_region = true;
      if (!in_region && ++ds_st_trace > 80)
        return;
      os << std::format("{} wg[{}] wf[{}] DS store: esz={} nelm={} ds2={}", wf.cu().full_path(),
                        d.wg_id, d.wf_id, d.elem_size, d.num_elems, d.ds2_active);
      uint32_t stride = d.num_elems * d.elem_size;
      for (uint32_t ln = 0; ln < d.wf_size; ++ln) {
        if (!(d.lane_mask & (1ULL << ln)))
          continue;
        os << std::format(" L{}:lds[{:#x}]<=", ln, static_cast<uint32_t>(d.per_lane_addr[ln]));
        for (uint32_t e = 0; e < d.num_elems; ++e) {
          uint32_t v = 0;
          uint32_t off = ln * stride + e * d.elem_size;
          uint32_t copy_bytes = std::min(d.elem_size, 4u);
          if (d.store_data.size() >= off + copy_bytes)
            std::memcpy(&v, &d.store_data[off], copy_bytes);
          os << std::format("{}{:#x}", e ? "," : "", v);
        }
        if (d.ds2_active) {
          uint32_t v2 = 0;
          if (stride >= 4 && d.ds2_store_data.size() >= ln * stride + 4)
            std::memcpy(&v2, &d.ds2_store_data[ln * stride], 4);
          os << std::format(",lds2[{:#x}]<={:#x}", static_cast<uint32_t>(d.ds2_per_lane_addr[ln]),
                            v2);
        }
      }
    });
  }
  return VmAccessOutcome::Complete;
}

MemoryAccessCompletion LocalMemPipeline::complete_access(Instruction &inst, Wavefront &wf,
                                                         MemoryAccessDeferredCompletion complete) {
  auto &d = *inst.data_as<VectorMemState>();
  if (d.ds2_active && d.is_load) {
    const uint32_t vgpr_count = d.destination_vgpr_count();
    auto &cu = wf.raw_cu();
    if (!cu.owns_vgpr_range(wf, d.dst_reg_base, vgpr_count) ||
        !cu.owns_vgpr_range(wf, d.ds2_dst_reg_base, d.ds2_destination_vgpr_count()))
      return MemoryAccessCompletion::Complete;
  }
  if (d.transpose != 0)
    transpose_response(d);
  MemoryAccessCompletion completion = vector_complete(d, wf, wf.raw_cu(), std::move(complete));

  // DS dual-access: write the second load or returning-atomic result.
  if (d.ds2_active && d.is_load) {
    auto &cu = wf.raw_cu();
    const uint32_t vgpr_count = d.ds2_destination_vgpr_count();
    // Both destination ranges were validated before the first writeback.
    // As in the ordinary dword-load path, VM completion does not emit an
    // instruction-side register observation. Resolve storage once per register.
    if (d.atomic_op == AtomicOp::NONE && !d.lds_stack_inputs && d.num_elems == 1 &&
        (d.elem_size == 4 || d.elem_size == 8) && vgpr_count == d.elem_size / 4 &&
        d.wf_size <= cu.vgpr_storage_lane_count()) {
      const uint64_t wave_mask = d.wf_size == 64 ? ~uint64_t{0} : (uint64_t{1} << d.wf_size) - 1;
      const uint64_t write_mask = d.lane_mask & wave_mask;
      if (write_mask) {
        for (uint32_t i = 0; i < vgpr_count; ++i) {
          auto *destination =
              reinterpret_cast<uint32_t *>(cu.raw_vgpr_data(d.ds2_dst_reg_base + i));
          for (uint64_t lanes = write_mask; lanes; lanes &= lanes - 1) {
            const uint32_t lane = std::countr_zero(lanes);
            std::memcpy(&destination[lane], &d.ds2_response_data[lane * d.elem_size + i * 4], 4);
          }
        }
      }
    } else {
      for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
        if (!(d.lane_mask & (1ULL << lane)))
          continue;
        for (uint32_t i = 0; i < vgpr_count; ++i) {
          uint32_t val = 0;
          uint32_t data_offset = lane * d.elem_size + i * 4;
          const uint32_t copy_size = std::min(d.elem_size, 4u);
          // Keep the same fixed-size copy specialization as the first destination.
          if (copy_size == 4)
            std::memcpy(&val, &d.ds2_response_data[data_offset], 4);
          else if (copy_size == 2)
            std::memcpy(&val, &d.ds2_response_data[data_offset], 2);
          else
            std::memcpy(&val, &d.ds2_response_data[data_offset], copy_size);
          cu.write_vgpr(d.ds2_dst_reg_base + i, lane, val);
        }
      }
    }
    // Per-lane dual-access completion trace.
    util::Logger::vm([&](auto &os) {
      static thread_local uint64_t ds2_comp_trace = 0;
      if (++ds2_comp_trace > 80)
        return;
      os << std::format("DS dual-access complete: dst1_v={} dst2_v={}", d.dst_reg_base,
                        d.ds2_dst_reg_base);
      for (uint32_t ln = 0; ln < d.wf_size; ++ln) {
        if (!(d.lane_mask & (1ULL << ln)))
          continue;
        uint32_t v1 = cu.read_vgpr_storage(d.dst_reg_base, ln);
        uint32_t v2 = cu.read_vgpr_storage(d.ds2_dst_reg_base, ln);
        os << std::format(" L{}:v{}={:#x},v{}={:#x}", ln, d.dst_reg_base, v1, d.ds2_dst_reg_base,
                          v2);
      }
    });
  }
  return completion;
}

VmAccessOutcome TensorDmaPipeline::initiate_access(Instruction &inst, Wavefront &wf) {
  return resume_tensor_dma(inst, wf);
}

MemoryAccessCompletion
TensorDmaPipeline::complete_access(Instruction & /*inst*/, Wavefront & /*wf*/,
                                   MemoryAccessDeferredCompletion /*complete*/) {
  return MemoryAccessCompletion::Complete;
}

} // namespace amdgpu
} // namespace rocjitsu
