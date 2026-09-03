// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/isa/arch/amdgpu/shared/ds_transpose.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_read.h"
#include "rocjitsu/isa/arch/amdgpu/shared/tensor_dma.h"
#include "rocjitsu/isa/isa_traits.h"
#include "rocjitsu/vm/amdgpu/cluster_lds_multicast.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/l1_scalar_cache.h"
#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/lds_barrier_cell.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "util/log.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <stdexcept>
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

WaitCounterType MemoryPipeline::issue_counter(const Instruction &inst) const {
  const DynamicInstState *state = inst.data();
  if (state == nullptr)
    return counter_type_;
  switch (state->tag()) {
  case SCALAR_MEM:
    return inst.data_as<ScalarMemState>()->wait_counter_type;
  case GLOBAL_MEM:
  case LOCAL_MEM:
    return inst.data_as<VectorMemState>()->wait_counter_type;
  default:
    return counter_type_;
  }
}

void MemoryPipeline::complete_entry(PipelineEntry entry) {
  MemoryAccessDeferredCompletion deferred_completion = [this, inst = entry.inst, wf = entry.wf,
                                                        counter = entry.counter,
                                                        wave_generation = entry.wave_generation]() {
    finish_completed_access(inst, *wf, counter, wave_generation);
  };
  const MemoryAccessCompletion completion =
      complete_access(*entry.inst, *entry.wf, std::move(deferred_completion));
  if (completion == MemoryAccessCompletion::Complete)
    finish_completed_access(entry.inst, *entry.wf, entry.counter, entry.wave_generation);
}

VmAccessOutcome MemoryPipeline::issue_impl(Instruction *inst, Wavefront &wf,
                                           bool retain_unavailable) {
  const WaitCounterType counter = issue_counter(*inst);
  wf.wait_counters().increment(counter);
  const VmAccessOutcome access = initiate_access(*inst, wf);
  if (access == VmAccessOutcome::Unavailable && retain_unavailable) {
    issued_.push(
        {.inst = inst, .wf = &wf, .counter = counter, .wave_generation = wf.dispatch_generation()});
    wf.set_state(WfState::VM_RETRY);
    wf.cu().request_functional_yield();
    return VmAccessOutcome::Complete;
  }
  if (access != VmAccessOutcome::Complete) {
    wf.release_wait_counter(counter);
    delete inst;
    return access;
  }
  complete_entry(
      {.inst = inst, .wf = &wf, .counter = counter, .wave_generation = wf.dispatch_generation()});
  return VmAccessOutcome::Complete;
}

VmAccessOutcome MemoryPipeline::issue(Instruction *inst, Wavefront &wf) {
  return issue_impl(inst, wf, false);
}

VmAccessOutcome MemoryPipeline::issue_deferred(Instruction *inst, Wavefront &wf) {
  return issue_impl(inst, wf, true);
}

void MemoryPipeline::defer_unavailable(Instruction *inst, Wavefront &wf) {
  const WaitCounterType counter = issue_counter(*inst);
  wf.wait_counters().increment(counter);
  issued_.push(
      {.inst = inst, .wf = &wf, .counter = counter, .wave_generation = wf.dispatch_generation()});
  wf.set_state(WfState::VM_RETRY);
  wf.cu().request_functional_yield();
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

void MemoryPipeline::tick() {
  const std::size_t pending = issued_.size();
  for (std::size_t index = 0; index < pending; ++index) {
    PipelineEntry entry = issued_.front();
    issued_.pop();
    if (entry.wf->dispatch_generation() != entry.wave_generation) {
      delete entry.inst;
      continue;
    }
    const VmAccessOutcome access = initiate_access(*entry.inst, *entry.wf);
    if (access == VmAccessOutcome::Unavailable) {
      issued_.push(entry);
      entry.wf->cu().request_functional_yield();
      continue;
    }
    if (access == VmAccessOutcome::Complete) {
      complete_entry(entry);
      continue;
    }

    entry.wf->release_wait_counter(entry.counter);
    delete entry.inst;
    if (fault_handler_)
      fault_handler_(*entry.wf, access);
    return;
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
    uint32_t lds_addr =
        d.lds_per_lane_addr ? d.per_lane_lds_addr[lane] : d.lds_base + lane * per_lane_bytes;
    if (lds_addr == kInvalidLdsAddress)
      continue;
    lds.write(lds_addr, &d.response_data[data_offset], per_lane_bytes);
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
    // GFX9/CDNA: out-of-range lanes return zeros, and those zeros are written to LDS.
    const uint64_t write_mask = arch_is_cdna_4_or_lower(cu.arch()) ? d.exec_mask : d.lane_mask;
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

  // Buffer load with LDS bit: scatter loaded data into LDS instead of VGPRs.
  // Each lane writes num_elems * elem_size bytes to LDS at lds_base + lane_offset.
  if (d.lds_dst)
    return complete_lds_dst_load(d, wf, cu, std::move(complete));

  // Atomics: response layout is [lane * elem_size], regular loads are
  // [lane * (num_elems * elem_size) + elem * elem_size].
  bool is_atomic = (d.atomic_op != AtomicOp::NONE);
  uint32_t stride = is_atomic ? d.elem_size : d.num_elems * d.elem_size;
  // Number of destination VGPRs: total bytes / 4, rounded up.
  // For sub-dword elements (u8, u16), at least 1 VGPR is used.
  // For 8-byte elements (b64), 2 VGPRs per element.
  uint32_t total_bytes = d.num_elems * d.elem_size;
  uint32_t vgpr_count =
      is_atomic ? std::max(1u, (d.elem_size + 3u) / 4u) : std::max(1u, (total_bytes + 3u) / 4u);
  if (!cu.owns_vgpr_range(wf, d.dst_reg_base, vgpr_count))
    return MemoryAccessCompletion::Complete;

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
        uint32_t val = 0;
        if (!cu.sram_ecc() && d.elem_size <= 2 && (d.d16_hi || d.d16_lo)) {
          const uint32_t old = cu.read_vgpr_storage(d.dst_reg_base + i, lane);
          val = d.d16_hi ? (old & 0x0000FFFFu) : (old & 0xFFFF0000u);
        }
        cu.write_vgpr(d.dst_reg_base + i, lane, val);
      }
    }
  }
  for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
    if (!(d.lane_mask & (1ULL << lane)))
      continue;
    for (uint32_t i = 0; i < vgpr_count; ++i) {
      uint32_t val = 0;
      uint32_t data_offset = lane * stride + i * 4;
      uint32_t copy_size =
          is_atomic ? std::min(d.elem_size - i * 4, 4u) : std::min(d.elem_size, 4u);
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

} // namespace

VmAccessOutcome ScalarMemPipeline::initiate_access(Instruction &inst, Wavefront &wf) {
  auto &d = *inst.data_as<ScalarMemState>();
  if (wf.address_space()) {
    const bool may_refresh_unready = d.translated.initialized && d.translated.access &&
                                     !d.translated.access->info().ready &&
                                     d.translated.completed_bytes == 0;
    if (!d.translated.initialized || may_refresh_unready) {
      d.translated.access = wf.snapshot_vm_access();
      if (!d.translated.access)
        return VmAccessOutcome::Faulted;
      d.translated.initialized = true;
    }
    if (d.translated.access->info().external) {
      if (d.is_load) {
        if (d.elem_size < 4) {
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

  if (d.is_load) {
    if (d.elem_size < 4) {
      uint8_t bytes[4] = {};
      l1_->load_bytes(d.addr, d.elem_size, bytes, wf.process_id());
      d.response_data[0] = extend_scalar_load(bytes, d.elem_size, d.sign_extend);
    } else {
      l1_->load(d.addr, d.num_dwords, d.response_data, wf.process_id());
    }
  } else {
    l1_->store(d.addr, d.num_dwords, d.store_data, wf.process_id());
  }
  return VmAccessOutcome::Complete;
}

MemoryAccessCompletion
ScalarMemPipeline::complete_access(Instruction &inst, Wavefront &wf,
                                   MemoryAccessDeferredCompletion /*complete*/) {
  auto &d = *inst.data_as<ScalarMemState>();
  if (!d.is_load)
    return MemoryAccessCompletion::Complete;
  if (d.dst_register.width != d.num_dwords)
    return MemoryAccessCompletion::Complete;
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

/// @brief Apply an integer atomic RMW operation (32-bit or 64-bit).
template <typename T> T apply_int_atomic(AtomicOp op, T old_val, T src_val, T cmp_val = 0) {
  using S = std::make_signed_t<T>;
  switch (op) {
  case AtomicOp::SWAP:
    return src_val;
  case AtomicOp::CMPSWAP:
    return (old_val == cmp_val) ? src_val : old_val;
  case AtomicOp::MSKOR:
    return (old_val & ~src_val) | cmp_val;
  case AtomicOp::ADD:
    return old_val + src_val;
  case AtomicOp::SUB:
    return old_val - src_val;
  case AtomicOp::RSUB:
    return src_val - old_val;
  case AtomicOp::SMIN:
    return static_cast<T>(std::min(static_cast<S>(old_val), static_cast<S>(src_val)));
  case AtomicOp::UMIN:
    return std::min(old_val, src_val);
  case AtomicOp::SMAX:
    return static_cast<T>(std::max(static_cast<S>(old_val), static_cast<S>(src_val)));
  case AtomicOp::UMAX:
    return std::max(old_val, src_val);
  case AtomicOp::AND:
    return old_val & src_val;
  case AtomicOp::OR:
    return old_val | src_val;
  case AtomicOp::XOR:
    return old_val ^ src_val;
  case AtomicOp::INC:
    return (old_val >= src_val) ? T{0} : old_val + 1;
  case AtomicOp::DEC:
    return (old_val == 0 || old_val > src_val) ? src_val : old_val - 1;
  default:
    return old_val;
  }
}

/// @brief Apply a floating-point atomic RMW operation.
template <typename F> F apply_fp_atomic(AtomicOp op, F old_val, F src_val) {
  switch (op) {
  case AtomicOp::FADD:
    return old_val + src_val;
  case AtomicOp::FMIN:
    return std::fmin(old_val, src_val);
  case AtomicOp::FMAX:
    return std::fmax(old_val, src_val);
  default:
    return old_val;
  }
}

uint32_t atomic_source_stride(const VectorMemState &d, const std::vector<uint8_t> &store_data) {
  const bool uses_two_sources =
      (d.atomic_op == AtomicOp::CMPSWAP || d.atomic_op == AtomicOp::MSKOR);
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

template <typename F>
VmAccessOutcome for_each_coalesced_lane_run(const uint64_t *addrs, uint64_t lane_mask,
                                            uint32_t wf_size, uint32_t stride, F &&fn) {
  uint64_t remaining = lane_mask;
  while (remaining) {
    const uint32_t first_lane = std::countr_zero(remaining);
    remaining &= ~(uint64_t{1} << first_lane);

    uint32_t last_lane = first_lane;
    while (last_lane + 1 < wf_size) {
      const uint32_t next_lane = last_lane + 1;
      const uint64_t next_bit = uint64_t{1} << next_lane;
      if (!(remaining & next_bit) || addrs[next_lane] != addrs[last_lane] + stride)
        break;
      remaining &= ~next_bit;
      last_lane = next_lane;
    }

    const VmAccessOutcome outcome = fn(first_lane, last_lane - first_lane + 1);
    if (outcome != VmAccessOutcome::Complete)
      return outcome;
  }
  return VmAccessOutcome::Complete;
}

void append_translated_requests(VectorMemState &d, uint64_t lane_mask, uint32_t addr_stride) {
  const uint32_t stride = d.num_elems * d.elem_size;
  auto append = [&](uint64_t address, uint32_t data_offset, uint32_t size) {
    d.translated.requests.push_back({.address = address, .data_offset = data_offset, .size = size});
  };

  if (addr_stride != 0) {
    for (uint32_t elem = 0; elem < d.num_elems; ++elem) {
      uint64_t mask =
          d.element_lane_masks.empty() ? lane_mask : (d.element_lane_masks[elem] & lane_mask);
      while (mask) {
        const uint32_t lane = std::countr_zero(mask);
        mask &= mask - 1;
        const uint64_t base = d.per_lane_addr[lane];
        uint32_t copied = elem * d.elem_size;
        const uint32_t elem_end = copied + d.elem_size;
        while (copied < elem_end) {
          const uint32_t byte_in_dword = static_cast<uint32_t>((base + copied) & 3);
          const uint32_t chunk = std::min(elem_end - copied, 4 - byte_in_dword);
          const uint64_t address =
              (base & ~uint64_t{3}) + ((base & 3) + copied) / 4 * addr_stride + byte_in_dword;
          append(address, lane * stride + copied, chunk);
          copied += chunk;
        }
      }
    }
    return;
  }

  if (!all_elements_use_lane_mask(d.element_lane_masks.view(), lane_mask, d.num_elems)) {
    const uint64_t full_lane_mask = fully_valid_lane_mask(d.element_lane_masks.view(), lane_mask);
    (void)for_each_coalesced_lane_run(d.per_lane_addr.data(), full_lane_mask, d.wf_size, stride,
                                      [&](uint32_t first_lane, uint32_t run_lanes) {
                                        append(d.per_lane_addr[first_lane], first_lane * stride,
                                               run_lanes * stride);
                                        return VmAccessOutcome::Complete;
                                      });
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

  (void)for_each_coalesced_lane_run(d.per_lane_addr.data(), lane_mask, d.wf_size, stride,
                                    [&](uint32_t first_lane, uint32_t run_lanes) {
                                      append(d.per_lane_addr[first_lane], first_lane * stride,
                                             run_lanes * stride);
                                      return VmAccessOutcome::Complete;
                                    });
}

VmAccessOutcome execute_translated_transfer(VectorMemState &d) {
  assert(d.translated.access.has_value());
  while (d.translated.request_index < d.translated.requests.size()) {
    const TranslatedMemoryRequest &request = d.translated.requests[d.translated.request_index];
    VmAccessOutcome outcome = VmAccessOutcome::Malformed;
    if (d.is_load) {
      outcome = d.translated.access->read(
          request.address,
          std::span<std::byte>(
              reinterpret_cast<std::byte *>(d.response_data.data() + request.data_offset),
              request.size),
          d.translated.completed_bytes);
    } else {
      outcome = d.translated.access->write(
          request.address,
          std::span<const std::byte>(
              reinterpret_cast<const std::byte *>(d.store_data.data() + request.data_offset),
              request.size),
          d.translated.completed_bytes);
    }
    if (outcome != VmAccessOutcome::Complete)
      return outcome;
    ++d.translated.request_index;
    d.translated.completed_bytes = 0;
  }
  return VmAccessOutcome::Complete;
}

VmAccessOutcome execute_translated_atomic_rmw(VectorMemState &d) {
  if (d.atomic_op == AtomicOp::APPEND || d.atomic_op == AtomicOp::CONSUME ||
      d.atomic_op == AtomicOp::BARRIER_ARRIVE) {
    return VmAccessOutcome::Malformed;
  }

  const uint32_t width = d.elem_size;
  d.response_data.resize(d.wf_size * width);
  const bool uses_two_sources = d.atomic_op == AtomicOp::CMPSWAP || d.atomic_op == AtomicOp::MSKOR;
  const uint32_t source_stride = atomic_source_stride(d, d.store_data);
  const bool is_fp = d.atomic_op == AtomicOp::FADD || d.atomic_op == AtomicOp::FMIN ||
                     d.atomic_op == AtomicOp::FMAX;

  assert(d.translated.access.has_value());
  for (uint32_t lane = d.translated.atomic_lane; lane < d.wf_size; ++lane) {
    if (!(d.lane_mask & (uint64_t{1} << lane)))
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
            is_fp ? std::bit_cast<uint32_t>(apply_fp_atomic(
                        d.atomic_op, std::bit_cast<float>(old32), std::bit_cast<float>(source)))
                  : apply_int_atomic(d.atomic_op, old32, source, compare);
      } else if (width == sizeof(uint64_t)) {
        uint64_t source = 0;
        uint64_t compare = 0;
        std::memcpy(&source, &d.store_data[lane * source_stride], sizeof(source));
        if (uses_two_sources)
          std::memcpy(&compare, &d.store_data[lane * source_stride + sizeof(source)],
                      sizeof(compare));
        compare_value = compare;
        new_value = is_fp ? std::bit_cast<uint64_t>(
                                apply_fp_atomic(d.atomic_op, std::bit_cast<double>(old_value),
                                                std::bit_cast<double>(source)))
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
void execute_atomic_rmw(VectorMemState &d, L2Cache *l2, uint32_t vmid) {
  const uint32_t esz = d.elem_size;
  d.response_data.resize(d.wf_size * esz);
  const bool uses_two_sources =
      (d.atomic_op == AtomicOp::CMPSWAP || d.atomic_op == AtomicOp::MSKOR);
  const uint32_t src_stride = atomic_source_stride(d, d.store_data);
  const bool is_fp = (d.atomic_op == AtomicOp::FADD || d.atomic_op == AtomicOp::FMIN ||
                      d.atomic_op == AtomicOp::FMAX);

  for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
    if (!(d.lane_mask & (1ULL << lane)))
      continue;

    uint64_t ea = d.per_lane_addr[lane];

    // Perform the atomic RMW under L2's atomic lock.
    l2->atomic_rmw(
        ea, esz,
        [&](uint8_t *line_data, uint32_t offset) {
          if (esz == 4) {
            uint32_t old_val;
            std::memcpy(&old_val, line_data + offset, 4);

            uint32_t new_val;
            if (is_fp) {
              float old_f = std::bit_cast<float>(old_val);
              float src_f;
              std::memcpy(&src_f, &d.store_data[lane * src_stride], 4);
              new_val = std::bit_cast<uint32_t>(apply_fp_atomic(d.atomic_op, old_f, src_f));
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
            if (is_fp) {
              double old_f = std::bit_cast<double>(old_val);
              double src_f;
              std::memcpy(&src_f, &d.store_data[lane * src_stride], 8);
              new_val = std::bit_cast<uint64_t>(apply_fp_atomic(d.atomic_op, old_f, src_f));
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
        vmid);
  }
}

/// @brief Perform a per-lane atomic RMW on LDS memory.
void execute_lds_atomic_rmw(VectorMemState &d, Lds *lds,
                            const std::array<uint64_t, 64> &per_lane_addr,
                            const std::vector<uint8_t> &store_data,
                            std::vector<uint8_t> &response_data) {
  const uint32_t esz = d.elem_size;
  response_data.resize(d.wf_size * esz);
  const bool uses_two_sources =
      (d.atomic_op == AtomicOp::CMPSWAP || d.atomic_op == AtomicOp::MSKOR);
  const uint32_t src_stride = atomic_source_stride(d, store_data);
  const bool is_fp = (d.atomic_op == AtomicOp::FADD || d.atomic_op == AtomicOp::FMIN ||
                      d.atomic_op == AtomicOp::FMAX);

  if (d.atomic_op == AtomicOp::APPEND || d.atomic_op == AtomicOp::CONSUME) {
    uint32_t addr = 0;
    bool any_lane = false;
    for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
      if (d.lane_mask & (1ULL << lane)) {
        addr = static_cast<uint32_t>(per_lane_addr[lane]);
        any_lane = true;
        break;
      }
    }
    if (!any_lane)
      return;

    const uint32_t old_val = lds->read32(addr);
    const uint32_t active_count = static_cast<uint32_t>(std::popcount(d.lane_mask));
    uint32_t active_rank = 0;
    for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
      if (!(d.lane_mask & (1ULL << lane)))
        continue;
      const uint32_t result =
          d.atomic_op == AtomicOp::APPEND ? old_val + active_rank : old_val - active_rank - 1;
      std::memcpy(&response_data[lane * 4], &result, 4);
      ++active_rank;
    }
    const uint32_t new_val =
        d.atomic_op == AtomicOp::APPEND ? old_val + active_count : old_val - active_count;
    lds->write32(addr, new_val);
    return;
  }

  for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
    if (!(d.lane_mask & (1ULL << lane)))
      continue;

    auto addr = static_cast<uint32_t>(per_lane_addr[lane]);

    if (esz == 4) {
      uint32_t old_val = lds->read32(addr);
      uint32_t new_val;
      if (is_fp) {
        float old_f = std::bit_cast<float>(old_val);
        float src_f;
        std::memcpy(&src_f, &store_data[lane * src_stride], 4);
        new_val = std::bit_cast<uint32_t>(apply_fp_atomic(d.atomic_op, old_f, src_f));
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
      } else if (is_fp) {
        double old_f = std::bit_cast<double>(old_val);
        double src_f;
        std::memcpy(&src_f, &store_data[lane * src_stride], 8);
        new_val = std::bit_cast<uint64_t>(apply_fp_atomic(d.atomic_op, old_f, src_f));
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

} // namespace

VmAccessOutcome GlobalMemPipeline::initiate_access(Instruction &inst, Wavefront &wf) {
  auto &d = *inst.data_as<VectorMemState>();
  d.wg_id = wf.wg_id();
  d.wf_id = wf.wf_id();
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
  const bool translated_address_space = d.translated.access && d.translated.access->info().external;

  if (d.atomic_op != AtomicOp::NONE) {
    if (translated_address_space)
      return execute_translated_atomic_rmw(d);
    execute_atomic_rmw(d, l2_, wf.process_id());
    return VmAccessOutcome::Complete;
  }

  // A FLAT wave can route some lanes to the private aperture and others to
  // global memory. The swizzle stride is a property of the lanes that landed in
  // scratch, not of the instruction, so issue the two groups separately rather
  // than striding a global lane's second dword by lane_count*4. Uniform waves
  // (including every dedicated SCRATCH op) take the single-request path.
  const uint64_t request_lanes = transpose_request_lane_mask(d);
  const uint64_t scratch_lanes = d.scratch_swizzle ? d.scratch_lane_mask & request_lanes : 0;
  const uint64_t plain_lanes = request_lanes & ~scratch_lanes;
  const uint32_t stride = d.scratch_addr_stride;

  if (translated_address_space) {
    if (d.translated.requests.empty() && d.translated.request_index == 0) {
      if (d.is_load)
        d.response_data.assign(d.wf_size * d.num_elems * d.elem_size, 0);
      if (scratch_lanes)
        append_translated_requests(d, scratch_lanes, stride);
      if (plain_lanes)
        append_translated_requests(d, plain_lanes, 0);
    }
    return execute_translated_transfer(d);
  }

  if (d.is_load) {
    const uint64_t request_lanes = transpose_request_lane_mask(d);
    const uint64_t scratch_request_lanes = scratch_lanes & request_lanes;
    const uint64_t plain_request_lanes = plain_lanes & request_lanes;
    d.response_data.assign(d.wf_size * d.num_elems * d.elem_size, 0);
    if (scratch_request_lanes) {
      l1_->load(d.per_lane_addr.data(), scratch_request_lanes, d.elem_size, d.num_elems,
                d.response_data.data(), d.mtype, d.non_temporal, d.request_force_l1_bypass,
                d.wf_size, wf.process_id(), stride, d.element_lane_masks.view());
    }
    if (plain_request_lanes) {
      l1_->load(d.per_lane_addr.data(), plain_request_lanes, d.elem_size, d.num_elems,
                d.response_data.data(), d.mtype, d.non_temporal, d.request_force_l1_bypass,
                d.wf_size, wf.process_id(), 0, d.element_lane_masks.view());
    }
  } else {
    if (scratch_lanes) {
      l1_->store(d.per_lane_addr.data(), scratch_lanes, d.elem_size, d.num_elems,
                 d.store_data.data(), d.mtype, d.non_temporal, d.wf_size, wf.process_id(), stride,
                 d.element_lane_masks.view());
    }
    if (plain_lanes) {
      l1_->store(d.per_lane_addr.data(), plain_lanes, d.elem_size, d.num_elems, d.store_data.data(),
                 d.mtype, d.non_temporal, d.wf_size, wf.process_id(), 0,
                 d.element_lane_masks.view());
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
  d.wg_id = wf.wg_id();
  d.wf_id = wf.wf_id();
  if (d.atomic_op != AtomicOp::NONE) {
    execute_lds_atomic_rmw(d, &lds, d.per_lane_addr, d.store_data, d.response_data);
    if (d.ds2_active) {
      execute_lds_atomic_rmw(d, &lds, d.ds2_per_lane_addr, d.ds2_store_data, d.ds2_response_data);
    }
    return VmAccessOutcome::Complete;
  }

  if (d.is_load) {
    d.response_data.resize(d.wf_size * d.num_elems * d.elem_size);
    lds.vector_load(d.per_lane_addr.data(), transpose_request_lane_mask(d), d.elem_size,
                    d.num_elems, d.response_data.data());
    if (d.ds2_active) {
      d.ds2_response_data.resize(d.wf_size * d.num_elems * d.elem_size);
      lds.vector_load(d.ds2_per_lane_addr.data(), d.lane_mask, d.elem_size, d.num_elems,
                      d.ds2_response_data.data());
    }
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
    lds.vector_store(d.per_lane_addr.data(), d.lane_mask, d.elem_size, d.num_elems,
                     d.store_data.data());
    if (d.ds2_active) {
      lds.vector_store(d.ds2_per_lane_addr.data(), d.lane_mask, d.elem_size, d.num_elems,
                       d.ds2_store_data.data());
    }
  }
  return VmAccessOutcome::Complete;
}

MemoryAccessCompletion LocalMemPipeline::complete_access(Instruction &inst, Wavefront &wf,
                                                         MemoryAccessDeferredCompletion complete) {
  auto &d = *inst.data_as<VectorMemState>();
  if (d.transpose != 0)
    transpose_response(d);
  MemoryAccessCompletion completion = vector_complete(d, wf, wf.raw_cu(), std::move(complete));

  // DS dual-access: write the second load or returning-atomic result.
  if (d.ds2_active && d.is_load) {
    auto &cu = wf.raw_cu();
    uint32_t vgpr_count = d.elem_size / 4;
    for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
      if (!(d.lane_mask & (1ULL << lane)))
        continue;
      for (uint32_t i = 0; i < vgpr_count; ++i) {
        uint32_t val = 0;
        uint32_t data_offset = lane * d.elem_size + i * 4;
        std::memcpy(&val, &d.ds2_response_data[data_offset], std::min(d.elem_size, 4u));
        cu.write_vgpr(d.ds2_dst_reg_base + i, lane, val);
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
