// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/isa/arch/amdgpu/shared/ds_transpose.h"
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

void write_lds_dst_load_direct(const VectorMemState &d, ComputeUnitCore &cu,
                               uint32_t per_lane_bytes) {
  for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
    if ((d.lane_mask & (1ULL << lane)) == 0)
      continue;
    uint32_t data_offset = lane * per_lane_bytes;
    if (data_offset + per_lane_bytes > d.response_data.size()) {
      throw std::runtime_error(std::format(
          "LDS-destination load payload too small: lane={} offset={} bytes={} payload={}", lane,
          data_offset, per_lane_bytes, d.response_data.size()));
    }
    uint32_t lds_addr =
        d.lds_per_lane_addr ? d.per_lane_lds_addr[lane] : d.lds_base + lane * per_lane_bytes;
    cu.lds().write(lds_addr, &d.response_data[data_offset], per_lane_bytes);
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
                      d.cu_path, d.wg_id, d.wf_id, d.lds_base, per_lane_bytes, d.cluster_mcast_mask,
                      target_count);
    for (uint32_t ln = 0; ln < d.wf_size; ++ln) {
      if (!(d.lane_mask & (1ULL << ln)))
        continue;
      uint32_t v = 0;
      if (per_lane_bytes >= 4)
        std::memcpy(&v, &d.response_data[ln * per_lane_bytes], 4);
      uint32_t lds_addr =
          d.lds_per_lane_addr ? d.per_lane_lds_addr[ln] : d.lds_base + ln * per_lane_bytes;
      os << std::format(" L{}:@{:#x}->lds[{:#x}]={:#x}", ln, d.per_lane_addr[ln], lds_addr, v);
    }
  });

  if (!d.cluster_multicast || cluster_downgrades_to_ordinary) {
    write_lds_dst_load_direct(d, cu, per_lane_bytes);
    return MemoryAccessCompletion::Complete;
  }

  auto txn = make_cluster_lds_multicast_transaction(d, wf, std::move(targets));
  auto result = cu.cluster_lds_multicast_engine().submit(std::move(txn), std::move(complete));

  return result == ClusterLdsMulticastResult::Deferred ? MemoryAccessCompletion::Deferred
                                                       : MemoryAccessCompletion::Complete;
}

MemoryAccessCompletion vector_complete(VectorMemState &d, Wavefront &wf,
                                       MemoryAccessDeferredCompletion complete) {
  ComputeUnitCore &cu = wf.cu();
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
  uint32_t vgpr_count = is_atomic ? (d.elem_size / 4) : std::max(1u, total_bytes / 4);

  // Zero destination VGPRs for OOB lanes. Per AMD ISA spec, out-of-bounds
  // buffer loads return 0. exec_mask is the original EXEC at issue time;
  // lane_mask has OOB lanes removed. The difference gives exec-active OOB lanes.
  uint64_t exec = d.exec_mask;
  uint64_t oob_mask = exec & ~d.lane_mask; // exec-active but OOB
  if (oob_mask) {
    util::Logger::vm([&](auto &os) {
      static uint64_t oob_count = 0;
      if (++oob_count > 10)
        return;
      os << d.cu_path << " wg[" << d.wg_id << "] wf[" << d.wf_id
         << "] OOB zeroing: exec=" << std::hex << d.exec_mask << " lane=" << d.lane_mask
         << " oob=" << oob_mask << std::dec << " dst=" << d.dst_reg_base << " vgprs=" << vgpr_count;
    });
    for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
      if (!(oob_mask & (1ULL << lane)))
        continue;
      for (uint32_t i = 0; i < vgpr_count; ++i) {
        uint32_t val = 0;
        if (!cu.sram_ecc() && d.elem_size <= 2 && (d.d16_hi || d.d16_lo)) {
          const uint32_t old = cu.read_vgpr(d.dst_reg_base + i, lane);
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
          uint32_t old = cu.read_vgpr(d.dst_reg_base + i, lane);
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

void ScalarMemPipeline::initiate_access(Instruction &inst, Wavefront &wf) {
  auto &d = *inst.data_as<ScalarMemState>();
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
}

MemoryAccessCompletion
ScalarMemPipeline::complete_access(Instruction &inst, Wavefront &wf,
                                   MemoryAccessDeferredCompletion /*complete*/) {
  auto &d = *inst.data_as<ScalarMemState>();
  if (!d.is_load)
    return MemoryAccessCompletion::Complete;
  auto &cu = wf.cu();
  for (uint32_t i = 0; i < d.num_dwords; ++i) {
    cu.write_sgpr(d.dst_reg_base + i, d.response_data[i]);
  }
  // Trace: log SMEM load values for debugging.
  util::Logger::vm([&](auto &os) {
    if (wf.wg_id() == 0) {
      static thread_local uint32_t slw_count = 0;
      if (++slw_count <= 100) {
        os << std::format("SMEM complete: addr={:#x} dst_s={} ndw={} data=[{:#x}", d.addr,
                          d.dst_reg_base, d.num_dwords, d.response_data[0]);
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

uint32_t atomic_source_stride(const VectorMemState &d) {
  const bool uses_two_sources =
      (d.atomic_op == AtomicOp::CMPSWAP || d.atomic_op == AtomicOp::MSKOR);
  const uint32_t fallback = uses_two_sources ? d.elem_size * 2 : d.elem_size;
  if (d.wf_size == 0 || d.store_data.empty())
    return fallback;

  const size_t per_lane = d.store_data.size() / d.wf_size;
  return per_lane == 0 ? fallback : static_cast<uint32_t>(per_lane);
}

/// @brief Perform a per-lane atomic RMW through L2.
///
/// Reads old value from L2, applies the atomic operation, writes new value
/// back. Invalidates the L1 line to prevent stale reads. Old values are
/// stored in response_data for GLC return.
void execute_atomic_rmw(VectorMemState &d, L2Cache *l2, L1VectorCache *l1, uint32_t vmid) {
  const uint32_t esz = d.elem_size;
  d.response_data.resize(d.wf_size * esz);
  const bool uses_two_sources =
      (d.atomic_op == AtomicOp::CMPSWAP || d.atomic_op == AtomicOp::MSKOR);
  const uint32_t src_stride = atomic_source_stride(d);
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

    // Invalidate stale L1 line.
    l1->invalidate(ea);
  }
}

/// @brief Perform a per-lane atomic RMW on LDS memory.
void execute_lds_atomic_rmw(VectorMemState &d, Lds *lds) {
  const uint32_t esz = d.elem_size;
  d.response_data.resize(d.wf_size * esz);
  const bool uses_two_sources =
      (d.atomic_op == AtomicOp::CMPSWAP || d.atomic_op == AtomicOp::MSKOR);
  const uint32_t src_stride = atomic_source_stride(d);
  const bool is_fp = (d.atomic_op == AtomicOp::FADD || d.atomic_op == AtomicOp::FMIN ||
                      d.atomic_op == AtomicOp::FMAX);

  if (d.atomic_op == AtomicOp::APPEND || d.atomic_op == AtomicOp::CONSUME) {
    uint32_t addr = 0;
    bool any_lane = false;
    for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
      if (d.lane_mask & (1ULL << lane)) {
        addr = static_cast<uint32_t>(d.per_lane_addr[lane]);
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
      std::memcpy(&d.response_data[lane * 4], &result, 4);
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

    auto addr = static_cast<uint32_t>(d.per_lane_addr[lane]);

    if (esz == 4) {
      uint32_t old_val = lds->read32(addr);
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
      lds->write32(addr, new_val);
      std::memcpy(&d.response_data[lane * 4], &old_val, 4);
    } else if (esz == 8) {
      uint64_t old_val = lds->read64(addr);
      uint64_t new_val;
      if (d.atomic_op == AtomicOp::BARRIER_ARRIVE) {
        uint64_t decrement = 0;
        const bool has_decrement = d.store_data.size() >= lane * src_stride + 8;
        if (has_decrement)
          std::memcpy(&decrement, &d.store_data[lane * src_stride], 8);
        new_val = lds_barrier_cell_update_arrive(old_val, has_decrement ? decrement : 1);
      } else if (is_fp) {
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
      lds->write64(addr, new_val);
      std::memcpy(&d.response_data[lane * 8], &old_val, 8);
    }
  }
}

} // namespace

void GlobalMemPipeline::initiate_access(Instruction &inst, Wavefront &wf) {
  auto &d = *inst.data_as<VectorMemState>();
  if (d.cu_path.empty()) {
    d.cu_path = wf.cu().full_path();
    d.wg_id = wf.wg_id();
    d.wf_id = wf.wf_id();
  }

  if (d.atomic_op != AtomicOp::NONE) {
    execute_atomic_rmw(d, l2_, l1_, wf.process_id());
    return;
  }

  if (d.is_load) {
    d.response_data.resize(d.wf_size * d.num_elems * d.elem_size);
    l1_->load(d.per_lane_addr.data(), d.lane_mask, d.elem_size, d.num_elems, d.response_data.data(),
              d.mtype, d.non_temporal, d.request_force_l1_bypass, wf.process_id());
  } else {
    l1_->store(d.per_lane_addr.data(), d.lane_mask, d.elem_size, d.num_elems, d.store_data.data(),
               d.mtype, d.non_temporal, wf.process_id());
  }
}

MemoryAccessCompletion GlobalMemPipeline::complete_access(Instruction &inst, Wavefront &wf,
                                                          MemoryAccessDeferredCompletion complete) {
  auto &d = *inst.data_as<VectorMemState>();
  if (d.transpose != 0)
    transpose_response(d);
  return vector_complete(d, wf, std::move(complete));
}

void LocalMemPipeline::initiate_access(Instruction &inst, Wavefront &wf) {
  auto &d = *inst.data_as<VectorMemState>();
  if (d.cu_path.empty()) {
    d.cu_path = wf.cu().full_path();
    d.wg_id = wf.wg_id();
    d.wf_id = wf.wf_id();
  }

  if (d.atomic_op != AtomicOp::NONE) {
    execute_lds_atomic_rmw(d, lds_);
    return;
  }

  if (d.is_load) {
    d.response_data.resize(d.wf_size * d.num_elems * d.elem_size);
    lds_->vector_load(d.per_lane_addr.data(), d.lane_mask, d.elem_size, d.num_elems,
                      d.response_data.data());
    if (d.ds2_active) {
      d.ds2_response_data.resize(d.wf_size * d.num_elems * d.elem_size);
      lds_->vector_load(d.ds2_per_lane_addr.data(), d.lane_mask, d.elem_size, d.num_elems,
                        d.ds2_response_data.data());
    }
    // Per-lane LDS load trace: log addresses and loaded values for first 4 lanes.
    util::Logger::vm([&](auto &os) {
      static thread_local uint64_t ds_ld_trace = 0;
      if (++ds_ld_trace > 80)
        return;
      os << std::format("{} wg[{}] wf[{}] DS load: esz={} nelm={} ds2={}", d.cu_path, d.wg_id,
                        d.wf_id, d.elem_size, d.num_elems, d.ds2_active);
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
      os << std::format("{} wg[{}] wf[{}] DS store: esz={} nelm={} ds2={}", d.cu_path, d.wg_id,
                        d.wf_id, d.elem_size, d.num_elems, d.ds2_active);
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
    lds_->vector_store(d.per_lane_addr.data(), d.lane_mask, d.elem_size, d.num_elems,
                       d.store_data.data());
    if (d.ds2_active) {
      lds_->vector_store(d.ds2_per_lane_addr.data(), d.lane_mask, d.elem_size, d.num_elems,
                         d.ds2_store_data.data());
    }
  }
}

MemoryAccessCompletion LocalMemPipeline::complete_access(Instruction &inst, Wavefront &wf,
                                                         MemoryAccessDeferredCompletion complete) {
  auto &d = *inst.data_as<VectorMemState>();
  if (d.transpose != 0)
    transpose_response(d);
  MemoryAccessCompletion completion = vector_complete(d, wf, std::move(complete));

  // DS dual-access (ds_read2/ds_write2): write the second access results.
  if (d.ds2_active && d.is_load) {
    auto &cu = wf.cu();
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
    // Per-lane DS read2 complete trace: show first 4 lanes with both accesses.
    util::Logger::vm([&](auto &os) {
      static thread_local uint64_t ds2_comp_trace = 0;
      if (++ds2_comp_trace > 80)
        return;
      os << std::format("DS read2 complete: dst1_v={} dst2_v={}", d.dst_reg_base,
                        d.ds2_dst_reg_base);
      for (uint32_t ln = 0; ln < d.wf_size; ++ln) {
        if (!(d.lane_mask & (1ULL << ln)))
          continue;
        uint32_t v1 = cu.read_vgpr(d.dst_reg_base, ln);
        uint32_t v2 = cu.read_vgpr(d.ds2_dst_reg_base, ln);
        os << std::format(" L{}:v{}={:#x},v{}={:#x}", ln, d.dst_reg_base, v1, d.ds2_dst_reg_base,
                          v2);
      }
    });
  }
  return completion;
}

} // namespace amdgpu
} // namespace rocjitsu
