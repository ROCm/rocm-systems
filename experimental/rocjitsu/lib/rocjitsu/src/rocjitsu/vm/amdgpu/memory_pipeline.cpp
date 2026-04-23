// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/l1_scalar_cache.h"
#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "util/log.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace rocjitsu {
namespace amdgpu {

namespace {

/// Shared complete_access logic for vector/LDS loads (write VGPRs from
/// response data). Used by both GlobalMemPipeline and LocalMemPipeline.
///
/// For atomics with elem_size=8 and num_elems=1, the 8-byte old value
/// occupies two consecutive VGPRs (low dword in vdst, high in vdst+1).
void vector_complete(VectorMemState &d, ComputeUnitCore &cu) {
  if (!d.is_load)
    return;

  // Buffer load with LDS bit: scatter loaded data into LDS instead of VGPRs.
  // Each lane writes num_elems * elem_size bytes to LDS at lds_base + lane_offset.
  if (d.lds_dst) {
    auto &lds = cu.lds();
    uint32_t per_lane_bytes = d.num_elems * d.elem_size;
    for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
      if (!(d.lane_mask & (1ULL << lane)))
        continue;
      uint32_t lds_addr = d.lds_base + lane * per_lane_bytes;
      uint32_t data_offset = lane * per_lane_bytes;
      for (uint32_t b = 0; b < per_lane_bytes; ++b)
        lds.write8(lds_addr + b, d.response_data[data_offset + b]);
    }
    return;
  }

  // Atomics: response layout is [lane * elem_size], regular loads are
  // [lane * (num_elems * elem_size) + elem * elem_size].
  bool is_atomic = (d.atomic_op != AtomicOp::NONE);
  uint32_t stride = is_atomic ? d.elem_size : d.num_elems * d.elem_size;
  uint32_t vgpr_count = is_atomic ? (d.elem_size / 4) : d.num_elems;

  for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
    if (!(d.lane_mask & (1ULL << lane)))
      continue;
    for (uint32_t i = 0; i < vgpr_count; ++i) {
      uint32_t val = 0;
      uint32_t data_offset = lane * stride + i * 4;
      // For atomics (8-byte element split across 2 VGPRs): copy 4 then 4.
      // For regular loads (one element per VGPR): always copy elem_size bytes.
      uint32_t copy_size =
          is_atomic ? std::min(d.elem_size - i * 4, 4u) : std::min(d.elem_size, 4u);
      std::memcpy(&val, &d.response_data[data_offset], copy_size);
      cu.write_vgpr(d.dst_reg_base + i, lane, val);
    }
  }
}

} // namespace

void ScalarMemPipeline::initiate_access(Instruction &inst, Wavefront & /*wf*/) {
  auto &d = *inst.data_as<ScalarMemState>();
  if (d.is_load) {
    l1_->load(d.addr, d.num_dwords, d.response_data);
  } else {
    l1_->store(d.addr, d.num_dwords, d.store_data);
  }
}

void ScalarMemPipeline::complete_access(Instruction &inst, Wavefront &wf) {
  auto &d = *inst.data_as<ScalarMemState>();
  if (!d.is_load)
    return;
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

/// @brief Perform a per-lane atomic RMW through L2.
///
/// Reads old value from L2, applies the atomic operation, writes new value
/// back. Invalidates the L1 line to prevent stale reads. Old values are
/// stored in response_data for GLC return.
void execute_atomic_rmw(VectorMemState &d, L2Cache *l2, L1VectorCache *l1) {
  const uint32_t esz = d.elem_size;
  d.response_data.resize(d.wf_size * esz);

  for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
    if (!(d.lane_mask & (1ULL << lane)))
      continue;

    uint64_t ea = d.per_lane_addr[lane];
    bool is_cmpswap = (d.atomic_op == AtomicOp::CMPSWAP);
    uint32_t src_stride = is_cmpswap ? esz * 2 : esz;
    bool is_fp = (d.atomic_op == AtomicOp::FADD || d.atomic_op == AtomicOp::FMIN ||
                  d.atomic_op == AtomicOp::FMAX);

    // Perform the atomic RMW under L2's atomic lock.
    l2->atomic_rmw(ea, esz, [&](uint8_t *line_data, uint32_t offset) {
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
          if (is_cmpswap)
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
          if (is_cmpswap)
            std::memcpy(&cmp_val, &d.store_data[lane * src_stride + 8], 8);
          new_val = apply_int_atomic(d.atomic_op, old_val, src_val, cmp_val);
        }

        std::memcpy(line_data + offset, &new_val, 8);
        std::memcpy(&d.response_data[lane * 8], &old_val, 8);
      }
    });

    // Invalidate stale L1 line.
    l1->invalidate(ea);
  }
}

/// @brief Perform a per-lane atomic RMW on LDS memory.
void execute_lds_atomic_rmw(VectorMemState &d, Lds *lds) {
  const uint32_t esz = d.elem_size;
  d.response_data.resize(d.wf_size * esz);

  for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
    if (!(d.lane_mask & (1ULL << lane)))
      continue;

    auto addr = static_cast<uint32_t>(d.per_lane_addr[lane]);
    bool is_cmpswap = (d.atomic_op == AtomicOp::CMPSWAP);
    uint32_t src_stride = is_cmpswap ? esz * 2 : esz;

    bool is_fp = (d.atomic_op == AtomicOp::FADD || d.atomic_op == AtomicOp::FMIN ||
                  d.atomic_op == AtomicOp::FMAX);

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
        if (is_cmpswap)
          std::memcpy(&cmp_val, &d.store_data[lane * src_stride + 4], 4);
        new_val = apply_int_atomic(d.atomic_op, old_val, src_val, cmp_val);
      }
      lds->write32(addr, new_val);
      std::memcpy(&d.response_data[lane * 4], &old_val, 4);
    } else if (esz == 8) {
      uint64_t old_val = lds->read64(addr);
      uint64_t new_val;
      if (is_fp) {
        double old_f = std::bit_cast<double>(old_val);
        double src_f;
        std::memcpy(&src_f, &d.store_data[lane * src_stride], 8);
        new_val = std::bit_cast<uint64_t>(apply_fp_atomic(d.atomic_op, old_f, src_f));
      } else {
        uint64_t src_val = 0, cmp_val = 0;
        std::memcpy(&src_val, &d.store_data[lane * src_stride], 8);
        if (is_cmpswap)
          std::memcpy(&cmp_val, &d.store_data[lane * src_stride + 8], 8);
        new_val = apply_int_atomic(d.atomic_op, old_val, src_val, cmp_val);
      }
      lds->write64(addr, new_val);
      std::memcpy(&d.response_data[lane * 8], &old_val, 8);
    }
  }
}

} // namespace

void GlobalMemPipeline::initiate_access(Instruction &inst, [[maybe_unused]] Wavefront &wf) {
  auto &d = *inst.data_as<VectorMemState>();

  if (d.atomic_op != AtomicOp::NONE) {
    execute_atomic_rmw(d, l2_, l1_);
    return;
  }

  if (d.is_load) {
    d.response_data.resize(d.wf_size * d.num_elems * d.elem_size);
    l1_->load(d.per_lane_addr.data(), d.lane_mask, d.elem_size, d.num_elems, d.response_data.data(),
              d.mtype, d.non_temporal);
  } else {
    // Trace: dump per-lane values for stores to tensor address range.
    util::Logger::vm([&](auto &os) {
      if (d.lane_mask == 0)
        return;
      uint64_t rm = d.lane_mask;
      bool hits_tensor = false;
      while (rm) {
        uint32_t ln = __builtin_ctzll(rm);
        rm &= rm - 1;
        if (d.per_lane_addr[ln] >= 0x4d00c00000ULL && d.per_lane_addr[ln] < 0x4d00c00100ULL) {
          hits_tensor = true;
          break;
        }
      }
      if (!hits_tensor)
        return;
      uint32_t stride = d.num_elems * d.elem_size;
      rm = d.lane_mask;
      int cnt = 0;
      while (rm && cnt < 6) {
        uint32_t ln = __builtin_ctzll(rm);
        rm &= rm - 1;
        uint64_t a = d.per_lane_addr[ln];
        if (a < 0x4d00c00000ULL || a >= 0x4d00c00100ULL)
          continue;
        uint32_t v = 0;
        if (stride > 0 && d.store_data.size() >= ln * stride + 4)
          std::memcpy(&v, &d.store_data[ln * stride], 4);
        if (cnt > 0)
          os << '\n' << std::format("[rj log VM] ");
        os << std::format("SLANE L{} @+{:#x} ={:#x} ipc={:#x} exec={:#x} wf={} wg={}", ln,
                          a - 0x4d00c00000ULL, v, d.issue_pc, d.lane_mask, wf.wf_id(), wf.wg_id());
        ++cnt;
      }
    });
    l1_->store(d.per_lane_addr.data(), d.lane_mask, d.elem_size, d.num_elems, d.store_data.data(),
               d.mtype, d.non_temporal);
  }
}

void GlobalMemPipeline::complete_access(Instruction &inst, Wavefront &wf) {
  vector_complete(*inst.data_as<VectorMemState>(), wf.cu());
}

void LocalMemPipeline::initiate_access(Instruction &inst, Wavefront & /*wf*/) {
  auto &d = *inst.data_as<VectorMemState>();

  if (d.atomic_op != AtomicOp::NONE) {
    execute_lds_atomic_rmw(d, lds_);
    return;
  }

  if (d.is_load) {
    d.response_data.resize(d.wf_size * d.num_elems * d.elem_size);
    lds_->vector_load(d.per_lane_addr.data(), d.lane_mask, d.elem_size, d.num_elems,
                      d.response_data.data());
  } else {
    lds_->vector_store(d.per_lane_addr.data(), d.lane_mask, d.elem_size, d.num_elems,
                       d.store_data.data());
  }
}

void LocalMemPipeline::complete_access(Instruction &inst, Wavefront &wf) {
  vector_complete(*inst.data_as<VectorMemState>(), wf.cu());
}

} // namespace amdgpu
} // namespace rocjitsu
