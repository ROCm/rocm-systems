// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/lds_inline_access.h"

#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h"
#include "rocjitsu/isa/arch/amdgpu/shared/ds_transpose.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstring>
#include <span>

namespace rocjitsu {
namespace amdgpu {
namespace {

/// Largest per-lane footprint of a plain DS access (b128).
constexpr uint32_t kMaxLaneBytes = 16;

/// One VGPR's lanes from raw storage. An unmaterialized register reads as the
/// shared zero backing, as it does through RegisterAccess.
const uint32_t *vgpr_lanes(const ComputeUnitCore &cu, uint32_t reg) {
  return reinterpret_cast<const uint32_t *>(cu.raw_vgpr_data(reg));
}

/// Gather a store payload lane-major from the data VGPRs: whole dwords, or the
/// low (D16_HI: high) bytes of one register for narrow stores.
void gather_payload(const LdsAccessRequest &req, const ComputeUnitCore &cu, uint64_t exec,
                    uint32_t stride, uint8_t *payload) {
  if (req.elem_size >= sizeof(uint32_t)) {
    for (uint32_t i = 0; i < stride / sizeof(uint32_t); ++i) {
      const uint32_t *source = vgpr_lanes(cu, req.data_vgpr + i);
      for (uint64_t lanes = exec; lanes; lanes &= lanes - 1) {
        const uint32_t lane = std::countr_zero(lanes);
        std::memcpy(&payload[lane * stride + i * sizeof(uint32_t)], &source[lane],
                    sizeof(uint32_t));
      }
    }
    return;
  }
  assert(req.num_elems == 1 && "narrow DS stores carry one element");
  const uint32_t *source = vgpr_lanes(cu, req.data_vgpr);
  for (uint64_t lanes = exec; lanes; lanes &= lanes - 1) {
    const uint32_t lane = std::countr_zero(lanes);
    const uint32_t value = req.d16_hi ? source[lane] >> 16 : source[lane];
    std::memcpy(&payload[lane * stride], &value, req.elem_size);
  }
}

/// Destination VGPRs a load writes.
uint32_t dst_vgpr_count(uint32_t stride) { return std::max(1u, (stride + 3u) / 4u); }

/// Write a load's lane-major response to its destination VGPRs: the
/// non-atomic part of the routed path's vector_complete(). DS lanes are never
/// dropped as out of bounds, so there is no OOB zeroing.
void write_back(const LdsAccessRequest &req, Wavefront &wf, ComputeUnitCore &cu, uint64_t exec,
                uint32_t stride, const uint8_t *response) {
  const uint32_t vgpr_count = dst_vgpr_count(stride);
  if (req.elem_size == sizeof(uint32_t) && !req.sign_extend && !req.d16_hi && !req.d16_lo) {
    for (uint32_t i = 0; i < vgpr_count; ++i) {
      auto *destination = reinterpret_cast<uint32_t *>(cu.raw_vgpr_data(req.dst_vgpr + i));
      for (uint64_t lanes = exec; lanes; lanes &= lanes - 1) {
        const uint32_t lane = std::countr_zero(lanes);
        std::memcpy(&destination[lane], &response[lane * stride + i * sizeof(uint32_t)],
                    sizeof(uint32_t));
      }
    }
    return;
  }
  const uint32_t copy_size = std::min<uint32_t>(req.elem_size, 4u);
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if (!(exec & (1ULL << lane)))
      continue;
    for (uint32_t i = 0; i < vgpr_count; ++i) {
      uint32_t value = 0;
      const uint8_t *source = &response[lane * stride + i * 4];
      if (copy_size == 4)
        std::memcpy(&value, source, 4);
      else if (copy_size == 2)
        std::memcpy(&value, source, 2);
      else
        std::memcpy(&value, source, copy_size);
      if (req.sign_extend && i == 0 && req.elem_size < 4) {
        if (req.elem_size == 1)
          value = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(value)));
        else
          value = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(value)));
      }
      if (copy_size <= 2 && (req.d16_hi || req.d16_lo)) {
        if (cu.sram_ecc()) {
          value = req.d16_hi ? value << 16 : value & 0xFFFFu;
        } else {
          const uint32_t old = cu.read_vgpr_storage(req.dst_vgpr + i, lane);
          value = req.d16_hi ? (old & 0xFFFFu) | (value << 16)
                             : (old & 0xFFFF0000u) | (value & 0xFFFFu);
        }
      }
      cu.write_vgpr(req.dst_vgpr + i, lane, value);
    }
  }
}

/// 1. Gate: whether @p req may complete here. Nothing has been read or built
///    yet, so declining costs a few loads and branches, and the handler falls
///    back to its VectorMemState.
[[gnu::always_inline]] inline bool can_complete_inline(const LdsAccessRequest &req,
                                                       const ComputeUnitCore &cu,
                                                       const Wavefront &wf, uint32_t stride) {
  // 1a. Decided once per execution by the issue loop (observers of the routed
  //     order, a debugger, RJ_DS_INLINE).
  if (!cu.completes_lds_inline())
    return false;
  // Source registers are read from raw storage. Each case where RegisterAccess
  // would do more than read them takes the VectorMemState path.
  //
  // 1b. Register-read observers expect the notifications RegisterAccess sends.
  if (cu.observes_register_access())
    return false;
  // 1c. The address register comes from the encoding. Outside the issuing wave's
  //     allocation, raw storage belongs to another wave or lies past the file.
  if (!cu.owns_vgpr_range(wf, req.addr_vgpr, 1))
    return false;
  // 1d. Same but for stores: they also read their payload registers.
  if (!req.is_load && !cu.owns_vgpr_range(wf, req.data_vgpr, std::max(1u, stride / 4u)))
    return false;
  // 1e. Same but for loads: they write their destination registers. Unowned, the
  //     VectorMemState path still performs the LDS read and skips the writeback,
  //     as the routed path does.
  if (req.is_load && !cu.owns_vgpr_range(wf, req.dst_vgpr, dst_vgpr_count(stride)))
    return false;
  return true;
}

} // namespace

bool LdsInlineAccess::complete(const Instruction &inst, const LdsAccessRequest &req,
                               Wavefront &wf) {
  ComputeUnitCore &cu = wf.raw_cu();
  const uint32_t stride = req.num_elems * req.elem_size;
  assert(stride != 0 && stride <= kMaxLaneBytes);

  // 1. Gate.
  if (!can_complete_inline(req, cu, wf, stride))
    return false;

  const uint32_t wave = wf.wf_size();
  const uint64_t exec = wf.exec(); // == lane_mask == exec_mask for DS
  Lds &lds = wf.lds();

  // 2. Addresses: one raw read of the address VGPR and a 32-bit sum, zero-extended
  //    for the LDS model. It never reads inactive lanes, so the array is not
  //    initialized.
  alignas(32) uint64_t addresses[64];
  addr_calc::ds_lane_addresses({vgpr_lanes(cu, req.addr_vgpr), wave}, exec, wave,
                               req.offset + wf.lds_base(), addresses);

  // 3. Hazards before any write: addresses are captured above, a store reads its
  //    payload before LDS changes, and a load writes VGPRs only after its LDS
  //    read, so a destination overlapping the address register is safe.

  // 4. Reader/writer by address pattern (lds_access_patterns.h, later): broadcast,
  //    contiguous, otherwise run-walk. Every access currently takes the run-walk in
  //    Lds::vector_load / vector_store, with Lds' bounds and lazy-backing rules:
  //    past capacity or unmaterialized reads as 0, writes past capacity are dropped.

  // 5. The access itself, through per-lane staging on the stack; only active
  //    lanes are written and read.
  alignas(64) uint8_t lanes[64 * kMaxLaneBytes];
  if (req.is_load) {
    // LDS -> staging -> destination VGPR(s), applying sign_extend / d16.
    lds.vector_load(addresses, exec, req.elem_size, req.num_elems, lanes);
    write_back(req, wf, cu, exec, stride, lanes);
  } else {
    // Payload VGPR(s) -> staging -> LDS, lanes ascending so the highest lane
    // wins on overlap.
    gather_payload(req, cu, exec, stride, lanes);
    lds.vector_store(addresses, exec, req.elem_size, req.num_elems, lanes);
  }

  // 6. No wait counters (acquire and release would cancel), no routing, no state.
  //    Register writes bypass instruction write callbacks, as VM completion does.
  cu.note_lds_inline_completion(inst); // CU caches the decoded instruction
  return true;
}

bool LdsInlineAccess::complete(const Instruction &inst, VectorMemState &d, Wavefront &wf) {
  ComputeUnitCore &cu = wf.raw_cu();
  if (!cu.completes_lds_inline() || d.lds_dst)
    return false;
  d.wf_size = wf.wf_size();
  LocalMemPipeline::initiate_lds_access(d, wf, transpose_request_lane_mask(d, d.wf_size));
  const MemoryAccessCompletion completion =
      LocalMemPipeline::complete_lds_access(d, wf, MemoryAccessDeferredCompletion{});
  assert(completion == MemoryAccessCompletion::Complete);
  (void)completion;
  cu.note_lds_inline_completion(inst);
  return true;
}

} // namespace amdgpu
} // namespace rocjitsu
