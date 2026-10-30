// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file lds_inline_access_steps.h
/// @brief The numbered steps of a plain DS load or store on the request path,
/// as templates over the access direction and the wave size.
/// @details Included by lds_inline_access.h, so a load's steps are inlined into
/// its generated handler together with its writeback.

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_scalar.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/lds_inline_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstring>

namespace rocjitsu {
namespace amdgpu {
namespace lds_inline {

/// One VGPR's lanes from raw storage. An unmaterialized register reads as the
/// shared zero backing, as it does through RegisterAccess.
inline const uint32_t *vgpr_lanes(const ComputeUnitCore &cu, uint32_t reg) {
  return reinterpret_cast<const uint32_t *>(cu.raw_vgpr_data(reg));
}

/// Gather a store payload lane-major from the data VGPRs: whole dwords, or the
/// low (D16_HI: high) bytes of one register for narrow stores.
inline void gather_payload(const LdsAccessRequest &req, const ComputeUnitCore &cu, uint64_t exec,
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
constexpr uint32_t dst_vgpr_count(uint32_t stride) { return std::max(1u, (stride + 3u) / 4u); }

/// 1. Gate: whether @p req may complete here. Nothing has been read or built
///    yet, so declining costs a few loads and branches, and the handler falls
///    back to its VectorMemState.
template <bool IsLoad>
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
  if constexpr (IsLoad) {
    // 1e. Same but for loads: they write their destination registers. Unowned,
    //     the VectorMemState path still performs the LDS read and skips the
    //     writeback, as the routed path does.
    return cu.owns_vgpr_range(wf, req.dst_vgpr, dst_vgpr_count(stride));
  } else {
    // 1d. Same but for stores: they also read their payload registers.
    return cu.owns_vgpr_range(wf, req.data_vgpr, std::max(1u, stride / 4u));
  }
}

/// Steps 2-5 for a wave of @p Wave lanes, after the gate passed: a store
/// completes; a load stops with its data read through @p load.group (in
/// @p staging or in place in LDS) and its destination in @p load.
template <bool IsLoad, uint32_t Wave>
[[gnu::always_inline]] inline void run_steps(const LdsAccessRequest &req, Wavefront &wf,
                                             ComputeUnitCore &cu, uint32_t stride, uint8_t *staging,
                                             LdsStagedLoad &load) {
  const uint64_t exec = wf.exec(); // == lane_mask == exec_mask for DS; no lanes past Wave
  Lds &lds = wf.lds();

  // 2. Addresses: lane l accesses (address_lanes[l] + offset) mod 2^32. Loads give
  //    the raw lanes and the offset to the reader, which adds the offset only where
  //    it needs an address; stores expand their active lanes to 64-bit addresses.
  const uint32_t *address_lanes = vgpr_lanes(cu, req.addr_vgpr);
  const uint32_t offset = req.offset + wf.lds_base();

  // 3. Hazards before any write: the reader reads the address lanes before the
  //    writeback starts, a store reads its payload before LDS changes, so a
  //    destination overlapping the address register is safe.

  // 4. Reader by address pattern (lds_access_patterns.h): contiguous 16-lane
  //    segments are block copies, b64 rows where lane l + 16 follows lane l one
  //    16-byte column per pair, up to four lanes fixed copies, other dense masks
  //    one bounds check and fixed copies. Lanes not fully backed keep Lds' exact
  //    rules. Stores keep the run walk in Lds::vector_store.

  // 5. The access itself, through per-lane staging on the caller's stack. The
  //    reader may write any lane of the span; writeback reads active lanes only.
  if constexpr (IsLoad) {
    // LDS -> staging, except 16-lane groups proved contiguous and backed,
    // which the writeback reads in place. The caller's typed writeback
    // (lds_writeback.h) then stores the active lanes to the destination
    // VGPR(s), applying sign_extend / d16; ownership was checked in 1e.
    lds.vector_load32<Wave>(address_lanes, offset, exec, req.elem_size, req.num_elems, staging,
                            load.group);
    for (uint32_t i = 0; i < dst_vgpr_count(stride); ++i)
      load.dst[i] = cu.raw_vgpr_data(req.dst_vgpr + i);
    load.exec = exec;
    load.wave = Wave;
    load.sram_ecc = cu.sram_ecc();
  } else {
    // Payload VGPR(s) -> staging -> LDS, lanes ascending so the highest lane
    // wins on overlap. Inactive address lanes are never read.
    (void)load;
    alignas(32) uint64_t addresses[64];
    addr_calc::ds_lane_addresses({address_lanes, Wave}, exec, Wave, offset, addresses);
    gather_payload(req, cu, exec, stride, staging);
    lds.vector_store(addresses, exec, req.elem_size, req.num_elems, staging);
  }
}

} // namespace lds_inline

/// A whole plain access, gate to step 6. A load stops before its writeback.
template <bool IsLoad>
[[gnu::always_inline]] inline bool
LdsInlineAccess::access(const Instruction &inst, const LdsAccessRequest &req, Wavefront &wf,
                        uint8_t *staging, LdsStagedLoad &load) {
  ComputeUnitCore &cu = wf.raw_cu();
  const uint32_t stride = req.num_elems * req.elem_size;
  assert(stride != 0 && stride <= kMaxLaneBytes);

  // 1. Gate.
  if (!lds_inline::can_complete_inline<IsLoad>(req, cu, wf, stride))
    return false;

  // 2-5, specialized by wave size.
  if (wf.wf_size() == 32)
    lds_inline::run_steps<IsLoad, 32>(req, wf, cu, stride, staging, load);
  else
    lds_inline::run_steps<IsLoad, 64>(req, wf, cu, stride, staging, load);

  // 6. No wait counters (acquire and release would cancel), no routing, no state.
  //    Register writes bypass instruction write callbacks, as VM completion does.
  cu.note_lds_inline_completion(inst); // CU caches the decoded instruction
  return true;
}

template <lds_writeback::DstFormat F>
[[gnu::always_inline]] inline bool
LdsInlineAccess::load(const Instruction &inst, const LdsAccessRequest &req, Wavefront &wf) {
  // Everything down to the reader's strategies is inlined, so the compiler sees
  // the whole path: the group pointers stay in registers, and a group read in
  // place goes from LDS to the destination VGPRs without passing through
  // staging.
  alignas(64) uint8_t staging[64 * kMaxLaneBytes];
  LdsStagedLoad staged;
  if (!access<true>(inst, req, wf, staging, staged))
    return false;
  lds_writeback::write_back_as<F>(staged.dst, staged.sram_ecc, staged.wave, staged.exec,
                                  staged.group);
  return true;
}

} // namespace amdgpu
} // namespace rocjitsu
