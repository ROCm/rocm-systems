// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/bit.h"

#include <algorithm>
#include <cstdint>
#include <optional>

namespace rocjitsu::amdgpu {

inline bool cdna1_ds_atomic_nullified(const VectorMemState &d, rocjitsu::IsaFamily arch,
                                      const RegAllocation &vgprs) {
  if (arch != rocjitsu::IsaFamily::Cdna1 || d.flat_local_lane_mask || !d.is_load ||
      d.atomic_op == AtomicOp::NONE) {
    return false;
  }
  // MI100 3.6.1 nullifies the RMW when any returned VGPR is unallocated.
  return !vgprs.contains(d.dst_reg_base, d.destination_vgpr_count()) ||
         (d.ds2_active && !vgprs.contains(d.ds2_dst_reg_base, d.ds2_destination_vgpr_count()));
}

class IndexedDsAccessBounds {
public:
  uint64_t begin;
  uint64_t end;
  uint64_t bytes;

  [[nodiscard]] bool contains(uint64_t address) const {
    return address >= begin && util::is_range_in_bounds(address, bytes, end);
  }
};

// CDNA1 through CDNA4 discard out-of-range LDS writes and zero out-of-range
// reads. Check the whole transfer at each address, including when the allocation
// size is not a multiple of the write size and the manuals leave writes undefined.
// A dual instruction supplies two independent addresses.
/// \NPI Qualify workgroup addressing and bounds when adding an ISA.
inline bool checks_entire_native_ds_address(rocjitsu::IsaFamily arch) {
  return arch == rocjitsu::IsaFamily::Cdna1 || arch == rocjitsu::IsaFamily::Cdna2 ||
         arch == rocjitsu::IsaFamily::Cdna3 || arch == rocjitsu::IsaFamily::Cdna4 ||
         arch == rocjitsu::IsaFamily::Rdna1 || arch == rocjitsu::IsaFamily::Rdna2 ||
         arch == rocjitsu::IsaFamily::Rdna3 || arch == rocjitsu::IsaFamily::Rdna3_5;
}

inline std::optional<IndexedDsAccessBounds> indexed_ds_access_bounds(const VectorMemState &state,
                                                                     rocjitsu::IsaFamily arch,
                                                                     uint32_t lds_base,
                                                                     uint32_t lds_size) {
  if (!checks_entire_native_ds_address(arch) || state.flat_local_lane_mask || state.gds ||
      (state.gds_m0 && arch != rocjitsu::IsaFamily::Rdna1)) {
    return std::nullopt;
  }
  const uint64_t begin = state.ds_base(lds_base);
  const uint64_t end = state.gds_m0 ? std::min<uint64_t>(65536, begin + (*state.gds_m0 & 0xffffu))
                                    : begin + lds_size;
  return IndexedDsAccessBounds{begin, end, uint64_t{state.elem_size} * state.num_elems};
}

} // namespace rocjitsu::amdgpu
