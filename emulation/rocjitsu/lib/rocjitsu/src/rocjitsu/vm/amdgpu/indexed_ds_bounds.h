// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/bit.h"

#include <algorithm>
#include <cstdint>
#include <optional>

namespace rocjitsu::amdgpu {

inline bool cdna1_ds_atomic_nullified(const VectorMemState &d, rj_code_arch_t arch,
                                      const RegAllocation &vgprs) {
  if (arch != ROCJITSU_CODE_ARCH_CDNA1 || d.flat_local_lane_mask || !d.is_load ||
      d.atomic_op == AtomicOp::NONE)
    return false;
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
inline bool checks_entire_native_ds_address(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
         arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4 ||
         arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2 ||
         arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5;
}

inline std::optional<IndexedDsAccessBounds> indexed_ds_access_bounds(const VectorMemState &state,
                                                                     rj_code_arch_t arch,
                                                                     uint32_t lds_base,
                                                                     uint32_t lds_size) {
  if (!checks_entire_native_ds_address(arch) || state.flat_local_lane_mask)
    return std::nullopt;
  const uint64_t begin = lds_base;
  const uint64_t end = begin + lds_size;
  return IndexedDsAccessBounds{begin, end, uint64_t{state.elem_size} * state.num_elems};
}

} // namespace rocjitsu::amdgpu
