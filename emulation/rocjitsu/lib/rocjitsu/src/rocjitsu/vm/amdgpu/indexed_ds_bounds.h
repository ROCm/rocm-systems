// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "util/bit.h"

#include <algorithm>
#include <cstdint>
#include <optional>

namespace rocjitsu::amdgpu {

struct IndexedDsAccessBounds {
  uint64_t begin;
  uint64_t end;
  uint64_t bytes;

  [[nodiscard]] bool contains(uint64_t address) const {
    return address >= begin && util::is_range_in_bounds(address, bytes, end);
  }
};

// Bounds for architectures that check the complete native DS transfer at each
// address. A dual instruction supplies two independent addresses.
inline bool checks_entire_native_ds_address(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA4 ||
         arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2 ||
         arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5;
}

inline std::optional<IndexedDsAccessBounds> indexed_ds_access_bounds(const VectorMemState &state,
                                                                     rj_code_arch_t arch,
                                                                     uint32_t lds_base,
                                                                     uint32_t lds_size) {
  if (!checks_entire_native_ds_address(arch) || state.flat_local_lane_mask || state.gds ||
      (state.gds_m0 && arch != ROCJITSU_CODE_ARCH_RDNA1))
    return std::nullopt;
  const uint64_t begin = state.ds_base(lds_base);
  const uint64_t end = state.gds_m0 ? std::min<uint64_t>(65536, begin + (*state.gds_m0 & 0xffffu))
                                    : begin + lds_size;
  return IndexedDsAccessBounds{begin, end, uint64_t{state.elem_size} * state.num_elems};
}

} // namespace rocjitsu::amdgpu
