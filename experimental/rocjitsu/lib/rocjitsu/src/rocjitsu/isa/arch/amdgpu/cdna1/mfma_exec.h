// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_CDNA1_MFMA_EXEC_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_CDNA1_MFMA_EXEC_H_

/// @file MFMA execution stubs for CDNA1 (GFX908, MI100).
///
/// Phase A stub — all MFMA execute() bodies in the generated vop3p.cpp
/// throw util::UnimplementedInst, so the functions declared here are never
/// called. Phase B will provide correct CDNA1 MFMA layouts (AccVGPR
/// encoding base 256, same register-dimension math as CDNA3 but different
/// accumulator encoding).

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "util/data_types.h"

#include <bit>
#include <cstdint>
#include <vector>

namespace rocjitsu {
namespace cdna1 {
namespace mfma {

struct InputLoc {
  uint32_t vgpr_offset;
  uint32_t lane;
  uint32_t sub_element;
};

struct OutputLoc {
  uint32_t reg;
  uint32_t lane;
};

inline uint32_t dst_base(uint32_t vb, int ev) { return vb + static_cast<uint32_t>(ev); }
inline uint32_t src_base(uint32_t vb, int ev) { return vb + static_cast<uint32_t>(ev); }
inline uint32_t resolve_acc(uint32_t, uint32_t dst, int, bool, uint32_t) { return dst; }

} // namespace mfma
} // namespace cdna1
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_CDNA1_MFMA_EXEC_H_
