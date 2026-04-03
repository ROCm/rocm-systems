// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_CDNA2_MFMA_EXEC_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_CDNA2_MFMA_EXEC_H_

/// @file MFMA execution stubs for CDNA2 (GFX90A, MI200).
///
/// Phase A stub — all MFMA execute() bodies in the generated vop3p.cpp
/// throw util::UnimplementedInst, so the functions declared here are never
/// called. Phase B will provide correct CDNA2 MFMA layouts (AccVGPR
/// encoding base 512).

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "util/data_types.h"

#include <bit>
#include <cstdint>
#include <vector>

namespace rocjitsu {
namespace cdna2 {
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
} // namespace cdna2
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_CDNA2_MFMA_EXEC_H_
